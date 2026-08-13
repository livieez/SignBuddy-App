"""
gesture_trainer_v2.py
---------------------
Upgraded MLP trainer for ASL landmark classification.
Implements all four recommended improvements over v1:

  Upgrade 1 — BatchNorm placement   : Dense → BN → ReLU → Dropout
  Upgrade 2 — Geometric features    : fingertip distances + joint angles
                                      appended to raw landmarks (63 → 93)
  Upgrade 3 — Smaller first layer   : 128 → 128 → 64 (vs 256 → 128 → 64)
  Upgrade 4 — Label smoothing       : CategoricalCrossentropy(smoothing=0.1)

Usage
-----
    python gesture_trainer_v2.py
    python gesture_trainer_v2.py --epochs 100 --label_smoothing 0.05

Output
------
    models/asl_model_v2.keras      — upgraded Keras model
    models/asl_model_v2.tflite     — TFLite export
    models/label_map.json          — {index: letter} (shared with v1 / RF)
    models/training_history_v2.png — loss & accuracy curves

Note on data splitting
-----------------------
The RAW landmarks are split into train/val/test BEFORE augmentation, and only
the train split is augmented. Geometric features are then computed separately
for each split. This guarantees val/test accuracy is measured on real
recorded samples only — never on synthetic jittered copies — so scores are
directly comparable against gesture_trainer.py and gesture_trainer_rf.py.
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from reporting import save_confusion_report

try:
    import tensorflow as tf
    keras = tf.keras
    layers = keras.layers
except ImportError:
    raise SystemExit("TensorFlow not found.\nInstall: pip install tensorflow")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Anchored to this file's folder so the scripts work no matter which
# directory you launch them from (src/ or the repo root).
SRC_DIR = Path(__file__).resolve().parent
DATA_ROOT      = SRC_DIR / "data"
MODEL_DIR      = SRC_DIR / "models" / "v2_model"
MODEL_PATH     = MODEL_DIR / "asl_model_v2.keras"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

# J and Z are motion signs (traced through the air), not static handshapes —
# a single-frame landmark snapshot can't represent them, so they're excluded
# for now. See MOTION_LETTERS if you later add sequence-based recognition.
MOTION_LETTERS = {"J", "Z"}
ASL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) not in MOTION_LETTERS]

FEATURES_PER_SAMPLE   = 63    # 21 landmarks × (x, y, z)
# 10 tip-tip + 5 tip-wrist + 15 joint angles (3 per finger). The 3rd angle per
# finger is the knuckle bend, obtained by anchoring each finger's bone chain at
# the WRIST (5-point chain → 3 interior triplets). Models trained with this take
# a 93-dim input, recorded in model_meta.json.
GEOMETRIC_FEATURE_DIM = 30    # 10 tip-tip + 5 tip-wrist + 15 joint angles
TOTAL_FEATURE_DIM     = FEATURES_PER_SAMPLE + GEOMETRIC_FEATURE_DIM   # 93

# Matches sample_s{session_id}_f{frame_index}.npy filenames written by the
# current data_collector.py. Older files saved as plain sample_N.npy (no
# session tag) won't match — see _load_letter_samples() for the fallback.
SESSION_FILENAME_PATTERN = re.compile(r"sample_s(\d+)_f\d+\.npy")

# ── MediaPipe landmark indices (21 landmarks) ─────────────────────────────────
WRIST       = 0
FINGERTIPS  = [4, 8, 12, 16, 20]   # thumb → pinky tips
FINGER_MCPS = [1, 5, 9, 13, 17]    # knuckle bases
# Bone chains per finger, anchored at the WRIST so each 5-point chain yields
# 3 interior joint angles (knuckle bend + two finger bends) → 15 angles total.
FINGER_BONES = [
    (0, 1, 2, 3, 4),      # thumb:  WRIST→CMC→MCP→IP→TIP
    (0, 5, 6, 7, 8),      # index:  WRIST→MCP→PIP→DIP→TIP
    (0, 9, 10, 11, 12),   # middle
    (0, 13, 14, 15, 16),  # ring
    (0, 17, 18, 19, 20),  # pinky
]


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 2 — Geometric feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

def compute_geometric_features(coords: np.ndarray) -> np.ndarray:
    """
    Given a (21, 3) normalized landmark array, compute:

      - 10 fingertip-to-fingertip distances   (all pairs of 5 tips)
      - 5  fingertip-to-wrist distances
      - 15 joint angles (3 angles per finger × 5 fingers)

    Returns a float32 array of shape (30,).

    Why each feature helps
    ----------------------
    Fingertip distances : directly encode finger spread and crossing
                          (e.g. R has crossed index+middle, U does not)
    Wrist distances     : encode how extended/curled each finger is
    Joint angles        : capture curl state per finger segment —
                          the dominant signal for distinguishing letters
                          that share finger positions but differ in curl
    """
    features = []
    features.extend(_fingertip_pair_distances(coords))
    features.extend(_fingertip_wrist_distances(coords))
    features.extend(_finger_joint_angles(coords))
    return np.array(features, dtype=np.float32)   # shape (30,)


def _fingertip_pair_distances(coords: np.ndarray) -> list[float]:
    """10 values: distance between every pair of the 5 fingertips."""
    return [
        np.linalg.norm(coords[i] - coords[j])
        for i, j in combinations(FINGERTIPS, 2)
    ]


def _fingertip_wrist_distances(coords: np.ndarray) -> list[float]:
    """5 values: distance from each fingertip back to the wrist."""
    return [np.linalg.norm(coords[tip] - coords[WRIST]) for tip in FINGERTIPS]


def _finger_joint_angles(coords: np.ndarray) -> list[float]:
    """
    15 values: 3 bend angles per finger (5 fingers × 3 interior joints).
    For each consecutive triplet (a, b, c) in a finger's 5-point bone chain:
        angle = arccos( (b-a)·(c-b) / (|b-a| |c-b|) )
    """
    angles = []
    for bone in FINGER_BONES:
        for k in range(len(bone) - 2):     # 3 triplets per 5-point chain
            a, b, c = coords[bone[k]], coords[bone[k + 1]], coords[bone[k + 2]]
            angles.append(_angle_between(b - a, c - b))
    return angles


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians [0, π] between two vectors; 0.0 if either is ~zero-length."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos_angle)


def build_feature_vector(raw_landmarks: np.ndarray) -> np.ndarray:
    """
    Combine raw (63,) landmark vector with (30,) geometric features → (93,).

    Input  : flat (63,) array from HandDetector.extract_landmark_array()
    Output : (93,) float32 feature vector
    """
    coords = raw_landmarks.reshape(21, 3)
    geometric = compute_geometric_features(coords)
    return np.concatenate([raw_landmarks, geometric])


def enrich_dataset(X_raw: np.ndarray) -> np.ndarray:
    """Apply build_feature_vector to every row in X_raw (N, 63) → (N, 93)."""
    return np.stack([build_feature_vector(row) for row in X_raw])


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading, splitting & augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_root: Path = DATA_ROOT):
    """
    Walk data_root/<letter>/*.npy and build raw X (N×63), y (N,), and a
    parallel session_ids (N,) array identifying which recording session each
    frame came from. Returns (X, y, session_ids, label_map).
    """
    features, labels, session_ids, label_map = [], [], [], {}

    for label_index, letter in enumerate(ASL_LETTERS):
        samples = _load_letter_samples(data_root / letter)
        if not samples:
            continue
        label_map[label_index] = letter
        for arr, session_id in samples:
            features.append(arr)
            labels.append(label_index)
            session_ids.append(session_id)

    if not features:
        raise ValueError(
            f"No samples found under '{data_root}'.\nRun data_collector.py first."
        )

    X = np.stack(features)
    y = np.array(labels, dtype=np.int32)
    sessions = np.array(session_ids)
    print(f"[DATA]  {len(X)} samples  |  {len(label_map)} classes  |  "
          f"{len(np.unique(sessions))} recording sessions")
    return X, y, sessions, label_map


def _load_letter_samples(letter_dir: Path) -> list[tuple]:
    """
    Load every valid sample_*.npy file in one letter's folder, paired with
    a session key identifying which recording session it came from.

    Files are named sample_s{session_id}_f{frame_index}.npy. Older files
    saved before session-tagging (plain sample_N.npy) are treated as their
    own singleton session each — a safe fallback, but re-recording with the
    current data_collector.py is recommended for proper session grouping.
    """
    if not letter_dir.exists():
        return []

    valid_samples = []
    for path in sorted(letter_dir.glob("sample_*.npy")):
        arr = np.load(path).astype(np.float32)
        if arr.shape != (FEATURES_PER_SAMPLE,):
            continue

        match = SESSION_FILENAME_PATTERN.match(path.name)
        if match:
            session_key = f"{letter_dir.name}_s{match.group(1)}"
        else:
            session_key = f"{letter_dir.name}_legacy_{path.stem}"

        valid_samples.append((arr, session_key))
    return valid_samples


def split_dataset(X, y, session_ids, val_frac: float = 0.15, test_frac: float = 0.10, seed: int = 42):
    """
    Split by whole RECORDING SESSION, stratified by letter, BEFORE any
    augmentation — so val/test always consist of real, unseen sessions,
    with every letter proportionally represented in each split.

    Why session-level, not frame-level: frames from the same session are
    near-duplicates of each other (same held hand pose, captured many times
    per second). Shuffling and splitting individual frames lets near-identical
    frames leak across train/val/test, inflating accuracy far beyond what the
    model actually achieves on a genuinely new sign. Splitting whole sessions
    guarantees every frame from one session lands entirely on one side.

    Why stratified by letter, not a flat pooled shuffle: pooling every
    letter's sessions together before shuffling can — purely by chance —
    leave some letters with zero sessions in val or test, or unevenly skew
    which letters are "easy" vs "hard" in each split. Splitting within each
    letter's own sessions first, then combining, guarantees every letter
    with enough sessions contributes proportionally to train/val/test.

    e.g. val_frac=0.15, test_frac=0.10 → ~75% train / ~15% val / ~10% test,
    measured in sessions per letter, not frames (so split sizes vary a bit
    per run, and a letter with very few sessions may still end up with 0 in
    val or test — see the [SPLIT] warning printed by train() for this).
    """
    rng = np.random.default_rng(seed)
    train_sessions, val_sessions, test_sessions = set(), set(), set()

    for label in np.unique(y):
        letter_mask = (y == label)
        letter_sessions = np.unique(session_ids[letter_mask])
        shuffled = rng.permutation(letter_sessions)

        n = len(shuffled)
        train_end = int(n * (1 - val_frac - test_frac))
        val_end   = int(n * (1 - test_frac))

        train_sessions.update(shuffled[:train_end])
        val_sessions.update(shuffled[train_end:val_end])
        test_sessions.update(shuffled[val_end:])

    train_mask = np.isin(session_ids, list(train_sessions))
    val_mask   = np.isin(session_ids, list(val_sessions))
    test_mask  = np.isin(session_ids, list(test_sessions))

    train_set = (X[train_mask], y[train_mask])
    val_set   = (X[val_mask], y[val_mask])
    test_set  = (X[test_mask], y[test_mask])
    return train_set, val_set, test_set


def report_split_coverage(y_train, y_val, y_test, label_map: dict):
    """
    Print how many letters ended up with zero val or test samples (possible
    with a small number of recorded sessions), so this is visible immediately
    rather than silently skewing the reported accuracy.
    """
    missing_val, missing_test = [], []
    for label, letter in label_map.items():
        in_val  = np.any(y_val == label)
        in_test = np.any(y_test == label)
        if not in_val:
            missing_val.append(letter)
        if not in_test:
            missing_test.append(letter)

    if missing_val:
        print(f"[SPLIT WARN]  Letters with 0 samples in val: {sorted(missing_val)}")
    if missing_test:
        print(f"[SPLIT WARN]  Letters with 0 samples in test: {sorted(missing_test)}")
    if not missing_val and not missing_test:
        print("[SPLIT]  Every letter has at least one session in both val and test.")


def augment(X: np.ndarray, y: np.ndarray, factor: int = 3, seed: int = 42) -> tuple:
    """
    Jitter + scale augmentation, applied to raw (63,) landmarks so geometric
    features get recomputed cleanly from the noisy copies afterward.

    Only ever call this on a TRAIN split — never on val/test — so evaluation
    always reflects real recorded samples.
    """
    rng = np.random.default_rng(seed)
    augmented_X, augmented_y = [X], [y]
    for _ in range(factor - 1):
        noise = rng.normal(0, 0.01, X.shape).astype(np.float32)
        scale = rng.uniform(0.95, 1.05, (X.shape[0], 1)).astype(np.float32)
        augmented_X.append((X + noise) * scale)
        augmented_y.append(y)
    return np.concatenate(augmented_X), np.concatenate(augmented_y)


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 1 + 3 + 4 — Improved model architecture
# ═══════════════════════════════════════════════════════════════════════════════

def mlp_block(x, units: int, dropout: float):
    """
    UPGRADE 1 — Pre-activation BatchNorm block:
        Dense (linear) → BatchNorm → ReLU → Dropout

    Why this order beats Dense→ReLU→BN→Dropout:
    BN normalises the pre-activation values so ReLU sees a zero-centred,
    unit-variance distribution, reducing dead neurons and stabilising
    gradient flow — especially important in early training epochs.
    """
    x = layers.Dense(units, use_bias=False)(x)   # bias redundant with BN
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_model(
    num_classes: int,
    input_dim: int = TOTAL_FEATURE_DIM,
    label_smoothing: float = 0.1,
) -> keras.Model:
    """
    UPGRADE 3 — Smaller first layer (128 → 128 → 64):
    With structured landmark + geometric features the first layer doesn't
    need to learn raw spatial relationships from scratch — 128 units carry
    enough capacity while reducing overfit risk on smaller datasets.

    UPGRADE 4 — Label smoothing via CategoricalCrossentropy:
    Softens one-hot targets from [0,…,1,…,0] to
    [ε/K, …, 1−ε(K-1)/K, …, ε/K].
    Penalises over-confident predictions on visually ambiguous pairs
    (M/N, R/U, G/H), improving calibration and generalisation.
    Requires one-hot y — see encode_labels() below.
    """
    inputs = keras.Input(shape=(input_dim,), name="landmarks_and_geo")

    x = mlp_block(inputs, units=128, dropout=0.3)
    x = mlp_block(x,      units=128, dropout=0.2)
    x = mlp_block(x,      units=64,  dropout=0.1)

    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)

    model = keras.Model(inputs, outputs, name="asl_mlp_v2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    model.summary()
    return model


def build_callbacks() -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Label encoding — sparse label_map indices aren't necessarily contiguous
# (a letter with zero recorded samples leaves a gap), so remap to 0..K-1.
# ═══════════════════════════════════════════════════════════════════════════════

def build_contiguous_label_mapping(label_map: dict) -> dict:
    """{original_label_index: contiguous_0_to_K_minus_1_index}"""
    return {original: contiguous for contiguous, original in enumerate(sorted(label_map))}


def encode_labels(y, idx_to_class: dict, num_classes: int):
    """Remap sparse label indices to 0..K-1, then one-hot encode."""
    y_contiguous = np.array([idx_to_class[i] for i in y], dtype=np.int32)
    y_one_hot = tf.keras.utils.to_categorical(y_contiguous, num_classes)
    return y_contiguous, y_one_hot


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    epochs: int = 80,
    batch_size: int = 64,
    augment_factor: int = 3,
    label_smoothing: float = 0.1,
    data_root: Path = DATA_ROOT,
    model_dir: Path = MODEL_DIR,
    seed: int = 42,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(seed)

    # ── 1. Load raw landmarks ─────────────────────────────────────────────────
    X_raw, y, session_ids, label_map = load_dataset(data_root)
    num_classes = len(label_map)

    # ── 2. Split RAW landmarks by SESSION, STRATIFIED BY LETTER — val/test are
    #        real, unseen recordings, with every letter proportionally
    #        represented ────────────────────────────────────────────────────────
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = \
        split_dataset(X_raw, y, session_ids, seed=seed)
    print(f"[SPLIT]  train={len(X_train_raw)}  val={len(X_val_raw)}  "
          f"test={len(X_test_raw)}  (split by session, stratified by letter — "
          f"val/test are real, unseen sessions)")
    report_split_coverage(y_train, y_val, y_test, label_map)

    # ── 3. Augment ONLY the training split (still raw, pre-geometry) ─────────
    X_train_raw, y_train = augment(X_train_raw, y_train, factor=augment_factor, seed=seed)
    print(f"[AUGMENT]  train expanded to {len(X_train_raw)} samples "
          f"(factor={augment_factor})")

    # ── 4. UPGRADE 2 — expand each split to (93,) with geometric features ────
    print("[FEAT]  Computing geometric features …")
    X_train = enrich_dataset(X_train_raw)
    X_val   = enrich_dataset(X_val_raw)
    X_test  = enrich_dataset(X_test_raw)
    print(f"[FEAT]  Feature vector: {FEATURES_PER_SAMPLE} raw + "
          f"{GEOMETRIC_FEATURE_DIM} geometric = {X_train.shape[1]} dims")

    # ── 5. One-hot encode for label smoothing (UPGRADE 4) ─────────────────────
    idx_to_class = build_contiguous_label_mapping(label_map)
    y_train_c, y_train_oh = encode_labels(y_train, idx_to_class, num_classes)
    y_val_c,   y_val_oh   = encode_labels(y_val,   idx_to_class, num_classes)
    y_test_c,  _          = encode_labels(y_test,  idx_to_class, num_classes)

    # ── 6. Build & train ────────────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        input_dim=X_train.shape[1],
        label_smoothing=label_smoothing,
    )
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=build_callbacks(),
        verbose=1,
    )

    # ── 7. Evaluate on untouched test split ─────────────────────────────────────
    # Sparse labels for eval — label smoothing only affects training loss.
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    test_accuracy = np.mean(y_pred == y_test_c)
    print(f"\n[TEST]  accuracy = {test_accuracy:.4f} ({test_accuracy * 100:.1f}%)")

    # ── 7b. Confusion matrix — shows WHICH letters get mixed up ──────────────
    remapped_label_map = {new: label_map[orig] for orig, new in idx_to_class.items()}
    save_confusion_report(y_test_c, y_pred, remapped_label_map, model_dir, title="ASL MLP v2")

    # ── 8. Save everything ─────────────────────────────────────────────────────
    _save_model_artifacts(model, remapped_label_map, X_train.shape[1], num_classes, model_dir, seed)
    _plot_history(history, model_dir / "training_history_v2.png")

    return model, remapped_label_map


def _save_model_artifacts(model, label_map: dict, input_dim: int, num_classes: int, model_dir: Path, seed: int):
    model.save(model_dir / "asl_model_v2.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(model_dir / "asl_model_v2.tflite", "wb") as f:
        f.write(tflite_model)

    with open(model_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Records feature-vector size (93, not 63) and the seed used, so a live
    # recognizer knows the expected input shape and this run is traceable
    # even when --seed was randomly generated rather than passed explicitly.
    meta = {"input_dim": input_dim, "num_classes": num_classes, "seed": seed}
    with open(model_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE]  Model   → {model_dir / 'asl_model_v2.keras'}")
    print(f"[SAVE]  TFLite  → {model_dir / 'asl_model_v2.tflite'}")
    print(f"[SAVE]  Labels  → {model_dir / 'label_map.json'}")
    print(f"[SAVE]  Meta    → {model_dir / 'model_meta.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_history(history, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("ASL MLP v2 — Training History", fontsize=13)

    axes[0].plot(history.history["loss"],     label="train", lw=2)
    axes[0].plot(history.history["val_loss"], label="val",   lw=2)
    axes[0].set_title("Loss (label-smoothed CE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="train", lw=2)
    axes[1].plot(history.history["val_accuracy"], label="val",   lw=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[PLOT]  Training curves → {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train upgraded ASL MLP v2")
    parser.add_argument("--epochs",          type=int,   default=80)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--augment_factor",  type=int,   default=3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--data_root",       type=Path,  default=DATA_ROOT)
    parser.add_argument("--model_dir",       type=Path,  default=MODEL_DIR)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the split/augmentation/model init. If omitted, "
             "a random seed is generated each run and printed to the console "
             "— note it down if you want to reproduce or compare this exact run later."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        print(f"[SEED]  No --seed given — using randomly generated seed={seed}")

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment_factor=args.augment_factor,
        label_smoothing=args.label_smoothing,
        data_root=args.data_root,
        model_dir=args.model_dir,
        seed=seed,
    )
