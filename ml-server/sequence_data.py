"""
sequence_data.py
----------------
Shared sequence utilities for the MOTION pipeline (letters traced through the
air, e.g. J and Z). Static letters are handled by the single-frame MLP; this
module handles anything whose meaning lives in the hand's *trajectory*.

Data expectations
-----------------
Motion samples are collected with `data_collector.py --motion`, which saves the
RAW (un-wrist-centered) landmark array per frame — see
HandDetector.extract_raw_landmark_array(). Files keep the same
`sample_s{session}_f{frame}.npy` naming, so each recording session is one
continuous trace, and frame_index gives the temporal order within it.

Why a per-SEQUENCE normalization (not per-frame)
------------------------------------------------
The static pipeline re-anchors the wrist to the origin every frame, which
erases where the hand *moved*. For J/Z that movement IS the sign. So here we
anchor once — to the first frame's wrist — and scale by the first frame's hand
size, then leave every later frame expressed relative to that origin. The
result keeps the wrist's path across the window intact while staying invariant
to where in the camera frame the sign started and how big the hand is.
"""

import re
from pathlib import Path

import numpy as np

FEATURES_PER_FRAME = 63    # 21 landmarks × (x, y, z), raw (un-centered)
WRIST      = 0
MIDDLE_MCP = 9

# Default temporal window: ~1 second at ~30 fps. Tune per how fast signs are made.
DEFAULT_WINDOW = 30
DEFAULT_STRIDE = 8

SESSION_FILENAME_PATTERN = re.compile(r"sample_s(\d+)_f(\d+)\.npy")


# ═══════════════════════════════════════════════════════════════════════════════
# Loading ordered sequences
# ═══════════════════════════════════════════════════════════════════════════════

def load_sequences(data_root: Path):
    """
    Walk data_root/<letter>/ and return one entry per recording SESSION:

        sequences  : list of (T, 63) float32 arrays, frames in temporal order
        labels     : list of int label indices (parallel to sequences)
        session_ids: list of session keys "<letter>_s<id>" (parallel)
        label_map  : {label_index: letter}

    A session with too few frames to be useful is still returned; windowing
    (make_windows) pads short sessions up to one window.
    """
    data_root = Path(data_root)
    letters = sorted(d.name for d in data_root.iterdir() if d.is_dir()) if data_root.exists() else []

    sequences, labels, session_ids, label_map = [], [], [], {}
    for label_index, letter in enumerate(letters):
        letter_dir = data_root / letter
        by_session = _group_session_frames(letter_dir)
        if not by_session:
            continue
        label_map[label_index] = letter
        for session_id, frames in by_session.items():
            sequences.append(np.stack(frames).astype(np.float32))   # (T, 63)
            labels.append(label_index)
            session_ids.append(f"{letter}_s{session_id}")

    if not sequences:
        raise ValueError(
            f"No motion sequences found under '{data_root}'.\n"
            "Collect some first, e.g.:  python data_collector.py --motion"
        )
    return sequences, labels, session_ids, label_map


def _group_session_frames(letter_dir: Path) -> dict:
    """
    Return {session_id: [frame_arr, ...]} with frames sorted by frame_index,
    so each session's list is a temporally-ordered trace. Files that don't
    match the session pattern or have the wrong shape are skipped.
    """
    if not letter_dir.exists():
        return {}

    # session_id -> list of (frame_index, array)
    buckets: dict[int, list] = {}
    for path in letter_dir.glob("sample_s*_f*.npy"):
        match = SESSION_FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        arr = np.load(path).astype(np.float32)
        if arr.shape != (FEATURES_PER_FRAME,):
            continue
        session_id  = int(match.group(1))
        frame_index = int(match.group(2))
        buckets.setdefault(session_id, []).append((frame_index, arr))

    ordered = {}
    for session_id, items in buckets.items():
        items.sort(key=lambda t: t[0])           # by frame_index
        ordered[session_id] = [arr for _, arr in items]
    return ordered


# ═══════════════════════════════════════════════════════════════════════════════
# Motion-preserving normalization + windowing
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Normalize a raw (T, 63) sequence ONCE, anchored to the first frame:
      - subtract the first frame's wrist position from every landmark of every
        frame  → the whole trajectory becomes relative to where the sign began
        (later frames' wrist ≠ 0 exactly when the hand moved — that's the point)
      - divide by the first frame's hand size (wrist→middle-MCP distance)
        → scale invariance without flattening the motion
    """
    coords = seq.reshape(len(seq), 21, 3).copy()
    anchor = coords[0, WRIST].copy()             # (3,)
    coords -= anchor                             # broadcast over T and landmarks

    scale = np.linalg.norm(coords[0, MIDDLE_MCP])
    if scale > 1e-6:
        coords /= scale
    return coords.reshape(len(seq), FEATURES_PER_FRAME)


def feature_dim(use_deltas: bool = False) -> int:
    """Per-frame feature width: 63 positions, doubled to 126 when deltas are on."""
    return FEATURES_PER_FRAME * (2 if use_deltas else 1)


def add_deltas(seq: np.ndarray) -> np.ndarray:
    """
    Append per-frame velocity to a normalized (T, 63) sequence → (T, 126).

    The position channels say WHERE the hand is; the delta channels say where
    it's GOING and how fast. For motion letters the direction/speed of travel is
    the discriminative signal (J hooks down-and-back, Z cuts across), and making
    it an explicit input beats asking the model to infer it from raw positions —
    especially with few training traces. Frame 0 has no predecessor, so its
    delta is zero.
    """
    deltas = np.zeros_like(seq)
    deltas[1:] = np.diff(seq, axis=0)
    return np.concatenate([seq, deltas], axis=1)


def make_windows(seq: np.ndarray, window: int = DEFAULT_WINDOW,
                 stride: int = DEFAULT_STRIDE) -> list:
    """
    Slice a (T, 63) sequence into fixed-length (window, 63) clips.

    - T >= window : sliding windows every `stride` frames (plus a final window
                    flush to the end so the tail isn't dropped).
    - T <  window : one window, left-padded by repeating the first frame.
    """
    T = len(seq)
    if T < window:
        pad = np.repeat(seq[:1], window - T, axis=0)
        return [np.concatenate([pad, seq], axis=0)]

    starts = list(range(0, T - window + 1, stride))
    if starts[-1] != T - window:
        starts.append(T - window)
    return [seq[s:s + window] for s in starts]


# ═══════════════════════════════════════════════════════════════════════════════
# Session-level, letter-stratified split → windowed tensors
# ═══════════════════════════════════════════════════════════════════════════════

def split_sessions(labels, session_ids, val_frac=0.15, test_frac=0.10, seed=42):
    """
    Assign whole sessions to train/val/test, stratified by letter (same policy
    as the static trainers) so windows from one trace never leak across splits.
    Returns three sets of session_id keys.
    """
    labels = np.asarray(labels)
    session_ids = np.asarray(session_ids)
    rng = np.random.default_rng(seed)

    train, val, test = set(), set(), set()
    for label in np.unique(labels):
        sess = np.unique(session_ids[labels == label])
        shuffled = rng.permutation(sess)
        n = len(shuffled)
        train_end = int(n * (1 - val_frac - test_frac))
        val_end   = int(n * (1 - test_frac))

        # With few sessions the proportional split can hand a class an EMPTY val
        # or test slice (e.g. n=3 → int(2.25)=2 and int(2.7)=2 → val is [2:2]).
        # That fails silently and badly: a val set missing a class makes
        # val_accuracy maximal for a model that never predicts it, so
        # EarlyStopping(restore_best_weights) actively selects the most biased
        # checkpoint. Guarantee one session each once a class has 3+ to spare.
        if n >= 3:
            train_end = min(train_end, n - 2)
            val_end   = min(max(val_end, train_end + 1), n - 1)

        train.update(shuffled[:train_end])
        val.update(shuffled[train_end:val_end])
        test.update(shuffled[val_end:])
    return train, val, test


def build_windowed_dataset(data_root: Path, window=DEFAULT_WINDOW,
                           stride=DEFAULT_STRIDE, seed=42, use_deltas=False):
    """
    Full motion-dataset build:
        load ordered sequences → split by session (stratified) → normalize each
        sequence → optionally append velocity → cut into windows.

    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test), label_map
    where each X is (N, window, feature_dim(use_deltas)) float32.

    Deltas are computed on the whole normalized sequence BEFORE windowing, so a
    window's first frame still carries its true velocity from the preceding
    frame rather than a fabricated zero.
    """
    sequences, labels, session_ids, label_map = load_sequences(data_root)
    train_s, val_s, test_s = split_sessions(labels, session_ids, seed=seed)
    dim = feature_dim(use_deltas)

    def collect(session_set):
        X, y = [], []
        for seq, label, sid in zip(sequences, labels, session_ids):
            if sid not in session_set:
                continue
            norm = normalize_sequence(seq)
            if use_deltas:
                norm = add_deltas(norm)
            for win in make_windows(norm, window, stride):
                X.append(win)
                y.append(label)
        if not X:
            return np.empty((0, window, dim), np.float32), np.empty((0,), np.int32)
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)

    return collect(train_s), collect(val_s), collect(test_s), label_map
