"""
model_loader.py
------------------
Loads a static model (v1/v2) and, if present, a motion sequence model —
mirrors the logic in asl_recognizer_hybrid.py's HybridRecognizer, minus the
live-webcam/UI parts (this only needs predict_static/predict_motion, called
per-request instead of per-frame in a cv2 loop).

Reuses sequence_data.py and gesture_trainer_v2.py DIRECTLY (imported, not
reimplemented) — copy those two files into this folder unchanged. This is
deliberate: motion normalization and v2's feature engineering must stay
byte-for-byte identical to what the models were actually trained with, so
importing the real files eliminates any risk of silently drifting from them.
"""

import json
from pathlib import Path

import numpy as np

import sequence_data as sd


def _load_label_map(path: Path) -> dict:
    raw = json.loads(Path(path).read_text())
    return {int(k): v for k, v in raw.items()}


class HybridModel:
    def __init__(
        self,
        static_model_dir: Path,
        static_type: str = "v2",
        seq_model_dir: Path | None = None,
    ):
        static_model_dir = Path(static_model_dir)
        import tensorflow as tf

        # ── Static model ────────────────────────────────────────────────────
        if static_type == "v2":
            from gesture_trainer_v2 import build_feature_vector
            self._static_model = tf.keras.models.load_model(
                str(static_model_dir / "asl_model_v2.keras")
            )
            self._static_features = build_feature_vector  # (63,) -> (93,) or (88,)
        elif static_type == "v1":
            self._static_model = tf.keras.models.load_model(
                str(static_model_dir / "asl_model.keras")
            )
            self._static_features = lambda lm: lm  # raw (63,)
        else:
            raise ValueError(f"Unknown static_type: {static_type!r}")

        self._static_labels = _load_label_map(static_model_dir / "label_map.json")
        print(f"[MODEL]  Static {static_type} model loaded from {static_model_dir}")

        # ── Motion (sequence) model — optional ─────────────────────────────
        self._seq_model = None
        self._window = sd.DEFAULT_WINDOW
        self._use_deltas = False

        if seq_model_dir is not None:
            seq_model_dir = Path(seq_model_dir)
            seq_path = seq_model_dir / "asl_seq_model.keras"
            if seq_path.exists():
                self._seq_model = tf.keras.models.load_model(str(seq_path))
                self._seq_labels = _load_label_map(seq_model_dir / "label_map.json")
                meta = json.loads((seq_model_dir / "seq_meta.json").read_text())
                self._window = int(meta.get("window", sd.DEFAULT_WINDOW))
                self._use_deltas = bool(meta.get("use_deltas", False))
                print(
                    f"[MODEL]  Motion seq model loaded from {seq_model_dir} "
                    f"(window={self._window}, deltas={self._use_deltas}, "
                    f"classes={list(self._seq_labels.values())})"
                )
            else:
                print(f"[INFO]  No seq model at '{seq_path}' — motion prediction unavailable.")

    @property
    def motion_available(self) -> bool:
        return self._seq_model is not None

    @property
    def window(self) -> int:
        return self._window

    def predict_static(self, landmarks: list[float]) -> tuple[str, float]:
        raw = np.array(landmarks, dtype=np.float32)
        x = np.asarray(self._static_features(raw)).reshape(1, -1).astype(np.float32)
        probs = self._static_model(x, training=False).numpy()[0]
        i = int(np.argmax(probs))
        return self._static_labels.get(i, "?"), float(probs[i])

    def predict_motion(self, sequence: list[list[float]]) -> tuple[str, float]:
        if self._seq_model is None:
            raise ValueError("No sequence model loaded — motion prediction unavailable.")

        seq = np.array(sequence, dtype=np.float32)  # (window, 63) raw
        norm = sd.normalize_sequence(seq)
        if self._use_deltas:
            norm = sd.add_deltas(norm)

        probs = self._seq_model(norm[None, ...].astype(np.float32), training=False).numpy()[0]
        i = int(np.argmax(probs))
        letter = self._seq_labels.get(i, "?")
        confidence = float(probs[i])

        # Reject classes (e.g. "_none") mean "this movement isn't a sign" —
        # same handling as HybridRecognizer._finalize().
        if letter.startswith("_"):
            return "", confidence
        return letter, confidence
