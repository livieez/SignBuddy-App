"""
reporting.py
------------
Shared evaluation reporting used by the trainers. Currently: a confusion-matrix
heatmap + per-letter classification report + the most-confused letter pairs, so
you can see WHICH letters are dragging accuracy down instead of guessing from a
single aggregate number.

scikit-learn is imported lazily and treated as optional — if it isn't installed
the report is skipped with a note rather than crashing a training run.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, ConfusionMatrixDisplay
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def save_confusion_report(y_true, y_pred, label_map: dict, model_dir: Path, title: str):
    """
    Save `confusion_matrix.png` + `classification_report.txt` into model_dir and
    print the biggest off-diagonal confusions (true → predicted).

    Parameters
    ----------
    y_true, y_pred : contiguous 0..K-1 integer labels for the test split.
    label_map      : {contiguous_index: letter}.
    model_dir      : folder to write the artifacts into.
    title          : plot title, e.g. "ASL MLP v2".
    """
    model_dir = Path(model_dir)
    labels = sorted(label_map)
    class_names = [label_map[i] for i in labels]

    if not _SKLEARN:
        print("[CONFUSION]  scikit-learn not installed — skipping confusion "
              "matrix (pip install scikit-learn to enable).")
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45, values_format="d"
    )
    ax.set_title(f"{title} — Confusion Matrix (test set)")
    plt.tight_layout()
    cm_path = model_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
    )
    (model_dir / "classification_report.txt").write_text(report)

    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)

    # Plain ASCII arrows here — some Windows consoles use cp1252 and choke on
    # the unicode arrow, and this print must never crash a training run.
    print(f"[CONFUSION]  Heatmap -> {cm_path}")
    print(f"[CONFUSION]  Report  -> {model_dir / 'classification_report.txt'}")
    if confusions:
        print("[CONFUSION]  Most-confused pairs (true -> predicted):")
        for count, true_l, pred_l in confusions[:8]:
            print(f"             {true_l} -> {pred_l}: {count}")
