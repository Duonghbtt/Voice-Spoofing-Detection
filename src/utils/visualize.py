from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def _ensure_parent_dir(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _as_prediction_dataframe(dataframe_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(dataframe_or_path, pd.DataFrame):
        return dataframe_or_path.copy()
    path = Path(dataframe_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not locate prediction CSV: {path}")
    return pd.read_csv(path)


def load_predictions_dataframe(dataframe_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load prediction outputs from a dataframe or CSV path."""

    dataframe = _as_prediction_dataframe(dataframe_or_path)
    required_columns = {"utt_id", "label", "score_spoof", "pred_label"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction data must contain columns {sorted(required_columns)}, "
            f"missing {sorted(missing_columns)}"
        )
    return dataframe


def prediction_arrays(
    dataframe_or_path: pd.DataFrame | str | Path,
    label_column: str = "label",
    score_column: str = "score_spoof",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract label and score arrays from a prediction dataframe or CSV."""

    dataframe = _as_prediction_dataframe(dataframe_or_path)
    if label_column not in dataframe.columns or score_column not in dataframe.columns:
        raise ValueError(
            f"Prediction data must contain '{label_column}' and '{score_column}' columns, "
            f"got columns {dataframe.columns.tolist()}"
        )
    labels = dataframe[label_column].to_numpy(dtype=np.int64, copy=False)
    scores = dataframe[score_column].to_numpy(dtype=np.float64, copy=False)
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and scores must be 1D arrays")
    if len(labels) != len(scores):
        raise ValueError(f"labels and scores must have the same length, got {len(labels)} and {len(scores)}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contains NaN or Inf values, cannot plot")
    return labels, scores


def load_confusion_matrix(matrix_or_path: np.ndarray | str | Path) -> np.ndarray:
    """Load a confusion matrix from memory or a .npy file."""

    if isinstance(matrix_or_path, np.ndarray):
        matrix = np.asarray(matrix_or_path)
    else:
        path = Path(matrix_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not locate confusion matrix file: {path}")
        matrix = np.load(path)
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix, got shape {matrix.shape}")
    return matrix


def load_attack_wise_eer_dataframe(dataframe_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load attack-wise EER results from a dataframe or CSV path."""

    if isinstance(dataframe_or_path, pd.DataFrame):
        dataframe = dataframe_or_path.copy()
    else:
        path = Path(dataframe_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not locate attack-wise EER CSV: {path}")
        dataframe = pd.read_csv(path)
    required_columns = {"attack_id", "num_samples", "num_bonafide", "num_spoof", "eer", "eer_threshold", "valid"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Attack-wise EER data must contain columns {sorted(required_columns)}, "
            f"missing {sorted(missing_columns)}"
        )
    return dataframe


def save_learning_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    output_path: str | Path,
    title: str = "Learning Curve",
) -> None:
    path = _ensure_parent_dir(output_path)

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_figure(
    matrix: np.ndarray | str | Path,
    output_path: str | Path,
    title: str = "ASVspoof2019 Eval Confusion Matrix",
) -> None:
    path = _ensure_parent_dir(output_path)
    matrix_array = load_confusion_matrix(matrix)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix_array, cmap="Blues")
    ax.set_xticks([0, 1], labels=["bonafide", "spoof"])
    ax.set_yticks([0, 1], labels=["bonafide", "spoof"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for row in range(matrix_array.shape[0]):
        for col in range(matrix_array.shape[1]):
            ax.text(col, row, int(matrix_array[row, col]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roc_curve_figure(
    labels: Sequence[int],
    scores: Sequence[float],
    output_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    """Save a ROC curve figure for spoof detection scores."""

    path = _ensure_parent_dir(output_path)
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("labels and scores must be 1D")
    if len(label_array) != len(score_array):
        raise ValueError(f"labels and scores must have the same length, got {len(label_array)} and {len(score_array)}")
    if np.unique(label_array).size < 2:
        raise ValueError("ROC curve requires both bonafide and spoof samples")
    if not np.isfinite(score_array).all():
        raise ValueError("scores contains NaN or Inf values, cannot draw ROC curve")

    fpr, tpr, _ = roc_curve(label_array, score_array, pos_label=1)
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.0, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.2, color="#7f7f7f", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_score_histogram(
    labels: Sequence[int],
    scores: Sequence[float],
    output_path: str | Path,
    title: str = "Score Distribution",
) -> None:
    """Save a score histogram split by bonafide and spoof labels."""

    path = _ensure_parent_dir(output_path)
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("labels and scores must be 1D")
    if len(label_array) != len(score_array):
        raise ValueError(f"labels and scores must have the same length, got {len(label_array)} and {len(score_array)}")
    if not np.isfinite(score_array).all():
        raise ValueError("scores contains NaN or Inf values, cannot draw histogram")

    bonafide_scores = score_array[label_array == 0]
    spoof_scores = score_array[label_array == 1]
    if bonafide_scores.size == 0 or spoof_scores.size == 0:
        raise ValueError("Histogram requires both bonafide and spoof samples")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    bins = np.linspace(0.0, 1.0, 31)
    ax.hist(bonafide_scores, bins=bins, alpha=0.7, label="bonafide", color="#4c72b0", density=True)
    ax.hist(spoof_scores, bins=bins, alpha=0.6, label="spoof", color="#c44e52", density=True)
    ax.set_xlabel("Spoof Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_attack_wise_eer_figure(
    df: pd.DataFrame | str | Path,
    output_path: str | Path,
    title: str = "Attack-wise EER",
) -> None:
    """Save a bar chart of attack-wise EER values."""

    path = _ensure_parent_dir(output_path)
    dataframe = load_attack_wise_eer_dataframe(df)
    plot_df = dataframe.copy()
    plot_df["valid"] = plot_df["valid"].map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            1: True,
            0: False,
        }
    ).fillna(False)
    plot_df = plot_df[plot_df["valid"] & plot_df["eer"].notna()].copy()
    if plot_df.empty:
        raise ValueError("No valid attack-wise EER rows available for plotting")

    plot_df["eer"] = plot_df["eer"].astype(float)
    plot_df = plot_df.sort_values(by="eer", ascending=False).reset_index(drop=True)

    fig_height = max(4.0, 0.45 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    bars = ax.barh(plot_df["attack_id"], plot_df["eer"], color="#c44e52", alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("EER")
    ax.set_ylabel("Attack ID")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            float(bar.get_width()) + 0.005,
            bar.get_y() + bar.get_height() / 2.0,
            f"{row['eer']:.3f} (n={int(row['num_spoof'])})",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
