from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve


@dataclass
class EvaluationArtifacts:
    loss: float | None
    accuracy: float
    eer: float
    eer_threshold: float
    confusion: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    utt_ids: list[str]


def logits_to_spoof_scores(logits: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(logits, np.ndarray):
        array = logits
    else:
        array = torch.softmax(logits, dim=1).detach().cpu().numpy()
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected logits/probabilities with shape (N, 2), got {array.shape}")
    return array[:, 1].astype(np.float32)


def compute_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    return float(accuracy_score(labels, predictions))


def compute_eer(labels: Sequence[int], spoof_scores: Sequence[float]) -> tuple[float, float]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(labels, spoof_scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    threshold = float(thresholds[index])
    return eer, threshold


def compute_confusion(labels: Sequence[int], predictions: Sequence[int]) -> np.ndarray:
    return confusion_matrix(labels, predictions, labels=[0, 1])


def prediction_dataframe(
    utt_ids: Iterable[str],
    labels: Sequence[int],
    scores: Sequence[float],
    predictions: Sequence[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utt_id": list(utt_ids),
            "label": list(labels),
            "score_spoof": list(scores),
            "pred_label": list(predictions),
        }
    )


def save_prediction_csv(
    output_path: str | Path,
    utt_ids: Iterable[str],
    labels: Sequence[int],
    scores: Sequence[float],
    predictions: Sequence[int],
) -> pd.DataFrame:
    dataframe = prediction_dataframe(utt_ids=utt_ids, labels=labels, scores=scores, predictions=predictions)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return dataframe
