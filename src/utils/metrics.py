from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve


@dataclass
class EvaluationArtifacts:
    """Container for evaluation outputs."""

    loss: float | None
    accuracy: float
    eer: float
    eer_threshold: float
    confusion: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    utt_ids: list[str]


def _to_numpy_1d(values: Sequence[Any] | np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Convert a sequence-like object into a 1D NumPy array."""

    if isinstance(values, np.ndarray):
        array = values
    elif torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    elif isinstance(values, (pd.Series, pd.Index)):
        array = values.to_numpy()
    else:
        array = np.asarray(list(values))

    array = np.asarray(array)
    if array.ndim == 0:
        array = array.reshape(1)
    else:
        array = np.ravel(array)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}")
    return array


def _prepare_binary_labels(labels: Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    """Validate binary labels and return an int64 array."""

    label_array = _to_numpy_1d(labels, "labels").astype(np.int64, copy=False)
    unique_labels = np.unique(label_array)
    if np.setdiff1d(unique_labels, np.asarray([0, 1], dtype=np.int64)).size:
        raise ValueError(f"Labels must only contain 0/1, got values {unique_labels.tolist()}")
    return label_array


def _prepare_predictions(predictions: Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    """Validate binary predictions and return an int64 array."""

    prediction_array = _to_numpy_1d(predictions, "predictions").astype(np.int64, copy=False)
    unique_predictions = np.unique(prediction_array)
    if np.setdiff1d(unique_predictions, np.asarray([0, 1], dtype=np.int64)).size:
        raise ValueError(f"Predictions must only contain 0/1, got values {unique_predictions.tolist()}")
    return prediction_array


def _prepare_scores(spoof_scores: Sequence[float] | np.ndarray | torch.Tensor) -> np.ndarray:
    """Validate spoof scores and return a finite float64 array."""

    score_array = _to_numpy_1d(spoof_scores, "spoof_scores").astype(np.float64, copy=False)
    invalid_mask = ~np.isfinite(score_array)
    if invalid_mask.any():
        invalid_indices = np.flatnonzero(invalid_mask)[:10].tolist()
        invalid_values = score_array[invalid_mask][:10].tolist()
        raise ValueError(
            "spoof_scores contains NaN or Inf values. "
            f"Invalid indices={invalid_indices}, invalid_values={invalid_values}"
        )
    return score_array


def _ensure_matching_lengths(**arrays: np.ndarray) -> None:
    """Raise a clear error when input arrays have inconsistent lengths."""

    lengths = {name: int(len(array)) for name, array in arrays.items()}
    if len(set(lengths.values())) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Input lengths must match, got {details}")


def logits_to_spoof_scores(logits: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert 2-class logits or probabilities into spoof scores."""

    if isinstance(logits, np.ndarray):
        array = np.asarray(logits, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"Expected logits/probabilities with shape (N, 2), got {array.shape}")
        if np.all((array >= 0.0) & (array <= 1.0)) and np.allclose(array.sum(axis=1), 1.0, atol=1e-4):
            probabilities = array
        else:
            shifted = array - array.max(axis=1, keepdims=True)
            exp_array = np.exp(shifted)
            probabilities = exp_array / np.clip(exp_array.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    else:
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected logits/probabilities with shape (N, 2), got {tuple(logits.shape)}")
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return _prepare_scores(probabilities[:, 1]).astype(np.float32)


def compute_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Compute binary accuracy."""

    label_array = _prepare_binary_labels(labels)
    prediction_array = _prepare_predictions(predictions)
    _ensure_matching_lengths(labels=label_array, predictions=prediction_array)
    return float(accuracy_score(label_array, prediction_array))


def compute_precision_recall_f1(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    """Compute binary precision, recall, and F1."""

    label_array = _prepare_binary_labels(labels)
    prediction_array = _prepare_predictions(predictions)
    _ensure_matching_lengths(labels=label_array, predictions=prediction_array)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_array,
        prediction_array,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_eer(labels: Sequence[int], spoof_scores: Sequence[float]) -> tuple[float, float]:
    """Compute EER and the corresponding decision threshold."""

    label_array = _prepare_binary_labels(labels)
    score_array = _prepare_scores(spoof_scores)
    _ensure_matching_lengths(labels=label_array, spoof_scores=score_array)

    unique_labels = np.unique(label_array)
    if unique_labels.size < 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(label_array, score_array, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    threshold = float(thresholds[index])
    return eer, threshold


def find_best_threshold(
    labels: Sequence[int],
    spoof_scores: Sequence[float],
    criterion: str = "eer",
) -> tuple[float, float]:
    """Find a threshold according to the requested criterion.

    Returns `(best_threshold, criterion_value)`.
    """

    label_array = _prepare_binary_labels(labels)
    score_array = _prepare_scores(spoof_scores)
    _ensure_matching_lengths(labels=label_array, spoof_scores=score_array)

    criterion_name = criterion.lower()
    if criterion_name == "eer":
        eer, threshold = compute_eer(label_array, score_array)
        return threshold, eer

    unique_scores = np.unique(score_array)
    if unique_scores.size == 0:
        raise ValueError("spoof_scores is empty")

    eps = float(np.finfo(np.float32).eps)
    candidate_thresholds = np.concatenate(
        (
            [float(unique_scores.min() - eps)],
            unique_scores.astype(np.float64, copy=False),
            [float(unique_scores.max() + eps)],
        )
    )

    best_threshold = float(candidate_thresholds[0])
    best_value = float("-inf")
    for threshold in candidate_thresholds:
        predictions = (score_array >= threshold).astype(np.int64)
        if criterion_name == "accuracy":
            metric_value = compute_accuracy(label_array, predictions)
        elif criterion_name == "f1":
            metric_value = compute_precision_recall_f1(label_array, predictions)["f1"]
        else:
            raise ValueError("criterion must be one of {'eer', 'accuracy', 'f1'}")

        if metric_value > best_value:
            best_value = float(metric_value)
            best_threshold = float(threshold)

    return best_threshold, best_value


def compute_confusion(labels: Sequence[int], predictions: Sequence[int]) -> np.ndarray:
    """Compute the 2x2 confusion matrix."""

    label_array = _prepare_binary_labels(labels)
    prediction_array = _prepare_predictions(predictions)
    _ensure_matching_lengths(labels=label_array, predictions=prediction_array)
    return confusion_matrix(label_array, prediction_array, labels=[0, 1])


def compute_attack_wise_eer(
    labels: Sequence[int],
    spoof_scores: Sequence[float],
    attack_ids: Sequence[str],
) -> pd.DataFrame:
    """Compute EER for each spoof attack against all bonafide samples."""

    label_array = _prepare_binary_labels(labels)
    score_array = _prepare_scores(spoof_scores)
    attack_array = _to_numpy_1d(attack_ids, "attack_ids").astype(str)
    attack_array = np.where(np.char.strip(attack_array) == "", "unknown", attack_array)
    _ensure_matching_lengths(labels=label_array, spoof_scores=score_array, attack_ids=attack_array)

    bonafide_mask = label_array == 0
    spoof_attack_ids = sorted({attack_id for attack_id, label in zip(attack_array.tolist(), label_array.tolist()) if label == 1})

    rows: list[dict[str, Any]] = []
    for attack_id in spoof_attack_ids:
        attack_mask = attack_array == attack_id
        subset_mask = bonafide_mask | attack_mask
        subset_labels = label_array[subset_mask]
        subset_scores = score_array[subset_mask]

        num_samples = int(subset_mask.sum())
        num_bonafide = int((subset_labels == 0).sum())
        num_spoof = int((subset_labels == 1).sum())
        valid = num_bonafide > 0 and num_spoof > 0
        eer = float("nan")
        eer_threshold = float("nan")
        if valid:
            eer, eer_threshold = compute_eer(subset_labels, subset_scores)

        rows.append(
            {
                "attack_id": str(attack_id),
                "num_samples": num_samples,
                "num_bonafide": num_bonafide,
                "num_spoof": num_spoof,
                "eer": float(eer),
                "eer_threshold": float(eer_threshold),
                "valid": bool(valid),
            }
        )

    columns = ["attack_id", "num_samples", "num_bonafide", "num_spoof", "eer", "eer_threshold", "valid"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def summarize_errors(
    labels: Sequence[int],
    predictions: Sequence[int],
    attack_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize classification errors for reporting."""

    label_array = _prepare_binary_labels(labels)
    prediction_array = _prepare_predictions(predictions)
    _ensure_matching_lengths(labels=label_array, predictions=prediction_array)

    error_mask = label_array != prediction_array
    bonafide_mask = label_array == 0
    spoof_mask = label_array == 1
    false_positive_mask = bonafide_mask & (prediction_array == 1)
    false_negative_mask = spoof_mask & (prediction_array == 0)

    summary: dict[str, Any] = {
        "num_samples": int(label_array.size),
        "num_errors": int(error_mask.sum()),
        "error_rate": float(error_mask.mean()) if label_array.size else 0.0,
        "num_bonafide": int(bonafide_mask.sum()),
        "num_spoof": int(spoof_mask.sum()),
        "false_positive_bonafide_as_spoof": {
            "count": int(false_positive_mask.sum()),
            "rate": float(false_positive_mask.sum() / bonafide_mask.sum()) if bonafide_mask.any() else 0.0,
        },
        "false_negative_spoof_as_bonafide": {
            "count": int(false_negative_mask.sum()),
            "rate": float(false_negative_mask.sum() / spoof_mask.sum()) if spoof_mask.any() else 0.0,
        },
    }

    if attack_ids is not None:
        attack_array = _to_numpy_1d(attack_ids, "attack_ids").astype(str)
        attack_array = np.where(np.char.strip(attack_array) == "", "unknown", attack_array)
        _ensure_matching_lengths(labels=label_array, predictions=prediction_array, attack_ids=attack_array)

        dataframe = pd.DataFrame(
            {
                "attack_id": attack_array,
                "label": label_array,
                "prediction": prediction_array,
            }
        )
        dataframe["is_error"] = dataframe["label"] != dataframe["prediction"]
        dataframe["false_positive"] = (dataframe["label"] == 0) & (dataframe["prediction"] == 1)
        dataframe["false_negative"] = (dataframe["label"] == 1) & (dataframe["prediction"] == 0)

        by_attack = (
            dataframe.groupby("attack_id", dropna=False)
            .agg(
                num_samples=("label", "size"),
                num_errors=("is_error", "sum"),
                num_false_positive=("false_positive", "sum"),
                num_false_negative=("false_negative", "sum"),
            )
            .reset_index()
        )
        by_attack["error_rate"] = by_attack["num_errors"] / by_attack["num_samples"].clip(lower=1)
        by_attack = by_attack.sort_values(
            by=["num_errors", "error_rate", "attack_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        hardest_attacks = by_attack[by_attack["attack_id"].astype(str).str.lower() != "bonafide"].head(10)
        summary["by_attack"] = safe_json_value(by_attack.to_dict(orient="records"))
        summary["hardest_attacks"] = safe_json_value(hardest_attacks.to_dict(orient="records"))

    return safe_json_value(summary)


def prediction_dataframe(
    utt_ids: Iterable[str],
    labels: Sequence[int],
    scores: Sequence[float],
    predictions: Sequence[int],
    extra_columns: Mapping[str, Sequence[Any]] | None = None,
) -> pd.DataFrame:
    """Build a prediction dataframe and keep the legacy column layout."""

    label_array = _prepare_binary_labels(labels)
    score_array = _prepare_scores(scores)
    prediction_array = _prepare_predictions(predictions)
    utt_id_list = [str(utt_id) for utt_id in utt_ids]
    _ensure_matching_lengths(
        utt_ids=np.asarray(utt_id_list, dtype=object),
        labels=label_array,
        scores=score_array,
        predictions=prediction_array,
    )

    dataframe = pd.DataFrame(
        {
            "utt_id": utt_id_list,
            "label": label_array.tolist(),
            "score_spoof": score_array.tolist(),
            "pred_label": prediction_array.tolist(),
        }
    )
    if extra_columns:
        for column_name, values in extra_columns.items():
            column_values = _to_numpy_1d(values, column_name)
            if len(column_values) != len(dataframe):
                raise ValueError(
                    f"Extra column '{column_name}' must have {len(dataframe)} values, got {len(column_values)}"
                )
            dataframe[str(column_name)] = column_values.tolist()
    return dataframe


def save_prediction_csv(
    output_path: str | Path,
    utt_ids: Iterable[str],
    labels: Sequence[int],
    scores: Sequence[float],
    predictions: Sequence[int],
    extra_columns: Mapping[str, Sequence[Any]] | None = None,
) -> pd.DataFrame:
    """Save prediction CSV with optional extra columns."""

    dataframe = prediction_dataframe(
        utt_ids=utt_ids,
        labels=labels,
        scores=scores,
        predictions=predictions,
        extra_columns=extra_columns,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return dataframe


def safe_json_value(obj: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-safe Python objects."""

    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return [safe_json_value(item) for item in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [safe_json_value(item) for item in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [safe_json_value(record) for record in obj.to_dict(orient="records")]
    if torch.is_tensor(obj):
        return safe_json_value(obj.detach().cpu().numpy())
    if isinstance(obj, Mapping):
        return {str(key): safe_json_value(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_json_value(item) for item in obj]
    return obj


def safe_numeric_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible JSON helper for dictionaries."""

    return safe_json_value(dict(payload))
