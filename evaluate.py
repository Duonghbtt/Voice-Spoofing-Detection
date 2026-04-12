from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.utils.dataset import PROTOCOL_FILES, resolve_data_root
from src.utils.metrics import (
    EvaluationArtifacts,
    compute_accuracy,
    compute_attack_wise_eer,
    compute_confusion,
    compute_eer,
    compute_precision_recall_f1,
    find_best_threshold,
    logits_to_spoof_scores,
    save_prediction_csv,
    safe_json_value,
    summarize_errors,
)
from src.utils.visualize import (
    save_attack_wise_eer_figure,
    save_confusion_matrix_figure,
    save_roc_curve_figure,
    save_score_histogram,
)
from train import PROFILE_PRESETS, create_spoof_dataloader, resolve_device, upsert_rows


LOGGER = logging.getLogger("evaluate")
ATTACK_PATTERN = re.compile(r"^A\d{2}$", re.IGNORECASE)
UTT_ID_PATTERN = re.compile(r"^LA_[EDT]_\d+$", re.IGNORECASE)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _resolve_autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    return torch.cuda.amp.autocast()


def _unwrap_model_output(output: Any) -> torch.Tensor:
    """Extract the main tensor output from a model return value."""

    if torch.is_tensor(output):
        return output

    if isinstance(output, Mapping):
        for key in ("logits", "output", "outputs", "scores", "score", "probabilities", "probs"):
            if key in output:
                try:
                    return _unwrap_model_output(output[key])
                except (TypeError, ValueError):
                    continue

        tensor_values = [value for value in output.values() if torch.is_tensor(value)]
        if len(tensor_values) == 1:
            return tensor_values[0]
        raise ValueError(
            "Could not infer logits tensor from model output dict. "
            f"Available keys: {list(output.keys())}"
        )

    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item) or isinstance(item, (dict, list, tuple)):
                try:
                    return _unwrap_model_output(item)
                except (TypeError, ValueError):
                    continue
        raise ValueError("Could not infer logits tensor from model output sequence")

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _extract_scores_and_predictions(output: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert model output into spoof scores and default predictions."""

    tensor = _unwrap_model_output(output).detach()
    if tensor.ndim == 0:
        tensor = tensor.view(1)

    if tensor.ndim == 2 and tensor.shape[1] == 2:
        scores = logits_to_spoof_scores(tensor)
        predictions = tensor.argmax(dim=1).detach().cpu().numpy().astype(np.int64, copy=False)
        return scores, predictions

    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]

    if tensor.ndim != 1:
        raise ValueError(
            "Expected model output with shape (N, 2), (N, 1), or (N,), "
            f"got shape {tuple(tensor.shape)}"
        )

    score_tensor = tensor.detach().cpu().float()
    if score_tensor.numel() == 0:
        raise ValueError("Model output is empty")

    if torch.all((score_tensor >= 0.0) & (score_tensor <= 1.0)):
        probabilities = score_tensor
    else:
        probabilities = torch.sigmoid(score_tensor)

    scores = probabilities.numpy().astype(np.float32, copy=False)
    predictions = (scores >= 0.5).astype(np.int64)
    return scores, predictions


def _normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("state_dict", "model_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, Mapping):
            return _normalize_state_dict_keys(state_dict)

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return _normalize_state_dict_keys(checkpoint)

    raise ValueError(
        "Unsupported checkpoint format. Expected keys like 'state_dict' or 'model_state_dict', "
        "or a raw state_dict mapping."
    )


def _infer_model_name_from_state_dict(state_dict: Mapping[str, Any]) -> str | None:
    keys = list(state_dict.keys())
    if any(key.startswith("backbone.") for key in keys):
        return "resnet"
    if any(".filter.weight" in key for key in keys):
        return "lcnn"
    if any(".block.0.weight" in key for key in keys):
        return "cnn"
    return None


def _checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    profile = str(checkpoint.get("profile", "custom"))
    preset = PROFILE_PRESETS.get(profile, {})

    model_name = checkpoint.get("model_name")
    if not isinstance(model_name, str):
        model_name = checkpoint.get("model") if isinstance(checkpoint.get("model"), str) else None
    if not model_name:
        model_name = _infer_model_name_from_state_dict(state_dict)
    if not model_name:
        raise ValueError(
            "Could not infer model_name from checkpoint metadata or state_dict. "
            "Please re-save the checkpoint with metadata."
        )

    feature_name = checkpoint.get("feature_name")
    if not isinstance(feature_name, str):
        feature_name = checkpoint.get("src/feature") if isinstance(checkpoint.get("src/feature"), str) else None
    if not feature_name:
        feature_name = preset.get("feature")
    if not feature_name:
        raise ValueError(
            "Could not infer feature_name from checkpoint metadata. "
            "Expected a checkpoint saved by the current training pipeline."
        )

    architecture_signature = checkpoint.get("architecture_signature", {})
    architecture_raw = architecture_signature.get("raw", {}) if isinstance(architecture_signature, Mapping) else {}

    in_channels = int(architecture_raw.get("in_channels", 1))
    num_classes = int(
        architecture_raw.get(
            "num_classes",
            len(checkpoint.get("class_mapping", {})) or 2,
        )
    )

    return {
        "profile": profile,
        "model_name": model_name,
        "feature_name": feature_name,
        "target_frames": int(checkpoint.get("target_frames", 128)),
        "normalize": bool(checkpoint.get("normalize", preset.get("normalize", False))),
        "batch_size": int(checkpoint.get("batch_size", 64 if model_name == "cnn" else 32)),
        "amp": bool(checkpoint.get("amp", preset.get("amp", False))),
        "in_channels": in_channels,
        "num_classes": max(1, num_classes),
    }


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Load a model and checkpoint metadata with flexible checkpoint parsing."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} must be a mapping, got {type(checkpoint)!r}"
        )

    state_dict = _extract_state_dict(checkpoint)
    metadata = _checkpoint_metadata(checkpoint, state_dict)
    model = build_model(
        metadata["model_name"],
        in_channels=metadata["in_channels"],
        num_classes=metadata["num_classes"],
    )

    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    LOGGER.info(
        "Loaded checkpoint: %s | profile=%s | model=%s | feature=%s",
        checkpoint_path.resolve(),
        metadata["profile"],
        metadata["model_name"],
        metadata["feature_name"],
    )
    if missing_keys:
        LOGGER.warning("Missing keys while loading checkpoint: %s", missing_keys[:20])
    if unexpected_keys:
        LOGGER.warning("Unexpected keys while loading checkpoint: %s", unexpected_keys[:20])

    model = model.to(device)
    return model, dict(checkpoint), metadata


def run_evaluation_loop(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = False,
    threshold: float | None = None,
    description: str = "Evaluating",
) -> EvaluationArtifacts:
    """Run evaluation with tqdm and flexible model-output handling."""

    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")

    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_scores: list[float] = []
    all_utt_ids: list[str] = []

    progress = tqdm(dataloader, desc=description, leave=False)
    with torch.no_grad():
        for inputs, labels, utt_ids in progress:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            with _resolve_autocast(device=device, enabled=use_amp):
                outputs = model(inputs)

            batch_scores, batch_predictions = _extract_scores_and_predictions(outputs)
            if threshold is not None:
                batch_predictions = (batch_scores >= threshold).astype(np.int64)

            label_array = labels.detach().cpu().numpy().astype(np.int64, copy=False)
            all_labels.extend(label_array.tolist())
            all_predictions.extend(batch_predictions.tolist())
            all_scores.extend(batch_scores.tolist())
            all_utt_ids.extend(str(utt_id) for utt_id in utt_ids)

    labels_array = np.asarray(all_labels, dtype=np.int64)
    prediction_array = np.asarray(all_predictions, dtype=np.int64)
    score_array = np.asarray(all_scores, dtype=np.float32)
    accuracy = compute_accuracy(labels_array, prediction_array)
    eer, eer_threshold = compute_eer(labels_array, score_array)
    confusion = compute_confusion(labels_array, prediction_array)

    return EvaluationArtifacts(
        loss=None,
        accuracy=accuracy,
        eer=eer,
        eer_threshold=eer_threshold,
        confusion=confusion,
        labels=labels_array,
        predictions=prediction_array,
        scores=score_array,
        utt_ids=all_utt_ids,
    )


def _protocol_search_roots(data_root: str | Path, protocol_root: str | Path | None) -> list[Path]:
    if protocol_root is not None:
        return [Path(protocol_root)]

    resolved_data_root = resolve_data_root(data_root)
    return [
        resolved_data_root / "raw" / "LA",
        resolved_data_root / "raw" / "ASVspoof2019" / "LA",
        resolved_data_root / "raw" / "LA_2019" / "LA",
        resolved_data_root / "raw",
        resolved_data_root,
    ]


def _zip_candidates(data_root: str | Path, protocol_root: str | Path | None, protocol_file: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if protocol_file is not None and Path(protocol_file).suffix.lower() == ".zip":
        candidates.append(Path(protocol_file))

    if protocol_root is not None:
        protocol_root_path = Path(protocol_root)
        if protocol_root_path.is_file() and protocol_root_path.suffix.lower() == ".zip":
            candidates.append(protocol_root_path)
        else:
            candidates.extend(
                [
                    protocol_root_path / "LA.zip",
                    protocol_root_path / "ASVspoof2019.zip",
                ]
            )

    resolved_data_root = resolve_data_root(data_root)
    candidates.extend(
        [
            resolved_data_root / "raw" / "LA.zip",
            resolved_data_root / "raw" / "ASVspoof2019" / "LA.zip",
            resolved_data_root / "LA.zip",
        ]
    )
    return candidates


def _resolve_protocol_file(
    *,
    split: str,
    data_root: str | Path,
    protocol_root: str | Path | None = None,
    protocol_file: str | Path | None = None,
) -> Path | None:
    if split not in PROTOCOL_FILES:
        raise ValueError(f"Unsupported ASVspoof2019 split: {split}")

    if protocol_file is not None:
        explicit_path = Path(protocol_file)
        if explicit_path.exists() and explicit_path.is_file() and explicit_path.suffix.lower() != ".zip":
            return explicit_path
        if explicit_path.exists() and explicit_path.suffix.lower() == ".zip":
            return None
        raise FileNotFoundError(f"Could not locate protocol file: {explicit_path}")

    protocol_name = PROTOCOL_FILES[split]
    for root in _protocol_search_roots(data_root=data_root, protocol_root=protocol_root):
        candidates = [
            root / protocol_name,
            root / "ASVspoof2019_LA_cm_protocols" / protocol_name,
            root / "LA" / "ASVspoof2019_LA_cm_protocols" / protocol_name,
            root / "ASVspoof2019" / "LA" / "ASVspoof2019_LA_cm_protocols" / protocol_name,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        if root.exists():
            recursive_matches = list(root.rglob(protocol_name))
            if recursive_matches:
                return recursive_matches[0]

    return None


def _read_protocol_lines(
    *,
    split: str,
    data_root: str | Path,
    protocol_root: str | Path | None = None,
    protocol_file: str | Path | None = None,
) -> tuple[list[str], str]:
    protocol_name = PROTOCOL_FILES[split]
    resolved_file = _resolve_protocol_file(
        split=split,
        data_root=data_root,
        protocol_root=protocol_root,
        protocol_file=protocol_file,
    )
    if resolved_file is not None:
        with resolved_file.open("r", encoding="utf-8") as handle:
            return handle.readlines(), str(resolved_file)

    for zip_path in _zip_candidates(data_root=data_root, protocol_root=protocol_root, protocol_file=protocol_file):
        if not zip_path.exists() or zip_path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(zip_path, "r") as archive:
            matched_name = next((name for name in archive.namelist() if name.endswith(protocol_name)), None)
            if matched_name is None:
                continue
            with archive.open(matched_name, "r") as handle:
                lines = [line.decode("utf-8") for line in handle]
            return lines, f"{zip_path}!{matched_name}"

    raise FileNotFoundError(
        f"Could not locate ASVspoof2019 protocol '{protocol_name}'. "
        f"protocol_root={protocol_root!r}, protocol_file={protocol_file!r}, data_root={data_root!r}"
    )


def _extract_utt_id(parts: list[str]) -> str | None:
    for token in parts:
        if UTT_ID_PATTERN.match(token):
            return token
    if len(parts) >= 2:
        # ASSUMPTION: official ASVspoof2019 protocol stores utt_id in the 2nd column.
        return parts[1]
    return None


def _extract_label(parts: list[str]) -> str | None:
    for token in reversed(parts):
        normalized = token.strip().lower()
        if normalized in {"bonafide", "spoof"}:
            return normalized
    return None


def _extract_attack_id(parts: list[str]) -> str | None:
    for token in parts:
        normalized = token.strip().upper()
        if ATTACK_PATTERN.match(normalized):
            return normalized
    return None


def parse_asvspoof2019_attack_map(
    *,
    split: str,
    data_root: str | Path,
    protocol_root: str | Path | None = None,
    protocol_file: str | Path | None = None,
) -> tuple[dict[str, str], str]:
    """Parse ASVspoof2019 protocol and map utt_id -> attack_id."""

    lines, source = _read_protocol_lines(
        split=split,
        data_root=data_root,
        protocol_root=protocol_root,
        protocol_file=protocol_file,
    )

    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        utt_id = _extract_utt_id(parts)
        label = _extract_label(parts)
        if utt_id is None or label is None:
            LOGGER.debug("Skipping malformed protocol line %s: %s", line_number, stripped)
            continue

        if label == "bonafide":
            mapping[utt_id] = "bonafide"
            continue

        attack_id = _extract_attack_id(parts) or "unknown"
        mapping[utt_id] = attack_id

    if not mapping:
        raise ValueError(f"Parsed protocol file but found no utterance mappings in {source}")
    return mapping, source


def _write_json(output_path: str | Path, payload: Any) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_json_value(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)


def _load_existing_2019_result(
    results_path: Path,
    profile: str,
    model: str,
    feature: str,
    checkpoint_path: str,
) -> dict[str, Any] | None:
    if not results_path.exists():
        return None

    dataframe = pd.read_csv(results_path)
    if dataframe.empty:
        return None

    mask = (
        (dataframe["profile"] == profile)
        & (dataframe["model"] == model)
        & (dataframe["feature"] == feature)
    )
    if "checkpoint" in dataframe.columns:
        mask &= dataframe["checkpoint"] == checkpoint_path
    matched = dataframe.loc[mask]
    if matched.empty:
        return None
    return matched.iloc[-1].to_dict()


def _write_eer_comparison(output_root: Path) -> None:
    results_2019_path = output_root / "results" / "results_2019.csv"
    results_2021_path = output_root / "results" / "results_2021.csv"
    if not results_2019_path.exists() or not results_2021_path.exists():
        return

    results_2019 = pd.read_csv(results_2019_path)
    results_2021 = pd.read_csv(results_2021_path)
    if results_2019.empty or results_2021.empty:
        return

    merged = results_2021.merge(
        results_2019[["profile", "model", "feature", "eer"]].rename(columns={"eer": "eer_2019"}),
        on=["profile", "model", "feature"],
        how="left",
        suffixes=("", "_dup"),
    )
    if "eer_2019_dup" in merged.columns:
        merged = merged.drop(columns=["eer_2019_dup"])

    comparison = merged[["profile", "model", "feature", "eer_2019", "eer_2021", "generalization_gap"]]
    comparison = comparison.sort_values(["profile", "model", "feature"]).reset_index(drop=True)
    comparison.to_csv(output_root / "results" / "eer_comparison.csv", index=False)


def _build_metrics_payload(
    *,
    dataset_name: str,
    checkpoint_path: str,
    metadata: Mapping[str, Any],
    artifacts: EvaluationArtifacts,
    decision_threshold: float,
) -> dict[str, Any]:
    prf = compute_precision_recall_f1(artifacts.labels, artifacts.predictions)
    eer_threshold, eer_value = find_best_threshold(artifacts.labels, artifacts.scores, criterion="eer")
    return {
        "dataset": dataset_name,
        "profile": str(metadata["profile"]),
        "model": str(metadata["model_name"]),
        "feature": str(metadata["feature_name"]),
        "checkpoint": checkpoint_path,
        "num_samples": int(len(artifacts.labels)),
        "num_bonafide": int((artifacts.labels == 0).sum()),
        "num_spoof": int((artifacts.labels == 1).sum()),
        "accuracy": float(artifacts.accuracy),
        "precision": float(prf["precision"]),
        "recall": float(prf["recall"]),
        "f1": float(prf["f1"]),
        "eer": float(eer_value),
        "eer_threshold": float(eer_threshold),
        "decision_threshold": float(decision_threshold),
        "confusion_matrix": safe_json_value(artifacts.confusion),
    }


def _save_standard_outputs(
    *,
    output_root: Path,
    profile: str,
    dataset_suffix: str,
    metrics_payload: Mapping[str, Any],
    artifacts: EvaluationArtifacts,
    save_metrics_json: bool,
    save_confusion_npy: bool,
    save_figures: bool,
) -> None:
    results_dir = output_root / "results"
    figures_dir = output_root / "figures"

    if save_metrics_json:
        _write_json(results_dir / f"metrics_{profile}_{dataset_suffix}.json", metrics_payload)

    if save_confusion_npy:
        np.save(results_dir / f"confusion_matrix_{profile}_{dataset_suffix}.npy", artifacts.confusion)

    if save_figures:
        save_confusion_matrix_figure(
            matrix=artifacts.confusion,
            output_path=figures_dir / f"{profile}_confusion_matrix_{dataset_suffix}.png",
            title=f"{profile.title()} ASVspoof {dataset_suffix} Confusion Matrix",
        )
        save_roc_curve_figure(
            labels=artifacts.labels,
            scores=artifacts.scores,
            output_path=figures_dir / f"{profile}_roc_{dataset_suffix}.png",
            title=f"{profile.title()} ROC ({dataset_suffix})",
        )
        save_score_histogram(
            labels=artifacts.labels,
            scores=artifacts.scores,
            output_path=figures_dir / f"{profile}_score_hist_{dataset_suffix}.png",
            title=f"{profile.title()} Score Distribution ({dataset_suffix})",
        )


def _attack_ids_for_predictions(utt_ids: list[str], attack_map: Mapping[str, str] | None) -> list[str] | None:
    if attack_map is None:
        return None
    return [str(attack_map.get(utt_id, "unknown")) for utt_id in utt_ids]


def _run_2019_evaluation(
    *,
    model: nn.Module,
    metadata: Mapping[str, Any],
    checkpoint_path: str,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, Any], EvaluationArtifacts]:
    eval_loader = create_spoof_dataloader(
        feature_name=metadata["feature_name"],
        split="eval",
        target_frames=metadata["target_frames"],
        batch_size=args.batch_size or metadata["batch_size"],
        data_root=args.data_root,
        protocol_root=args.protocol_root,
        normalize=metadata["normalize"],
        specaugment=False,
        training=False,
        num_workers=args.num_workers,
        device=device,
    )

    decision_threshold = float(args.threshold) if args.threshold is not None else 0.5
    artifacts = run_evaluation_loop(
        model=model,
        dataloader=eval_loader,
        device=device,
        use_amp=use_amp,
        threshold=args.threshold,
        description="Evaluating ASVspoof 2019",
    )
    metrics_payload = _build_metrics_payload(
        dataset_name="2019_eval",
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        artifacts=artifacts,
        decision_threshold=decision_threshold,
    )

    attack_map: dict[str, str] | None = None
    attack_source: str | None = None
    try:
        attack_map, attack_source = parse_asvspoof2019_attack_map(
            split="eval",
            data_root=args.data_root,
            protocol_root=args.protocol_root,
            protocol_file=args.protocol_file,
        )
        LOGGER.info("Loaded attack map for ASVspoof2019 eval from %s", attack_source)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Could not parse ASVspoof2019 attack IDs: %s", exc)

    attack_ids = _attack_ids_for_predictions(artifacts.utt_ids, attack_map)
    predictions_path = Path(args.output_root) / "results" / f"predictions_{metadata['profile']}_2019.csv"
    save_prediction_csv(
        output_path=predictions_path,
        utt_ids=artifacts.utt_ids,
        labels=artifacts.labels,
        scores=artifacts.scores,
        predictions=artifacts.predictions,
        extra_columns={"attack_id": attack_ids} if attack_ids is not None else None,
    )

    _save_standard_outputs(
        output_root=Path(args.output_root),
        profile=metadata["profile"],
        dataset_suffix="2019",
        metrics_payload=metrics_payload,
        artifacts=artifacts,
        save_metrics_json=args.save_metrics_json,
        save_confusion_npy=args.save_confusion_npy,
        save_figures=args.save_figures,
    )

    if args.per_attack:
        results_dir = Path(args.output_root) / "results"
        figures_dir = Path(args.output_root) / "figures"
        if attack_ids is not None:
            attack_wise_df = compute_attack_wise_eer(
                labels=artifacts.labels,
                spoof_scores=artifacts.scores,
                attack_ids=attack_ids,
            )
        else:
            attack_wise_df = pd.DataFrame(
                columns=["attack_id", "num_samples", "num_bonafide", "num_spoof", "eer", "eer_threshold", "valid"]
            )

        attack_wise_path = results_dir / f"attack_wise_eer_{metadata['profile']}_2019.csv"
        attack_wise_df.to_csv(attack_wise_path, index=False)

        error_summary = {
            "dataset": "2019_eval",
            "profile": metadata["profile"],
            "model": metadata["model_name"],
            "feature": metadata["feature_name"],
            "checkpoint": checkpoint_path,
            "attack_map_source": attack_source,
            "summary": summarize_errors(
                labels=artifacts.labels,
                predictions=artifacts.predictions,
                attack_ids=attack_ids,
            ),
        }
        _write_json(results_dir / f"error_summary_{metadata['profile']}_2019.json", error_summary)

        if args.save_figures and not attack_wise_df.empty and attack_wise_df["valid"].astype(bool).any():
            save_attack_wise_eer_figure(
                df=attack_wise_df,
                output_path=figures_dir / f"{metadata['profile']}_attack_wise_eer_2019.png",
                title=f"{metadata['profile'].title()} Attack-wise EER (2019)",
            )

    result_row = {
        "profile": metadata["profile"],
        "model": metadata["model_name"],
        "feature": metadata["feature_name"],
        "accuracy": metrics_payload["accuracy"],
        "precision": metrics_payload["precision"],
        "recall": metrics_payload["recall"],
        "f1": metrics_payload["f1"],
        "eer": metrics_payload["eer"],
        "eer_threshold": metrics_payload["eer_threshold"],
        "decision_threshold": metrics_payload["decision_threshold"],
        "num_samples": metrics_payload["num_samples"],
        "checkpoint": checkpoint_path,
        "predictions_path": str(predictions_path.resolve()),
    }
    upsert_rows(
        csv_path=Path(args.output_root) / "results" / "results_2019.csv",
        rows=[result_row],
        key_columns=["profile", "model", "feature"],
        sort_columns=["profile", "model", "feature"],
    )
    return result_row, artifacts


def _run_2021_evaluation(
    *,
    model: nn.Module,
    metadata: Mapping[str, Any],
    checkpoint_path: str,
    result_2019: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, Any], EvaluationArtifacts]:
    eval_loader = create_spoof_dataloader(
        feature_name=metadata["feature_name"],
        split="eval_2021",
        target_frames=metadata["target_frames"],
        batch_size=args.batch_size or metadata["batch_size"],
        data_root=args.data_root,
        feature_root=args.eval_2021_features,
        normalize=metadata["normalize"],
        specaugment=False,
        training=False,
        num_workers=args.num_workers,
        device=device,
        labels_path=args.eval_2021_labels,
    )

    decision_threshold = float(args.threshold) if args.threshold is not None else 0.5
    artifacts = run_evaluation_loop(
        model=model,
        dataloader=eval_loader,
        device=device,
        use_amp=use_amp,
        threshold=args.threshold,
        description="Evaluating ASVspoof 2021",
    )
    metrics_payload = _build_metrics_payload(
        dataset_name="2021_eval",
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        artifacts=artifacts,
        decision_threshold=decision_threshold,
    )

    predictions_path = Path(args.output_root) / "results" / f"predictions_{metadata['profile']}_2021.csv"
    save_prediction_csv(
        output_path=predictions_path,
        utt_ids=artifacts.utt_ids,
        labels=artifacts.labels,
        scores=artifacts.scores,
        predictions=artifacts.predictions,
    )
    _save_standard_outputs(
        output_root=Path(args.output_root),
        profile=metadata["profile"],
        dataset_suffix="2021",
        metrics_payload=metrics_payload,
        artifacts=artifacts,
        save_metrics_json=args.save_metrics_json,
        save_confusion_npy=args.save_confusion_npy,
        save_figures=args.save_figures,
    )

    eer_2019 = float(result_2019["eer"])
    result_row = {
        "profile": metadata["profile"],
        "model": metadata["model_name"],
        "feature": metadata["feature_name"],
        "accuracy": metrics_payload["accuracy"],
        "precision": metrics_payload["precision"],
        "recall": metrics_payload["recall"],
        "f1": metrics_payload["f1"],
        "eer_2021": metrics_payload["eer"],
        "eer_threshold": metrics_payload["eer_threshold"],
        "decision_threshold": metrics_payload["decision_threshold"],
        "num_samples": metrics_payload["num_samples"],
        "eer_2019": eer_2019,
        "generalization_gap": float(metrics_payload["eer"]) - eer_2019,
        "checkpoint": checkpoint_path,
        "predictions_path": str(predictions_path.resolve()),
    }
    upsert_rows(
        csv_path=Path(args.output_root) / "results" / "results_2021.csv",
        rows=[result_row],
        key_columns=["profile", "model", "feature"],
        sort_columns=["profile", "model", "feature"],
    )
    _write_eer_comparison(Path(args.output_root))
    return result_row, artifacts


def _debug_attack_map(args: argparse.Namespace) -> None:
    attack_map, source = parse_asvspoof2019_attack_map(
        split="eval",
        data_root=args.data_root,
        protocol_root=args.protocol_root,
        protocol_file=args.protocol_file,
    )
    counts = pd.Series(list(attack_map.values()), dtype="string").value_counts().sort_index()
    LOGGER.info("Parsed %s utterance attack mappings from %s", len(attack_map), source)
    for attack_id, count in counts.items():
        LOGGER.info("  %s: %s", attack_id, int(count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained voice spoofing detection model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_2019", action="store_true")
    parser.add_argument("--eval_2021", action="store_true")
    parser.add_argument("--eval_2021_features", default=None)
    parser.add_argument("--eval_2021_labels", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--protocol_root", default=None)
    parser.add_argument("--protocol_file", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--per_attack", action="store_true")
    parser.add_argument("--attack_map_only", action="store_true")

    parser.add_argument("--save_metrics_json", dest="save_metrics_json", action="store_true")
    parser.add_argument("--no_save_metrics_json", dest="save_metrics_json", action="store_false")
    parser.set_defaults(save_metrics_json=True)

    parser.add_argument("--save_confusion_npy", dest="save_confusion_npy", action="store_true")
    parser.add_argument("--no_save_confusion_npy", dest="save_confusion_npy", action="store_false")
    parser.set_defaults(save_confusion_npy=True)

    parser.add_argument("--save_figures", dest="save_figures", action="store_true")
    parser.add_argument("--no_save_figures", dest="save_figures", action="store_false")
    parser.set_defaults(save_figures=True)
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    output_root = Path(args.output_root)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)

    if args.attack_map_only:
        _debug_attack_map(args)
        if not args.eval_2019 and not args.eval_2021:
            return

    device = resolve_device(args.device)
    checkpoint_path = str(Path(args.checkpoint).resolve())
    model, checkpoint, metadata = load_model_from_checkpoint(args.checkpoint, device=device)

    use_amp = bool(metadata["amp"]) and device.type == "cuda"
    run_2019 = args.eval_2019 or not args.eval_2021
    result_2019: dict[str, Any] | None = None

    if run_2019:
        result_2019, _ = _run_2019_evaluation(
            model=model,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
            args=args,
            device=device,
            use_amp=use_amp,
        )

    if args.eval_2021:
        if result_2019 is None:
            result_2019 = _load_existing_2019_result(
                results_path=output_root / "results" / "results_2019.csv",
                profile=metadata["profile"],
                model=metadata["model_name"],
                feature=metadata["feature_name"],
                checkpoint_path=checkpoint_path,
            )
            if result_2019 is None:
                LOGGER.info("No cached ASVspoof2019 result found; evaluating 2019 first for generalization gap.")
                result_2019, _ = _run_2019_evaluation(
                    model=model,
                    metadata=metadata,
                    checkpoint_path=checkpoint_path,
                    args=args,
                    device=device,
                    use_amp=use_amp,
                )

        _run_2021_evaluation(
            model=model,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
            result_2019=result_2019,
            args=args,
            device=device,
            use_amp=use_amp,
        )

    LOGGER.info(
        "Evaluation complete for %s (%s + %s).",
        metadata["profile"],
        metadata["model_name"],
        metadata["feature_name"],
    )


if __name__ == "__main__":
    main()
