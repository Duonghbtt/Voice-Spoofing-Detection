from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import build_model
from src.utils.dataset import SpoofDataset, build_2019_samples, build_2021_samples
from src.utils.metrics import EvaluationArtifacts, compute_accuracy, compute_confusion, compute_eer, logits_to_spoof_scores
from src.utils.visualize import save_learning_curve


CLASS_MAPPING = {"bonafide": 0, "spoof": 1}
SIGNATURE_VERSION = "v1"
DEFAULT_MODEL_KWARGS = {
    "cnn": {"dropout": 0.3},
    "lcnn": {"dropout": 0.3},
    "resnet": {},
}
_MISSING = object()

PROFILE_PRESETS = {
    "baseline": {
        "model": "cnn",
        "feature": "mfcc",
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-3,
        "scheduler": "step",
        "step_size": 15,
        "gamma": 0.5,
        "normalize": False,
        "specaugment": False,
        "amp": False,
        "early_stopping_patience": None,
    },
    "optimized": {
        "model": "lcnn",
        "feature": "lfcc",
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-3,
        "scheduler": "cosine",
        "step_size": 15,
        "gamma": 0.5,
        "normalize": True,
        "specaugment": True,
        "amp": True,
        "early_stopping_patience": 8,
    },
}


@dataclass
class ExperimentConfig:
    profile: str
    model: str
    feature: str
    epochs: int
    batch_size: int
    lr: float
    scheduler: str
    step_size: int
    gamma: float
    target_frames: int
    normalize: bool
    specaugment: bool
    amp: bool
    early_stopping_patience: int | None
    data_root: str
    protocol_root: str | None
    output_root: str
    seed: int
    num_workers: int
    device: str | None
    resume: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_batch_size(model_name: str) -> int:
    return 64 if model_name == "cnn" else 32


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    profile = args.profile
    if profile is None and args.model is None and args.feature is None:
        profile = "baseline"

    preset = PROFILE_PRESETS.get(profile or "", {})
    model = args.model or preset.get("model") or "cnn"
    feature = args.feature or preset.get("feature") or "mfcc"

    return ExperimentConfig(
        profile=profile or "custom",
        model=model,
        feature=feature,
        epochs=args.epochs if args.epochs is not None else preset.get("epochs", 50),
        batch_size=args.batch_size if args.batch_size is not None else preset.get("batch_size", _default_batch_size(model)),
        lr=args.lr if args.lr is not None else preset.get("lr", 1e-3),
        scheduler=args.scheduler or preset.get("scheduler", "step"),
        step_size=args.step_size if args.step_size is not None else preset.get("step_size", 15),
        gamma=args.gamma if args.gamma is not None else preset.get("gamma", 0.5),
        target_frames=args.target_frames,
        normalize=args.normalize if args.normalize is not None else preset.get("normalize", False),
        specaugment=args.specaugment if args.specaugment is not None else preset.get("specaugment", False),
        amp=args.amp if args.amp is not None else preset.get("amp", False),
        early_stopping_patience=(
            args.early_stopping_patience
            if args.early_stopping_patience is not None
            else preset.get("early_stopping_patience")
        ),
        data_root=args.data_root,
        protocol_root=args.protocol_root,
        output_root=args.output_root,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
    )


def _empty_history() -> Dict[str, List[float]]:
    return {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_eer": [],
        "lr": [],
    }


def _default_best_metrics() -> Dict[str, float]:
    return {"val_eer": math.inf, "val_loss": math.inf, "val_accuracy": 0.0, "epoch": 0}


def _model_kwargs_for_signature(model_name: str) -> Dict[str, object]:
    return copy.deepcopy(DEFAULT_MODEL_KWARGS.get(model_name, {}))


def _architecture_signature_payload(config: ExperimentConfig) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model_name": config.model,
        "feature_name": config.feature,
        "in_channels": 1,
        "num_classes": len(CLASS_MAPPING),
        "class_mapping": CLASS_MAPPING,
        "target_frames": config.target_frames,
    }
    model_kwargs = _model_kwargs_for_signature(config.model)
    if model_kwargs:
        payload["model_kwargs"] = model_kwargs
    return payload


def _architecture_signature_defaults(config: ExperimentConfig) -> Dict[str, object]:
    defaults: Dict[str, object] = {
        "in_channels": 1,
        "num_classes": len(CLASS_MAPPING),
        "class_mapping": CLASS_MAPPING,
        "target_frames": 128,
    }
    model_kwargs = _model_kwargs_for_signature(config.model)
    if model_kwargs:
        defaults["model_kwargs"] = model_kwargs
    return defaults


def _resume_signature_payload(config: ExperimentConfig) -> Dict[str, object]:
    return {
        "normalize": config.normalize,
        "specaugment": config.specaugment,
        "amp": config.amp,
        "scheduler": config.scheduler,
        "step_size": config.step_size,
        "gamma": config.gamma,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "early_stopping_patience": config.early_stopping_patience,
        "seed": config.seed,
    }


def _resume_signature_defaults(config: ExperimentConfig) -> Dict[str, object]:
    preset = PROFILE_PRESETS.get(config.profile or "", {})
    return {
        "normalize": preset.get("normalize", False),
        "specaugment": preset.get("specaugment", False),
        "amp": preset.get("amp", False),
        "scheduler": preset.get("scheduler", "step"),
        "step_size": preset.get("step_size", 15),
        "gamma": preset.get("gamma", 0.5),
        "lr": preset.get("lr", 1e-3),
        "batch_size": preset.get("batch_size", _default_batch_size(config.model)),
        "early_stopping_patience": preset.get("early_stopping_patience"),
        "seed": 42,
    }


def _canonicalize_signature_value(value: Any, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        value_dict = dict(value)
        default_dict = default if isinstance(default, Mapping) else {}
        normalized: Dict[str, Any] = {}

        for key in sorted(set(value_dict) | set(default_dict)):
            if key not in value_dict:
                continue

            default_value = default_dict.get(key, _MISSING)
            normalized_value = _canonicalize_signature_value(value_dict[key], default_value)
            if default_value is not _MISSING:
                normalized_default = _canonicalize_signature_value(default_value, default_value)
                if normalized_value == normalized_default:
                    continue
            normalized[str(key)] = normalized_value
        return normalized

    if isinstance(value, list):
        default_list = default if isinstance(default, list) else []
        return [
            _canonicalize_signature_value(
                item,
                default_list[index] if index < len(default_list) else _MISSING,
            )
            for index, item in enumerate(value)
        ]

    return value


def _build_signature(raw_payload: Dict[str, object], defaults: Dict[str, object]) -> Dict[str, object]:
    canonical_payload = _canonicalize_signature_value(raw_payload, defaults)
    canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    hash_value = hashlib.sha256(f"{SIGNATURE_VERSION}{canonical_json}".encode("utf-8")).hexdigest()
    return {
        "signature_version": SIGNATURE_VERSION,
        "raw": copy.deepcopy(raw_payload),
        "canonical_json": canonical_json,
        "hash": hash_value,
    }


def build_architecture_signature(config: ExperimentConfig) -> Dict[str, object]:
    return _build_signature(
        raw_payload=_architecture_signature_payload(config),
        defaults=_architecture_signature_defaults(config),
    )


def build_resume_signature(config: ExperimentConfig) -> Dict[str, object]:
    return _build_signature(
        raw_payload=_resume_signature_payload(config),
        defaults=_resume_signature_defaults(config),
    )


def _copy_nested_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _copy_nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_nested_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _copy_state_dict_to_cpu(state_dict: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    if state_dict is None:
        return None
    return {name: _copy_nested_to_cpu(value) for name, value in state_dict.items()}


def checkpoint_dir_for_config(config: ExperimentConfig) -> Path:
    return Path(config.output_root) / "checkpoints" / f"{config.profile}_{config.model}_{config.feature}"


def checkpoint_paths_for_config(config: ExperimentConfig) -> Dict[str, Path]:
    checkpoint_dir = checkpoint_dir_for_config(config)
    return {
        "dir": checkpoint_dir,
        "last": checkpoint_dir / "last.ckpt",
        "best": checkpoint_dir / "best.ckpt",
    }


def _coerce_history(history: Optional[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    normalized = _empty_history()
    if not history:
        return normalized
    for key in normalized:
        normalized[key] = list(history.get(key, []))
    return normalized


def _coerce_best_metrics(payload: Mapping[str, Any]) -> Dict[str, float]:
    best_metrics = _default_best_metrics()
    saved = payload.get("best_val_metrics")
    if isinstance(saved, Mapping):
        for key in best_metrics:
            if key in saved:
                best_metrics[key] = float(saved[key]) if key != "epoch" else int(saved[key])

    if "best_eer" in payload:
        best_metrics["val_eer"] = float(payload["best_eer"])
    if "best_epoch" in payload:
        best_metrics["epoch"] = int(payload["best_epoch"])
    return best_metrics


def _compare_signature(
    *,
    checkpoint_signature: Mapping[str, Any],
    current_signature: Mapping[str, Any],
    signature_name: str,
) -> None:
    saved_version = str(checkpoint_signature.get("signature_version"))
    current_version = str(current_signature.get("signature_version"))
    saved_hash = str(checkpoint_signature.get("hash"))
    current_hash = str(current_signature.get("hash"))
    saved_canonical = str(checkpoint_signature.get("canonical_json"))
    current_canonical = str(current_signature.get("canonical_json"))

    if saved_version == current_version and saved_hash == current_hash:
        return

    mismatch_details = [
        f"{signature_name} mismatch while validating resume checkpoint.",
        f"saved_signature_version={saved_version}",
        f"current_signature_version={current_version}",
        f"saved_hash={saved_hash}",
        f"current_hash={current_hash}",
        f"saved_canonical_json={saved_canonical}",
        f"current_canonical_json={current_canonical}",
        "saved_raw_signature=" + json.dumps(checkpoint_signature.get("raw", {}), sort_keys=True, ensure_ascii=True),
        "current_raw_signature=" + json.dumps(current_signature.get("raw", {}), sort_keys=True, ensure_ascii=True),
    ]

    raise ValueError("\n".join(mismatch_details))


def _validate_resume_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    config: ExperimentConfig,
    architecture_signature: Mapping[str, Any],
    resume_signature: Mapping[str, Any],
) -> None:
    saved_profile = checkpoint.get("profile")
    if saved_profile != config.profile:
        raise ValueError(
            f"Resume checkpoint profile mismatch: saved='{saved_profile}' current='{config.profile}'"
        )

    saved_epoch = int(checkpoint.get("epoch", 0))
    if config.epochs < saved_epoch:
        raise ValueError(
            f"Requested epochs={config.epochs} is lower than saved epoch={saved_epoch} in last.ckpt"
        )

    checkpoint_architecture = checkpoint.get("architecture_signature")
    checkpoint_resume = checkpoint.get("resume_signature")
    if not isinstance(checkpoint_architecture, Mapping):
        raise ValueError("Resume checkpoint is missing architecture_signature metadata")
    if not isinstance(checkpoint_resume, Mapping):
        raise ValueError("Resume checkpoint is missing resume_signature metadata")

    _compare_signature(
        checkpoint_signature=checkpoint_architecture,
        current_signature=architecture_signature,
        signature_name="architecture_signature",
    )
    _compare_signature(
        checkpoint_signature=checkpoint_resume,
        current_signature=resume_signature,
        signature_name="resume_signature",
    )


def _checkpoint_metadata(
    *,
    config: ExperimentConfig,
    architecture_signature: Mapping[str, Any],
    training_complete: bool,
) -> Dict[str, object]:
    return {
        "model_name": config.model,
        "feature_name": config.feature,
        "profile": config.profile,
        "target_frames": config.target_frames,
        "normalize": config.normalize,
        "specaugment": config.specaugment,
        "amp": config.amp,
        "batch_size": config.batch_size,
        "scheduler": config.scheduler,
        "step_size": config.step_size,
        "gamma": config.gamma,
        "lr": config.lr,
        "epochs": config.epochs,
        "seed": config.seed,
        "class_mapping": CLASS_MAPPING,
        "training_complete": training_complete,
        "architecture_signature": copy.deepcopy(dict(architecture_signature)),
    }


def _save_last_checkpoint(
    *,
    path: Path,
    config: ExperimentConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_metrics: Mapping[str, float],
    best_state_dict: Mapping[str, Any] | None,
    history: Dict[str, List[float]],
    patience_counter: int,
    architecture_signature: Mapping[str, Any],
    resume_signature: Mapping[str, Any],
    training_complete: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_metadata(
        config=config,
        architecture_signature=architecture_signature,
        training_complete=training_complete,
    )
    payload.update(
        {
            "state_dict": _copy_state_dict_to_cpu(model.state_dict()),
            "optimizer_state_dict": _copy_nested_to_cpu(optimizer.state_dict()),
            "scheduler_state_dict": _copy_nested_to_cpu(scheduler.state_dict()),
            "scaler_state_dict": _copy_nested_to_cpu(scaler.state_dict()),
            "epoch": epoch,
            "best_eer": float(best_metrics["val_eer"]),
            "best_epoch": int(best_metrics["epoch"]),
            "best_val_metrics": copy.deepcopy(dict(best_metrics)),
            "best_state_dict": _copy_state_dict_to_cpu(best_state_dict),
            "history": copy.deepcopy(history),
            "patience_counter": patience_counter,
            "resume_signature": copy.deepcopy(dict(resume_signature)),
        }
    )
    torch.save(payload, path)


def _save_best_checkpoint(
    *,
    path: Path,
    config: ExperimentConfig,
    best_metrics: Mapping[str, float],
    best_state_dict: Mapping[str, Any] | None,
    architecture_signature: Mapping[str, Any],
    training_complete: bool,
) -> None:
    if best_state_dict is None:
        raise RuntimeError(
            f"Cannot write best checkpoint at {path} because best_state_dict is unavailable."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_metadata(
        config=config,
        architecture_signature=architecture_signature,
        training_complete=training_complete,
    )
    payload.update(
        {
            "state_dict": _copy_state_dict_to_cpu(best_state_dict),
            "best_eer": float(best_metrics["val_eer"]),
            "best_epoch": int(best_metrics["epoch"]),
            "best_val_metrics": copy.deepcopy(dict(best_metrics)),
            "epoch": int(best_metrics["epoch"]),
        }
    )
    torch.save(payload, path)


def _resolve_autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    return autocast()


def create_spoof_dataloader(
    *,
    feature_name: str,
    split: str,
    target_frames: int,
    batch_size: int,
    data_root: str,
    protocol_root: str | None = None,
    feature_root: str | None = None,
    normalize: bool = False,
    specaugment: bool = False,
    training: bool = False,
    num_workers: int = 0,
    device: torch.device | None = None,
    labels_path: str | None = None,
) -> DataLoader:
    if split == "eval_2021":
        samples = build_2021_samples(
            feature_name=feature_name,
            labels_path=labels_path,
            data_root=data_root,
            feature_root=feature_root,
        )
    else:
        samples = build_2019_samples(
            feature_name=feature_name,
            split=split,
            data_root=data_root,
            protocol_root=protocol_root,
            feature_root=feature_root,
        )

    dataset = SpoofDataset(
        samples=samples,
        target_frames=target_frames,
        training=training,
        normalize=normalize,
        apply_lfcc_specaugment=specaugment and training,
        feature_name=feature_name,
    )
    active_device = device or resolve_device(None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=active_device.type == "cuda",
    )


def create_scheduler(config: ExperimentConfig, optimizer: torch.optim.Optimizer):
    if config.scheduler == "step":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def run_evaluation_loop(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    use_amp: bool = False,
) -> EvaluationArtifacts:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: List[int] = []
    all_predictions: List[int] = []
    all_scores: List[float] = []
    all_utt_ids: List[str] = []

    with torch.no_grad():
        for inputs, labels, utt_ids in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with _resolve_autocast(device=device, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, labels) if criterion is not None else None

            probabilities = logits_to_spoof_scores(logits)
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            label_array = labels.detach().cpu().numpy()

            if loss is not None:
                total_loss += float(loss.item()) * labels.size(0)
                total_samples += labels.size(0)

            all_labels.extend(label_array.tolist())
            all_predictions.extend(predictions.tolist())
            all_scores.extend(probabilities.tolist())
            all_utt_ids.extend(list(utt_ids))

    accuracy = compute_accuracy(all_labels, all_predictions)
    eer, threshold = compute_eer(all_labels, all_scores)
    confusion = compute_confusion(all_labels, all_predictions)
    average_loss = total_loss / total_samples if criterion is not None and total_samples else None

    return EvaluationArtifacts(
        loss=average_loss,
        accuracy=accuracy,
        eer=eer,
        eer_threshold=threshold,
        confusion=confusion,
        labels=np.asarray(all_labels, dtype=np.int64),
        predictions=np.asarray(all_predictions, dtype=np.int64),
        scores=np.asarray(all_scores, dtype=np.float32),
        utt_ids=all_utt_ids,
    )


def _is_better(current_eer: float, current_loss: float, best_eer: float, best_loss: float) -> bool:
    current_eer_cmp = math.inf if math.isnan(current_eer) else current_eer
    best_eer_cmp = math.inf if math.isnan(best_eer) else best_eer
    if current_eer_cmp < best_eer_cmp:
        return True
    if math.isclose(current_eer_cmp, best_eer_cmp, rel_tol=1e-9, abs_tol=1e-9):
        return current_loss < best_loss
    return False


def upsert_rows(
    csv_path: str | Path,
    rows: List[Dict],
    key_columns: List[str],
    sort_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame(rows)

    if path.exists():
        current_frame = pd.read_csv(path)
    else:
        current_frame = pd.DataFrame(columns=new_frame.columns)

    if not current_frame.empty:
        shared_keys = [column for column in key_columns if column in current_frame.columns and column in new_frame.columns]
        if shared_keys:
            current_keys = current_frame[shared_keys].astype(str).agg("||".join, axis=1)
            new_keys = set(new_frame[shared_keys].astype(str).agg("||".join, axis=1))
            current_frame = current_frame.loc[~current_keys.isin(new_keys)]

    if current_frame.empty:
        combined = new_frame.copy()
    elif new_frame.empty:
        combined = current_frame.copy()
    else:
        combined = pd.concat([current_frame, new_frame], ignore_index=True)
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def load_checkpoint_bundle(checkpoint_path: str | Path, device: torch.device | str = "cpu"):
    bundle = torch.load(checkpoint_path, map_location=device)
    model = build_model(bundle["model_name"])
    model.load_state_dict(bundle["state_dict"])
    return model, bundle


def create_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return GradScaler(enabled=use_amp)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)

    for inputs, labels, _ in progress:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with _resolve_autocast(device=device, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, labels)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, total_samples)


def train_experiment(config: ExperimentConfig) -> Dict[str, object]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = config.amp and device.type == "cuda"
    checkpoint_paths = checkpoint_paths_for_config(config)
    architecture_signature = build_architecture_signature(config)
    resume_signature = build_resume_signature(config)

    train_loader = create_spoof_dataloader(
        feature_name=config.feature,
        split="train",
        target_frames=config.target_frames,
        batch_size=config.batch_size,
        data_root=config.data_root,
        protocol_root=config.protocol_root,
        normalize=config.normalize,
        specaugment=config.specaugment,
        training=True,
        num_workers=config.num_workers,
        device=device,
    )
    val_loader = create_spoof_dataloader(
        feature_name=config.feature,
        split="dev",
        target_frames=config.target_frames,
        batch_size=config.batch_size,
        data_root=config.data_root,
        protocol_root=config.protocol_root,
        normalize=config.normalize,
        specaugment=False,
        training=False,
        num_workers=config.num_workers,
        device=device,
    )

    model = build_model(config.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = create_scheduler(config, optimizer)
    scaler = create_grad_scaler(use_amp)

    history = _empty_history()
    best_state = None
    best_metrics = _default_best_metrics()
    patience_counter = 0
    start_epoch = 1

    if config.resume:
        if checkpoint_paths["last"].exists():
            resume_bundle = torch.load(checkpoint_paths["last"], map_location="cpu")
            _validate_resume_checkpoint(
                checkpoint=resume_bundle,
                config=config,
                architecture_signature=architecture_signature,
                resume_signature=resume_signature,
            )

            model.load_state_dict(resume_bundle["state_dict"])
            optimizer.load_state_dict(resume_bundle["optimizer_state_dict"])
            scheduler.load_state_dict(resume_bundle["scheduler_state_dict"])

            scaler_state = resume_bundle.get("scaler_state_dict")
            if scaler_state and use_amp:
                scaler.load_state_dict(scaler_state)

            start_epoch = int(resume_bundle["epoch"]) + 1
            history = _coerce_history(resume_bundle.get("history"))
            best_metrics = _coerce_best_metrics(resume_bundle)
            best_state = _copy_state_dict_to_cpu(resume_bundle.get("best_state_dict"))
            patience_counter = int(resume_bundle.get("patience_counter", 0))

            if not checkpoint_paths["best"].exists():
                print(
                    f"Warning: missing best checkpoint at {checkpoint_paths['best']}. "
                    "Resuming from last.ckpt and recreating best.ckpt later."
                )
        else:
            print(f"No last checkpoint found at {checkpoint_paths['last']}. Starting fresh training.")

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            epochs=config.epochs,
        )
        val_metrics = run_evaluation_loop(
            model=model,
            dataloader=val_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics.loss if val_metrics.loss is not None else float("nan")))
        history["val_accuracy"].append(val_metrics.accuracy)
        history["val_eer"].append(val_metrics.eer)
        history["lr"].append(current_lr)

        improved = _is_better(
            current_eer=val_metrics.eer,
            current_loss=float(val_metrics.loss if val_metrics.loss is not None else math.inf),
            best_eer=float(best_metrics["val_eer"]),
            best_loss=float(best_metrics["val_loss"]),
        )
        if improved:
            best_state = _copy_state_dict_to_cpu(model.state_dict())
            best_metrics = {
                "val_eer": float(val_metrics.eer),
                "val_loss": float(val_metrics.loss if val_metrics.loss is not None else math.inf),
                "val_accuracy": float(val_metrics.accuracy),
                "epoch": epoch,
            }
            patience_counter = 0
            _save_best_checkpoint(
                path=checkpoint_paths["best"],
                config=config,
                best_metrics=best_metrics,
                best_state_dict=best_state,
                architecture_signature=architecture_signature,
                training_complete=False,
            )
        else:
            patience_counter += 1

        scheduler.step()
        _save_last_checkpoint(
            path=checkpoint_paths["last"],
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metrics=best_metrics,
            best_state_dict=best_state,
            history=history,
            patience_counter=patience_counter,
            architecture_signature=architecture_signature,
            resume_signature=resume_signature,
            training_complete=False,
        )

        if config.early_stopping_patience is not None and patience_counter >= config.early_stopping_patience:
            break

    output_root = Path(config.output_root)
    if best_state is None and checkpoint_paths["best"].exists():
        best_bundle = torch.load(checkpoint_paths["best"], map_location="cpu")
        best_state = _copy_state_dict_to_cpu(best_bundle.get("state_dict"))
        best_metrics = _coerce_best_metrics(best_bundle)

    if best_state is None:
        raise RuntimeError(
            "best_state_dict is None at finalization while best.ckpt is missing or invalid. "
            "Cannot create a valid best checkpoint."
        )

    _save_best_checkpoint(
        path=checkpoint_paths["best"],
        config=config,
        best_metrics=best_metrics,
        best_state_dict=best_state,
        architecture_signature=architecture_signature,
        training_complete=True,
    )

    completed_epoch = max(best_metrics["epoch"], start_epoch - 1)
    if history["train_loss"]:
        completed_epoch = len(history["train_loss"])
    _save_last_checkpoint(
        path=checkpoint_paths["last"],
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=completed_epoch,
        best_metrics=best_metrics,
        best_state_dict=best_state,
        history=history,
        patience_counter=patience_counter,
        architecture_signature=architecture_signature,
        resume_signature=resume_signature,
        training_complete=True,
    )

    save_learning_curve(
        train_losses=history["train_loss"],
        val_losses=history["val_loss"],
        output_path=output_root / "figures" / f"{config.profile}_learning_curve.png",
        title=f"{config.profile.title()} ({config.model.upper()} + {config.feature.upper()})",
    )

    log_rows = []
    for epoch_index, (train_loss, val_loss, val_accuracy, val_eer, lr_value) in enumerate(
        zip(
            history["train_loss"],
            history["val_loss"],
            history["val_accuracy"],
            history["val_eer"],
            history["lr"],
        ),
        start=1,
    ):
        log_rows.append(
            {
                "profile": config.profile,
                "model": config.model,
                "feature": config.feature,
                "epoch": epoch_index,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_eer": val_eer,
                "lr": lr_value,
            }
        )
    upsert_rows(
        csv_path=output_root / "results" / "train_log.csv",
        rows=log_rows,
        key_columns=["profile", "model", "feature", "epoch"],
        sort_columns=["profile", "model", "feature", "epoch"],
    )

    return {
        "checkpoint_path": str(checkpoint_paths["best"]),
        "last_checkpoint_path": str(checkpoint_paths["last"]),
        "best_val_metrics": best_metrics,
        "history": history,
        "config": config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train voice spoofing detection models.")
    parser.add_argument("--profile", choices=["baseline", "optimized"], default=None)
    parser.add_argument("--model", choices=["cnn", "resnet", "lcnn"], default=None)
    parser.add_argument("--feature", choices=["mfcc", "lfcc", "spectrogram"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler", choices=["step", "cosine"], default=None)
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--target_frames", type=int, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--protocol_root", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=None)

    parser.add_argument("--specaugment", dest="specaugment", action="store_true")
    parser.add_argument("--no_specaugment", dest="specaugment", action="store_false")
    parser.set_defaults(specaugment=None)

    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_experiment_config(args)
    result = train_experiment(config)
    best_metrics = result["best_val_metrics"]
    print(
        f"Saved checkpoint to {result['checkpoint_path']} | "
        f"best_dev_eer={best_metrics['val_eer']:.4f} | "
        f"best_dev_accuracy={best_metrics['val_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
