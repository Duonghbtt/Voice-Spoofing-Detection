from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from src.models import build_model, canonicalize_model_name
from src.utils.dataset import SpoofDataset, build_2019_samples, build_2021_samples, canonicalize_feature_name
from src.utils.metrics import EvaluationArtifacts, compute_accuracy, compute_confusion, compute_eer, logits_to_spoof_scores


DEFAULT_BATCH_SIZE_MATRIX = {
    ("cnn", "mfcc"): 64,
    ("cnn", "lfcc"): 64,
    ("cnn", "spectrogram"): 32,
    ("lcnn", "mfcc"): 32,
    ("lcnn", "lfcc"): 32,
    ("lcnn", "spectrogram"): 16,
    ("resnet18", "mfcc"): 32,
    ("resnet18", "lfcc"): 32,
    ("resnet18", "spectrogram"): 8,
}
CUDA_4GB_BATCH_SIZE_MATRIX = {
    ("cnn", "mfcc"): 256,
    ("cnn", "lfcc"): 256,
    ("cnn", "spectrogram"): 160,
    ("lcnn", "mfcc"): 192,
    ("lcnn", "lfcc"): 192,
    ("lcnn", "spectrogram"): 128,
    ("resnet18", "mfcc"): 256,
    ("resnet18", "lfcc"): 256,
    ("resnet18", "spectrogram"): 128,
}
WINDOWS_CUDA_4GB_BATCH_SIZE_MATRIX = {
    ("cnn", "mfcc"): 128,
    ("cnn", "lfcc"): 128,
    ("cnn", "spectrogram"): 80,
    ("lcnn", "mfcc"): 96,
    ("lcnn", "lfcc"): 96,
    ("lcnn", "spectrogram"): 64,
    ("resnet18", "mfcc"): 128,
    ("resnet18", "lfcc"): 128,
    ("resnet18", "spectrogram"): 64,
}
CUDA_4GB_MIN_FREE_MEMORY_GB = 3.0
NON_WINDOWS_MAX_WORKERS = 4
WINDOWS_AUTO_WORKERS = 0
WINDOWS_MAX_WORKERS = 2
CUDA_PREFETCH_FACTOR = 4


def _is_windows_host() -> bool:
    return os.name == "nt"


def _resolve_cuda_device_index(device_name: str | torch.device | None) -> int:
    resolved_device = device_name if isinstance(device_name, torch.device) else resolve_device(device_name)
    if resolved_device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, received {resolved_device}.")
    if resolved_device.index is not None:
        return int(resolved_device.index)
    return int(torch.cuda.current_device())


def _cuda_total_memory_gb(device_name: str | torch.device | None) -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        device_index = _resolve_cuda_device_index(device_name)
        properties = torch.cuda.get_device_properties(device_index)
        return float(properties.total_memory) / float(1024**3)
    except Exception:
        return None


def _cuda_free_memory_gb(device_name: str | torch.device | None) -> float | None:
    if not torch.cuda.is_available():
        return None

    mem_get_info = getattr(torch.cuda, "mem_get_info", None)
    if mem_get_info is None:
        return None

    try:
        device_index = _resolve_cuda_device_index(device_name)
        try:
            free_bytes, _ = mem_get_info(device_index)
        except TypeError:
            with torch.cuda.device(device_index):
                free_bytes, _ = mem_get_info()
        return float(free_bytes) / float(1024**3)
    except Exception:
        return None


def default_batch_size(
    model_name: str,
    feature_name: str,
    device_name: str | torch.device | None = None,
) -> int:
    canonical_model = canonicalize_model_name(model_name)
    canonical_feature = canonicalize_feature_name(feature_name)
    base_batch_size = DEFAULT_BATCH_SIZE_MATRIX[(canonical_model, canonical_feature)]

    resolved_device = resolve_device(device_name)
    if resolved_device.type != "cuda":
        return base_batch_size

    total_memory_gb = _cuda_total_memory_gb(resolved_device)
    free_memory_gb = _cuda_free_memory_gb(resolved_device)
    if (
        total_memory_gb is not None
        and total_memory_gb >= 3.5
        and (free_memory_gb is None or free_memory_gb >= CUDA_4GB_MIN_FREE_MEMORY_GB)
    ):
        batch_size_matrix = WINDOWS_CUDA_4GB_BATCH_SIZE_MATRIX if _is_windows_host() else CUDA_4GB_BATCH_SIZE_MATRIX
        return batch_size_matrix.get((canonical_model, canonical_feature), base_batch_size)
    return base_batch_size


def configure_torch_runtime(device_name: str | torch.device | None = None) -> None:
    resolved_device = resolve_device(device_name)
    if resolved_device.type != "cuda":
        return

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _default_worker_count() -> int:
    if _is_windows_host():
        return WINDOWS_AUTO_WORKERS

    cpu_count = os.cpu_count() or 0
    return max(0, min(8, cpu_count - 1))


def resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return _default_worker_count()
    if num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if _is_windows_host():
        if num_workers > WINDOWS_MAX_WORKERS:
            warnings.warn(
                f"Windows explicit num_workers={num_workers} may be unstable; values above {WINDOWS_MAX_WORKERS} can trigger shared-memory failures.",
                UserWarning,
                stacklevel=2,
            )
        return num_workers
    return min(num_workers, NON_WINDOWS_MAX_WORKERS)


def resolve_device(device_name: str | torch.device | None = None) -> torch.device:
    if isinstance(device_name, torch.device):
        return device_name
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def experiment_dir_name(model_name: str, feature_name: str) -> str:
    canonical_model = canonicalize_model_name(model_name)
    canonical_feature = canonicalize_feature_name(feature_name)
    return f"{canonical_model}_{canonical_feature}"


def experiment_output_paths(output_root: str | Path, model_name: str, feature_name: str) -> Dict[str, Path]:
    experiment_name = experiment_dir_name(model_name=model_name, feature_name=feature_name)
    root = Path(output_root)
    return {
        "experiment": experiment_name,
        "checkpoint_dir": root / "checkpoints" / experiment_name,
        "result_dir": root / "results" / experiment_name,
        "figure_dir": root / "figures" / experiment_name,
    }


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
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": training and len(dataset) > 0,
        "num_workers": num_workers,
        "pin_memory": active_device.type == "cuda" and not _is_windows_host(),
    }
    if num_workers > 0 and not _is_windows_host():
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = CUDA_PREFETCH_FACTOR if active_device.type == "cuda" else 2

    return DataLoader(
        dataset,
        **dataloader_kwargs,
    )


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
        for batch_index, (inputs, labels, utt_ids) in enumerate(dataloader):
            transfer_non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=transfer_non_blocking)
            labels = labels.to(device, non_blocking=transfer_non_blocking)

            with _resolve_autocast(device=device, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, labels) if criterion is not None else None

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "Non-finite logits detected during evaluation "
                    f"(batch_index={batch_index}, utt_ids={list(utt_ids)[:5]})."
                )
            probabilities = logits_to_spoof_scores(logits)
            if not np.isfinite(probabilities).all():
                raise RuntimeError(
                    "Non-finite spoof scores detected during evaluation "
                    f"(batch_index={batch_index}, utt_ids={list(utt_ids)[:5]})."
                )
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            label_array = labels.detach().cpu().numpy()

            if loss is not None:
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite loss detected during evaluation "
                        f"(batch_index={batch_index}, utt_ids={list(utt_ids)[:5]})."
                    )
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
    bundle["model_name"] = canonicalize_model_name(str(bundle["model_name"]))
    model = build_model(bundle["model_name"])
    model.load_state_dict(bundle["state_dict"])
    return model, bundle
