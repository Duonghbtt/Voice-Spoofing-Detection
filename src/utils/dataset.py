from __future__ import annotations

import csv
import logging
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)
LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
    "0": 0,
    "1": 1,
    0: 0,
    1: 1,
}
PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
SUPPORTED_SPLITS = ("train", "dev", "eval", "eval_2021")
FEATURE_LAYOUTS = {
    "mfcc": {
        "train": {"bundle": "output_npy_2019", "feature_dir": "output_mfcc", "split_dir": "train", "labels_file": "labels_train.csv"},
        "dev": {"bundle": "output_npy_2019", "feature_dir": "output_mfcc", "split_dir": "dev", "labels_file": "labels_dev.csv"},
        "eval": {"bundle": "output_npy_2019", "feature_dir": "output_mfcc", "split_dir": "eval", "labels_file": "labels_eval.csv"},
        "eval_2021": {"bundle": "output_npy_2021", "feature_dir": "output_mfcc", "split_dir": "eval_2021", "labels_file": "labels_eval_2021.csv"},
    },
    "lfcc": {
        "train": {"bundle": "output_npy_2019", "feature_dir": "output_lfcc", "split_dir": "train", "labels_file": "labels_train.csv"},
        "dev": {"bundle": "output_npy_2019", "feature_dir": "output_lfcc", "split_dir": "dev", "labels_file": "labels_dev.csv"},
        "eval": {"bundle": "output_npy_2019", "feature_dir": "output_lfcc", "split_dir": "eval", "labels_file": "labels_eval.csv"},
        "eval_2021": {"bundle": "output_npy_2021", "feature_dir": "output_lfcc", "split_dir": "eval_2021", "labels_file": "labels_eval_2021.csv"},
    },
    "spectrogram": {
        "train": {"bundle": "output_spectrogram_2019", "feature_dir": None, "split_dir": "train", "labels_file": "labels_train.csv"},
        "dev": {"bundle": "output_spectrogram_2019", "feature_dir": None, "split_dir": "dev", "labels_file": "labels_dev.csv"},
        "eval": {"bundle": "output_spectrogram_2019", "feature_dir": None, "split_dir": "eval", "labels_file": "labels_eval.csv"},
        "eval_2021": {"bundle": "output_spectrogram_2021", "feature_dir": None, "split_dir": "eval", "labels_file": "labels_eval_2021.csv"},
    },
}
FEATURE_NAME_ALIASES = {
    "mfcc": "mfcc",
    "output_mfcc": "mfcc",
    "lfcc": "lfcc",
    "output_lfcc": "lfcc",
    "spectrogram": "spectrogram",
    "spec": "spectrogram",
    "output_spec": "spectrogram",
    "output_spectrogram": "spectrogram",
}
INVALID_DATA_ROOT_NAMES = {
    "features",
    "output_npy_2019",
    "output_npy_2021",
    "output_spectrogram_2019",
    "output_spectrogram_2021",
}


@dataclass(frozen=True)
class SampleRecord:
    utt_id: str
    path: Path
    label: int


def canonicalize_label(label: object) -> int:
    if label not in LABEL_MAP:
        normalized = str(label).strip().lower()
        if normalized not in LABEL_MAP:
            raise ValueError(f"Unsupported label value: {label}")
        return LABEL_MAP[normalized]
    return LABEL_MAP[label]


def canonicalize_feature_name(feature_name: str) -> str:
    normalized = feature_name.strip().lower()
    if normalized not in FEATURE_NAME_ALIASES:
        raise ValueError(f"Unsupported feature name: {feature_name}")
    return FEATURE_NAME_ALIASES[normalized]


def resolve_data_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    trailing_parts = [part.lower() for part in root.parts[-4:]]
    if root.name.lower() in INVALID_DATA_ROOT_NAMES or any(part in INVALID_DATA_ROOT_NAMES for part in trailing_parts):
        raise ValueError(
            f"--data_root must point to the parent data directory (for example 'data'), not '{root}'."
        )
    return root


def _resolve_feature_layout(feature_name: str, split: str) -> Dict[str, str | None]:
    canonical_feature = canonicalize_feature_name(feature_name)
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {', '.join(SUPPORTED_SPLITS)}")
    return FEATURE_LAYOUTS[canonical_feature][split]


def _is_explicit_split_directory(path: Path, split_dir_name: str) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if path.name.lower() == split_dir_name.lower():
        return True
    return any(path.glob("*.npy"))


def _candidate_feature_directories(root: Path, layout: Dict[str, str | None]) -> List[Path]:
    bundle_name = str(layout["bundle"])
    feature_dir_name = layout["feature_dir"]
    split_dir_name = str(layout["split_dir"])
    candidates: List[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    add(root)
    add(root / split_dir_name)

    if feature_dir_name is not None:
        add(root / feature_dir_name)
        add(root / feature_dir_name / split_dir_name)

    add(root / bundle_name)
    add(root / bundle_name / split_dir_name)

    if feature_dir_name is not None:
        add(root / bundle_name / feature_dir_name)
        add(root / bundle_name / feature_dir_name / split_dir_name)

    add(root / "features")
    add(root / "features" / bundle_name)
    add(root / "features" / bundle_name / split_dir_name)

    if feature_dir_name is not None:
        add(root / "features" / bundle_name / feature_dir_name)
        add(root / "features" / bundle_name / feature_dir_name / split_dir_name)

    return candidates


def resolve_feature_directory(
    feature_name: str,
    split: str,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> Path:
    layout = _resolve_feature_layout(feature_name=feature_name, split=split)

    if feature_root is None:
        base_root = resolve_data_root(data_root) / "features"
        feature_dir = base_root / str(layout["bundle"])
        if layout["feature_dir"] is not None:
            feature_dir = feature_dir / str(layout["feature_dir"])
        feature_dir = feature_dir / str(layout["split_dir"])
        if feature_dir.exists() and feature_dir.is_dir():
            return feature_dir
        raise FileNotFoundError(
            f"Could not locate feature directory for feature='{feature_name}', split='{split}' at '{feature_dir}'."
        )

    explicit_root = Path(feature_root)
    for candidate in _candidate_feature_directories(explicit_root, layout):
        if _is_explicit_split_directory(candidate, str(layout["split_dir"])):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve feature directory for feature='{feature_name}', split='{split}' from explicit root '{explicit_root}'."
    )


def resolve_labels_csv_path(
    feature_name: str,
    split: str,
    *,
    labels_path: str | Path | None = None,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> Path | None:
    resolved_labels_path, explicit_path, expected_path = _discover_labels_csv_path(
        feature_name=feature_name,
        split=split,
        labels_path=labels_path,
        data_root=data_root,
        feature_root=feature_root,
    )

    if labels_path is not None and explicit_path is not None and resolved_labels_path != explicit_path:
        LOGGER.warning("Khong tim thay file nhan CSV duoc chi dinh: %s. Se thu tu dong resolve.", explicit_path)
    if resolved_labels_path is not None:
        return resolved_labels_path

    LOGGER.warning(
        "Khong tim thay file nhan CSV cho feature='%s', split='%s' tai '%s'.",
        canonicalize_feature_name(feature_name),
        split,
        expected_path,
    )
    return None


def _protocol_search_roots(data_root: Path, protocol_root: str | Path | None) -> List[Path]:
    if protocol_root is not None:
        return [Path(protocol_root)]

    return [
        data_root / "raw" / "LA",
        data_root / "raw" / "ASVspoof2019" / "LA",
        data_root / "raw",
        data_root,
    ]


def _resolve_protocol_file(
    split: str,
    data_root: Path,
    protocol_root: str | Path | None = None,
) -> Optional[Path]:
    protocol_name = PROTOCOL_FILES[split]
    for root in _protocol_search_roots(data_root, protocol_root):
        candidates = [
            root / protocol_name,
            root / "ASVspoof2019_LA_cm_protocols" / protocol_name,
            root / "LA" / "ASVspoof2019_LA_cm_protocols" / protocol_name,
            root / "ASVspoof2019" / "LA" / "ASVspoof2019_LA_cm_protocols" / protocol_name,
        ]
        file_candidate = next((candidate for candidate in candidates if candidate.exists()), None)
        if file_candidate is not None:
            return file_candidate

        recursive_matches = list(root.rglob(protocol_name)) if root.exists() else []
        if recursive_matches:
            return recursive_matches[0]
    return None


def _zip_candidates(data_root: Path, protocol_root: str | Path | None) -> List[Path]:
    if protocol_root is not None:
        root = Path(protocol_root)
        return [root / "LA.zip", root] if root.suffix == ".zip" else [root / "LA.zip"]
    return [data_root / "raw" / "LA.zip", data_root / "LA.zip"]


def load_protocol_labels(
    split: str,
    data_root: str | Path = "data",
    protocol_root: str | Path | None = None,
) -> Dict[str, int]:
    if split not in PROTOCOL_FILES:
        raise ValueError(f"Unsupported ASVspoof2019 split: {split}")

    resolved_data_root = resolve_data_root(data_root)
    labels: Dict[str, int] = {}
    protocol_path = _resolve_protocol_file(split, resolved_data_root, protocol_root)
    if protocol_path is not None:
        with protocol_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = raw_line.strip().split()
                if len(parts) < 2:
                    continue
                utt_id = parts[1]
                labels[utt_id] = canonicalize_label(parts[-1])
        return labels

    protocol_name = PROTOCOL_FILES[split]
    for zip_path in _zip_candidates(resolved_data_root, protocol_root):
        if not zip_path.exists() or zip_path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(zip_path, "r") as archive:
            matched_name = next((name for name in archive.namelist() if name.endswith(protocol_name)), None)
            if matched_name is None:
                continue
            with archive.open(matched_name, "r") as handle:
                for raw_line in handle:
                    parts = raw_line.decode("utf-8").strip().split()
                    if len(parts) < 2:
                        continue
                    utt_id = parts[1]
                    labels[utt_id] = canonicalize_label(parts[-1])
            return labels

    raise FileNotFoundError(
        f"Could not locate ASVspoof2019 protocol file '{protocol_name}'. "
        f"Looked in protocol_root={protocol_root!r} and data_root={resolved_data_root}"
    )


def _discover_labels_csv_path(
    feature_name: str,
    split: str,
    *,
    labels_path: str | Path | None = None,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
    feature_dir: Path | None = None,
) -> tuple[Path | None, Path | None, Path]:
    explicit_path = Path(labels_path) if labels_path is not None else None
    if explicit_path is not None and explicit_path.exists() and explicit_path.is_file():
        return explicit_path, explicit_path, explicit_path

    resolved_feature_dir = feature_dir or resolve_feature_directory(
        feature_name=feature_name,
        split=split,
        data_root=data_root,
        feature_root=feature_root,
    )
    expected_path = resolved_feature_dir.parent / str(_resolve_feature_layout(feature_name=feature_name, split=split)["labels_file"])
    if expected_path.exists() and expected_path.is_file():
        return expected_path, explicit_path, expected_path
    return None, explicit_path, expected_path


def _load_labels_from_csv_file(labels_path: Path, *, warn: bool = True) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            if warn:
                LOGGER.warning("File nhan CSV '%s' khong co header hop le.", labels_path)
            return labels

        normalized_fieldnames = {field.strip().lower(): field for field in reader.fieldnames}
        utt_field = next(
            (
                normalized_fieldnames[name]
                for name in ("utt_id", "filename", "file", "utterance_id")
                if name in normalized_fieldnames
            ),
            None,
        )
        label_field = normalized_fieldnames.get("label")
        if utt_field is None or label_field is None:
            if warn:
                LOGGER.warning("File nhan CSV '%s' thieu cot utt_id/filename hoac label.", labels_path)
            return labels

        for row in reader:
            utt_raw = str(row.get(utt_field, "")).strip()
            label_raw = row.get(label_field)
            if not utt_raw:
                continue
            if label_raw is None or str(label_raw).strip() == "":
                continue
            labels[Path(utt_raw).stem] = canonicalize_label(label_raw)
    return labels


def load_feature_csv_labels(
    feature_name: str,
    split: str,
    *,
    labels_path: str | Path | None = None,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> Dict[str, int]:
    resolved_labels_path = resolve_labels_csv_path(
        feature_name=feature_name,
        split=split,
        labels_path=labels_path,
        data_root=data_root,
        feature_root=feature_root,
    )
    if resolved_labels_path is None:
        return {}
    return _load_labels_from_csv_file(resolved_labels_path)


def _resolve_2019_labels(
    *,
    feature_name: str,
    split: str,
    feature_dir: Path,
    data_root: str | Path = "data",
    protocol_root: str | Path | None = None,
    feature_root: str | Path | None = None,
) -> Dict[str, int]:
    feature_ids = {feature_path.stem for feature_path in feature_dir.glob("*.npy")}
    csv_path, _, _ = _discover_labels_csv_path(
        feature_name=feature_name,
        split=split,
        data_root=data_root,
        feature_root=feature_root,
        feature_dir=feature_dir,
    )
    csv_issue = f"Khong tim thay labels CSV hop le cho split='{split}'."

    if csv_path is not None:
        try:
            csv_labels = _load_labels_from_csv_file(csv_path, warn=False)
            if not csv_labels:
                raise ValueError(f"File nhan CSV '{csv_path}' khong co ban ghi hop le.")
            matched_ids = feature_ids.intersection(csv_labels)
            if not matched_ids:
                raise ValueError(f"File nhan CSV '{csv_path}' khong khop utt_id voi feature trong '{feature_dir}'.")
            return csv_labels
        except Exception as error:
            csv_issue = str(error)

    try:
        protocol_labels = load_protocol_labels(split=split, data_root=data_root, protocol_root=protocol_root)
        # CSV khong dung duoc thi moi fallback sang protocol.
        LOGGER.warning(
            "CSV khong hop le, fallback sang protocol cho split='%s'. Chi tiet: %s",
            split,
            csv_issue,
        )
        return protocol_labels
    except FileNotFoundError as error:
        raise RuntimeError(f"Khong co nguon nhan hop le cho split='{split}'.") from error


def resolve_external_labels_path(
    labels_path: str | Path | None = None,
    *,
    feature_name: str = "mfcc",
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> Path | None:
    return resolve_labels_csv_path(
        feature_name=feature_name,
        split="eval_2021",
        labels_path=labels_path,
        data_root=data_root,
        feature_root=feature_root,
    )


def load_external_labels(
    labels_path: str | Path | None = None,
    *,
    feature_name: str = "mfcc",
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> Dict[str, int]:
    return load_feature_csv_labels(
        feature_name=feature_name,
        split="eval_2021",
        labels_path=labels_path,
        data_root=data_root,
        feature_root=feature_root,
    )


def _preview_ids(values: Sequence[str], max_items: int = 5) -> str:
    return ", ".join(list(values)[:max_items])


def _records_from_directory(feature_dir: Path, labels: Dict[str, int]) -> List[SampleRecord]:
    samples: List[SampleRecord] = []
    missing_label_ids: List[str] = []
    feature_ids: List[str] = []

    for feature_path in sorted(feature_dir.glob("*.npy")):
        utt_id = feature_path.stem
        feature_ids.append(utt_id)
        if utt_id not in labels:
            missing_label_ids.append(utt_id)
            continue
        samples.append(SampleRecord(utt_id=utt_id, path=feature_path, label=labels[utt_id]))

    extra_label_ids = sorted(set(labels) - set(feature_ids))
    if missing_label_ids:
        LOGGER.warning(
            "Bo qua %d file .npy trong '%s' vi khong tim thay nhan. Vi du: %s",
            len(missing_label_ids),
            feature_dir,
            _preview_ids(missing_label_ids),
        )
    if extra_label_ids:
        LOGGER.warning(
            "Co %d utt_id trong file nhan nhung khong co file .npy trong '%s'. Vi du: %s",
            len(extra_label_ids),
            feature_dir,
            _preview_ids(extra_label_ids),
        )
    if not samples:
        raise RuntimeError(f"Khong co sample co nhan hop le trong {feature_dir}")
    return samples


def build_2019_samples(
    feature_name: str,
    split: str,
    data_root: str | Path = "data",
    protocol_root: str | Path | None = None,
    feature_root: str | Path | None = None,
) -> List[SampleRecord]:
    if split not in PROTOCOL_FILES:
        raise ValueError(f"Unsupported ASVspoof2019 split: {split}")

    feature_dir = resolve_feature_directory(
        feature_name=feature_name,
        split=split,
        data_root=data_root,
        feature_root=feature_root,
    )
    labels = _resolve_2019_labels(
        feature_name=feature_name,
        split=split,
        feature_dir=feature_dir,
        data_root=data_root,
        protocol_root=protocol_root,
        feature_root=feature_root,
    )
    return _records_from_directory(feature_dir=feature_dir, labels=labels)


def build_2021_samples(
    feature_name: str,
    labels_path: str | Path | None = None,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> List[SampleRecord]:
    labels = load_external_labels(
        labels_path,
        feature_name=feature_name,
        data_root=data_root,
        feature_root=feature_root,
    )
    feature_dir = resolve_feature_directory(
        feature_name=feature_name,
        split="eval_2021",
        data_root=data_root,
        feature_root=feature_root,
    )
    return _records_from_directory(feature_dir=feature_dir, labels=labels)


def _pad_or_crop(feature: np.ndarray, target_frames: int, training: bool) -> np.ndarray:
    current_frames = feature.shape[1]
    if current_frames == target_frames:
        return feature

    if current_frames > target_frames:
        if training:
            start = random.randint(0, current_frames - target_frames)
        else:
            start = (current_frames - target_frames) // 2
        return feature[:, start : start + target_frames]

    pad_width = target_frames - current_frames
    return np.pad(feature, ((0, 0), (0, pad_width)), mode="constant")


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    mean = float(feature.mean())
    std = float(feature.std())
    if std < 1e-6:
        std = 1.0
    return (feature - mean) / std


def _sample_mask_width(max_width: int, length: int) -> int:
    if length <= 1 or max_width <= 0:
        return 0
    width = random.randint(1, min(max_width, length - 1))
    return width


def apply_specaugment(
    feature: np.ndarray,
    num_time_masks: int = 2,
    num_freq_masks: int = 1,
    max_time_mask_pct: float = 0.12,
    max_freq_mask_pct: float = 0.15,
) -> np.ndarray:
    augmented = feature.copy()
    num_freq_bins, num_frames = augmented.shape

    max_time_width = max(1, int(num_frames * max_time_mask_pct))
    max_freq_width = max(1, int(num_freq_bins * max_freq_mask_pct))

    for _ in range(num_time_masks):
        width = _sample_mask_width(max_time_width, num_frames)
        if width == 0:
            continue
        start = random.randint(0, num_frames - width)
        augmented[:, start : start + width] = 0.0

    for _ in range(num_freq_masks):
        width = _sample_mask_width(max_freq_width, num_freq_bins)
        if width == 0:
            continue
        start = random.randint(0, num_freq_bins - width)
        augmented[start : start + width, :] = 0.0

    return augmented


class SpoofDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleRecord],
        target_frames: int = 128,
        training: bool = False,
        normalize: bool = False,
        apply_lfcc_specaugment: bool = False,
        feature_name: str = "mfcc",
    ) -> None:
        self.samples = list(samples)
        self.target_frames = target_frames
        self.training = training
        self.normalize = normalize
        self.apply_lfcc_specaugment = apply_lfcc_specaugment
        self.feature_name = canonicalize_feature_name(feature_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        feature = np.load(sample.path).astype(np.float32)
        if feature.ndim != 2:
            raise ValueError(f"Expected 2D feature for {sample.path}, got shape {feature.shape}")

        feature = _pad_or_crop(feature, target_frames=self.target_frames, training=self.training)
        if self.normalize:
            feature = _normalize_feature(feature)
        if self.training and self.apply_lfcc_specaugment and self.feature_name == "lfcc":
            feature = apply_specaugment(feature)

        tensor = torch.from_numpy(feature).unsqueeze(0)
        label = torch.tensor(sample.label, dtype=torch.long)
        return tensor, label, sample.utt_id
