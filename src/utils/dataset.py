from __future__ import annotations

import csv
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


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
EXTERNAL_2021_LABELS_FILE = "labels_eval_2021.csv"
CANONICAL_2019_BUNDLE = "output_npy_2019"
CANONICAL_2021_BUNDLE = "output_npy_2021"
CANONICAL_FEATURE_DIRS = {
    "mfcc": "output_mfcc",
    "output_mfcc": "output_mfcc",
    "lfcc": "output_lfcc",
    "output_lfcc": "output_lfcc",
    "spectrogram": "output_spectrogram",
    "spec": "output_spectrogram",
    "output_spec": "output_spectrogram",
    "output_spectrogram": "output_spectrogram",
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


def resolve_data_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    trailing_parts = [part.lower() for part in root.parts[-4:]]
    if root.name.lower() == "features" or any(
        part in {CANONICAL_2019_BUNDLE, CANONICAL_2021_BUNDLE} for part in trailing_parts
    ):
        raise ValueError(
            f"--data_root must point to the parent data directory (for example 'data'), not '{root}'."
        )
    return root


def _canonical_feature_directory_name(feature_name: str) -> str:
    normalized = feature_name.strip().lower()
    if normalized in CANONICAL_FEATURE_DIRS:
        return CANONICAL_FEATURE_DIRS[normalized]
    if normalized.startswith("output_"):
        return normalized
    return f"output_{normalized}"


def _expected_feature_directory(data_root: Path, feature_name: str, split: str) -> Path:
    feature_dir = _canonical_feature_directory_name(feature_name)
    if split == "eval_2021":
        return data_root / "features" / CANONICAL_2021_BUNDLE / feature_dir / "eval_2021"
    if split in PROTOCOL_FILES:
        return data_root / "features" / CANONICAL_2019_BUNDLE / feature_dir / split
    raise ValueError(f"Unsupported split '{split}'. Expected one of: train, dev, eval, eval_2021")


def _expected_external_labels_path(data_root: Path) -> Path:
    return data_root / "features" / CANONICAL_2021_BUNDLE / EXTERNAL_2021_LABELS_FILE


def resolve_feature_directory(
    feature_name: str,
    split: str,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> Path:
    if feature_root is not None:
        root = Path(feature_root)
        split_dir = root / split
        if split_dir.exists() and split_dir.is_dir():
            return split_dir
        if root.exists() and root.is_dir() and any(root.glob("*.npy")):
            return root
        raise FileNotFoundError(f"Could not resolve feature directory from explicit root: {root}")

    resolved_data_root = resolve_data_root(data_root)
    feature_dir = _expected_feature_directory(resolved_data_root, feature_name, split)
    if not feature_dir.exists() or not feature_dir.is_dir():
        raise FileNotFoundError(
            f"Could not locate feature directory for feature='{feature_name}', split='{split}' at '{feature_dir}'. "
            "Use --data_root as the parent data directory, for example 'data'."
        )
    return feature_dir


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


def resolve_external_labels_path(
    labels_path: str | Path | None = None,
    *,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> Path:
    if labels_path is not None:
        candidate = Path(labels_path)
        if candidate.exists() and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Could not locate ASVspoof2021 labels file: {candidate}")

    _ = feature_root
    resolved_data_root = resolve_data_root(data_root)
    candidate = _expected_external_labels_path(resolved_data_root)
    if candidate.exists() and candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not locate ASVspoof2021 labels file at '{candidate}'. "
        "Provide --eval_2021_labels explicitly or place the file at that canonical path."
    )


def load_external_labels(
    labels_path: str | Path | None = None,
    *,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    resolved_labels_path = resolve_external_labels_path(
        labels_path=labels_path,
        data_root=data_root,
        feature_root=feature_root,
    )
    with resolved_labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("2021 labels file must contain a header row")
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
            raise ValueError("2021 labels file must contain columns like utt_id/filename and label")
        for row in reader:
            utt_id = Path(row[utt_field].strip()).stem
            labels[utt_id] = canonicalize_label(row[label_field])
    return labels


def _records_from_directory(feature_dir: Path, labels: Dict[str, int]) -> List[SampleRecord]:
    samples: List[SampleRecord] = []
    for feature_path in sorted(feature_dir.glob("*.npy")):
        utt_id = feature_path.stem
        if utt_id not in labels:
            continue
        samples.append(SampleRecord(utt_id=utt_id, path=feature_path, label=labels[utt_id]))
    if not samples:
        raise RuntimeError(f"No labeled .npy samples found in {feature_dir}")
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

    labels = load_protocol_labels(split=split, data_root=data_root, protocol_root=protocol_root)
    feature_dir = resolve_feature_directory(
        feature_name=feature_name,
        split=split,
        data_root=data_root,
        feature_root=feature_root,
    )
    return _records_from_directory(feature_dir=feature_dir, labels=labels)


def build_2021_samples(
    feature_name: str,
    labels_path: str | Path | None = None,
    data_root: str | Path = "data",
    feature_root: str | Path | None = None,
) -> List[SampleRecord]:
    labels = load_external_labels(labels_path, data_root=data_root, feature_root=feature_root)
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
        self.feature_name = feature_name

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
