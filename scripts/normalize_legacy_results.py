from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.runtime import experiment_dir_name


TRAIN_LOG_FIELDS = [
    "profile",
    "model",
    "feature",
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_eer",
    "lr",
]
RESULTS_2019_FIELDS = [
    "profile",
    "model",
    "feature",
    "accuracy",
    "eer",
    "num_samples",
    "checkpoint",
]
RESULTS_2021_FIELDS = [
    "profile",
    "model",
    "feature",
    "eer_2021",
    "accuracy",
    "num_samples",
    "eer_2019",
    "generalization_gap",
    "checkpoint",
]
EER_COMPARISON_FIELDS = [
    "profile",
    "model",
    "feature",
    "eer_2019",
    "eer_2021",
    "generalization_gap",
    "checkpoint",
]
AGGREGATE_MAPPING_SOURCES = (
    "results_2019.csv",
    "results_2021.csv",
    "train_log.csv",
    "eer_comparison.csv",
)


@dataclass
class NormalizationSummary:
    created_files: int = 0
    skipped_files: int = 0
    warnings: list[str] = field(default_factory=list)

    def created(self, path: Path) -> None:
        self.created_files += 1
        print(f"CREATED {path}")

    def skipped(self, path: Path, reason: str) -> None:
        self.skipped_files += 1
        print(f"SKIPPED {path}: {reason}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARNING {message}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_binary_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"0", "0.0", "bonafide", "bona_fide", "genuine"}:
        return 0
    if normalized in {"1", "1.0", "spoof", "fake"}:
        return 1

    float_value = float(normalized)
    int_value = int(float_value)
    if float_value != int_value or int_value not in (0, 1):
        raise ValueError(f"Expected a binary label, received {value!r}.")
    return int_value


def compute_prediction_metrics(path: Path) -> dict[str, float | int]:
    rows = read_csv_rows(path)
    num_samples = len(rows)
    if num_samples == 0:
        raise ValueError(f"Prediction file {path} is empty.")

    correct_predictions = 0
    for row in rows:
        label = parse_binary_label(row["label"])
        predicted = parse_binary_label(row["pred_label"])
        correct_predictions += int(label == predicted)

    return {
        "accuracy": correct_predictions / num_samples,
        "num_samples": num_samples,
    }


def prediction_row_count(path: Path) -> int:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Prediction file {path} is empty.")
    return len(rows)


def profile_mapping(results_dir: Path, summary: NormalizationSummary) -> dict[str, tuple[str, str]]:
    candidates: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for filename in AGGREGATE_MAPPING_SOURCES:
        path = results_dir / filename
        if not path.exists():
            continue

        for row in read_csv_rows(path):
            profile = row.get("profile", "").strip()
            model = row.get("model", "").strip()
            feature = row.get("feature", "").strip()
            if profile and model and feature:
                candidates[profile].add((model, feature))

    if not candidates:
        summary.warn(f"No profile-to-experiment mapping could be inferred from {results_dir}.")
        return {}

    resolved: dict[str, tuple[str, str]] = {}
    for profile, combinations in sorted(candidates.items()):
        if len(combinations) == 1:
            resolved[profile] = next(iter(combinations))
            continue

        combinations_text = ", ".join(
            f"{model}/{feature}" for model, feature in sorted(combinations)
        )
        summary.warn(
            f"Ambiguous mapping for profile {profile!r}: {combinations_text}. Skipping this profile."
        )
    return resolved


def group_rows_by_profile(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not path.exists():
        return grouped

    for row in read_csv_rows(path):
        profile = row.get("profile", "").strip()
        if profile:
            grouped[profile].append(row)
    return grouped


def prediction_source_path(results_dir: Path, profile: str, year: int) -> Path:
    return results_dir / f"predictions_{profile}_{year}.csv"


def sorted_train_log_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def sort_key(row: dict[str, str]) -> tuple[int, str]:
        epoch_text = row.get("epoch", "").strip()
        if epoch_text.isdigit():
            return (int(epoch_text), row.get("checkpoint", ""))
        return (sys.maxsize, row.get("checkpoint", ""))

    return sorted(rows, key=sort_key)


def normalize_results_2019_rows(
    rows: list[dict[str, str]],
    num_samples: int,
) -> list[dict[str, str]]:
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "profile": row.get("profile", ""),
                "model": row.get("model", ""),
                "feature": row.get("feature", ""),
                "accuracy": row.get("accuracy", ""),
                "eer": row.get("eer", ""),
                "num_samples": str(num_samples),
                "checkpoint": row.get("checkpoint", ""),
            }
        )
    return sorted(normalized_rows, key=lambda row: row["checkpoint"])


def normalize_results_2021_rows(
    rows: list[dict[str, str]],
    accuracy: float,
    num_samples: int,
) -> list[dict[str, str]]:
    normalized_rows = []
    for row in rows:
        generalization_gap = row.get("generalization_gap", "").strip()
        if not generalization_gap and row.get("eer_2021") and row.get("eer_2019"):
            generalization_gap = str(float(row["eer_2021"]) - float(row["eer_2019"]))

        normalized_rows.append(
            {
                "profile": row.get("profile", ""),
                "model": row.get("model", ""),
                "feature": row.get("feature", ""),
                "eer_2021": row.get("eer_2021", ""),
                "accuracy": str(accuracy),
                "num_samples": str(num_samples),
                "eer_2019": row.get("eer_2019", ""),
                "generalization_gap": generalization_gap,
                "checkpoint": row.get("checkpoint", ""),
            }
        )
    return sorted(normalized_rows, key=lambda row: row["checkpoint"])


def eer_comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "profile": row["profile"],
                "model": row["model"],
                "feature": row["feature"],
                "eer_2019": row["eer_2019"],
                "eer_2021": row["eer_2021"],
                "generalization_gap": row["generalization_gap"],
                "checkpoint": row["checkpoint"],
            }
        )
    return sorted(normalized_rows, key=lambda row: row["checkpoint"])


def copy_file_if_missing(source: Path, destination: Path, summary: NormalizationSummary) -> None:
    if destination.exists():
        summary.skipped(destination, "destination already exists")
        return
    if not source.exists():
        summary.warn(f"Missing source file {source}.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    summary.created(destination)


def write_file_if_missing(
    destination: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    summary: NormalizationSummary,
) -> None:
    if destination.exists():
        summary.skipped(destination, "destination already exists")
        return
    if not rows:
        summary.warn(f"No rows available for {destination}.")
        return

    write_csv_rows(destination, fieldnames=fieldnames, rows=rows)
    summary.created(destination)


def normalize_legacy_results(results_dir: Path) -> NormalizationSummary:
    summary = NormalizationSummary()
    mapping = profile_mapping(results_dir, summary)

    train_log_by_profile = group_rows_by_profile(results_dir / "train_log.csv")
    results_2019_by_profile = group_rows_by_profile(results_dir / "results_2019.csv")
    results_2021_by_profile = group_rows_by_profile(results_dir / "results_2021.csv")

    for profile, (model, feature) in sorted(mapping.items()):
        experiment_dir = results_dir / experiment_dir_name(model_name=model, feature_name=feature)
        experiment_dir.mkdir(parents=True, exist_ok=True)

        train_rows = sorted_train_log_rows(train_log_by_profile.get(profile, []))
        write_file_if_missing(
            destination=experiment_dir / "train_log.csv",
            fieldnames=TRAIN_LOG_FIELDS,
            rows=[{field: row.get(field, "") for field in TRAIN_LOG_FIELDS} for row in train_rows],
            summary=summary,
        )

        prediction_2019_source = prediction_source_path(results_dir, profile, 2019)
        prediction_2021_source = prediction_source_path(results_dir, profile, 2021)

        copy_file_if_missing(
            source=prediction_2019_source,
            destination=experiment_dir / "predictions_2019.csv",
            summary=summary,
        )
        copy_file_if_missing(
            source=prediction_2021_source,
            destination=experiment_dir / "predictions_2021.csv",
            summary=summary,
        )

        if not prediction_2019_source.exists():
            summary.warn(
                f"Cannot normalize results_2019 for profile {profile!r} because {prediction_2019_source} is missing."
            )
            normalized_results_2019 = []
        else:
            try:
                normalized_results_2019 = normalize_results_2019_rows(
                    rows=results_2019_by_profile.get(profile, []),
                    num_samples=prediction_row_count(prediction_2019_source),
                )
            except ValueError as error:
                summary.warn(f"Cannot normalize results_2019 for profile {profile!r}: {error}")
                normalized_results_2019 = []

        write_file_if_missing(
            destination=experiment_dir / "results_2019.csv",
            fieldnames=RESULTS_2019_FIELDS,
            rows=normalized_results_2019,
            summary=summary,
        )

        if not prediction_2021_source.exists():
            summary.warn(
                f"Cannot normalize results_2021 for profile {profile!r} because {prediction_2021_source} is missing."
            )
            normalized_results_2021 = []
        else:
            try:
                prediction_2021_metrics = compute_prediction_metrics(prediction_2021_source)
                normalized_results_2021 = normalize_results_2021_rows(
                    rows=results_2021_by_profile.get(profile, []),
                    accuracy=float(prediction_2021_metrics["accuracy"]),
                    num_samples=int(prediction_2021_metrics["num_samples"]),
                )
            except ValueError as error:
                summary.warn(f"Cannot normalize results_2021 for profile {profile!r}: {error}")
                normalized_results_2021 = []

        write_file_if_missing(
            destination=experiment_dir / "results_2021.csv",
            fieldnames=RESULTS_2021_FIELDS,
            rows=normalized_results_2021,
            summary=summary,
        )
        write_file_if_missing(
            destination=experiment_dir / "eer_comparison.csv",
            fieldnames=EER_COMPARISON_FIELDS,
            rows=eer_comparison_rows(normalized_results_2021),
            summary=summary,
        )

    print(
        "SUMMARY "
        f"created={summary.created_files} "
        f"skipped={summary.skipped_files} "
        f"warnings={len(summary.warnings)}"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize legacy CSV files in outputs/results to the experiment directory layout."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs") / "results",
        help="Path to the legacy results directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"ERROR Results directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    normalize_legacy_results(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
