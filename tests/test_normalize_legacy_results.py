from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


class NormalizeLegacyResultsScriptTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.results_dir = self.root / "outputs" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_legacy_fixture(self) -> None:
        write_csv(
            self.results_dir / "train_log.csv",
            fieldnames=[
                "profile",
                "model",
                "feature",
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_eer",
                "lr",
            ],
            rows=[
                {
                    "profile": "baseline",
                    "model": "cnn",
                    "feature": "mfcc",
                    "epoch": "2",
                    "train_loss": "0.20",
                    "val_loss": "0.25",
                    "val_accuracy": "0.85",
                    "val_eer": "0.12",
                    "lr": "0.001",
                },
                {
                    "profile": "baseline",
                    "model": "cnn",
                    "feature": "mfcc",
                    "epoch": "1",
                    "train_loss": "0.30",
                    "val_loss": "0.35",
                    "val_accuracy": "0.80",
                    "val_eer": "0.16",
                    "lr": "0.001",
                },
                {
                    "profile": "optimized",
                    "model": "lcnn",
                    "feature": "lfcc",
                    "epoch": "1",
                    "train_loss": "0.15",
                    "val_loss": "0.22",
                    "val_accuracy": "0.88",
                    "val_eer": "0.09",
                    "lr": "0.001",
                },
            ],
        )
        write_csv(
            self.results_dir / "results_2019.csv",
            fieldnames=["profile", "model", "feature", "accuracy", "eer", "checkpoint"],
            rows=[
                {
                    "profile": "baseline",
                    "model": "cnn",
                    "feature": "mfcc",
                    "accuracy": "0.91",
                    "eer": "0.10",
                    "checkpoint": "outputs/checkpoints/baseline_cnn_mfcc.pth",
                },
                {
                    "profile": "optimized",
                    "model": "lcnn",
                    "feature": "lfcc",
                    "accuracy": "0.82",
                    "eer": "0.14",
                    "checkpoint": "outputs/checkpoints/optimized_lcnn_lfcc/best.ckpt",
                },
            ],
        )
        write_csv(
            self.results_dir / "results_2021.csv",
            fieldnames=[
                "profile",
                "model",
                "feature",
                "eer_2021",
                "eer_2019",
                "generalization_gap",
                "checkpoint",
            ],
            rows=[
                {
                    "profile": "baseline",
                    "model": "cnn",
                    "feature": "mfcc",
                    "eer_2021": "0.70",
                    "eer_2019": "0.10",
                    "generalization_gap": "0.60",
                    "checkpoint": "outputs/checkpoints/baseline_cnn_mfcc.pth",
                },
                {
                    "profile": "optimized",
                    "model": "lcnn",
                    "feature": "lfcc",
                    "eer_2021": "0.45",
                    "eer_2019": "0.14",
                    "generalization_gap": "0.31",
                    "checkpoint": "outputs/checkpoints/optimized_lcnn_lfcc/best.ckpt",
                },
            ],
        )
        write_csv(
            self.results_dir / "eer_comparison.csv",
            fieldnames=[
                "profile",
                "model",
                "feature",
                "eer_2019",
                "eer_2021",
                "generalization_gap",
            ],
            rows=[
                {
                    "profile": "baseline",
                    "model": "cnn",
                    "feature": "mfcc",
                    "eer_2019": "0.10",
                    "eer_2021": "0.70",
                    "generalization_gap": "0.60",
                },
                {
                    "profile": "optimized",
                    "model": "lcnn",
                    "feature": "lfcc",
                    "eer_2019": "0.14",
                    "eer_2021": "0.45",
                    "generalization_gap": "0.31",
                },
            ],
        )
        write_csv(
            self.results_dir / "predictions_baseline_2019.csv",
            fieldnames=["utt_id", "label", "score_spoof", "pred_label"],
            rows=[
                {"utt_id": "utt-a", "label": "1", "score_spoof": "0.90", "pred_label": "1"},
                {"utt_id": "utt-b", "label": "0", "score_spoof": "0.10", "pred_label": "0"},
                {"utt_id": "utt-c", "label": "1", "score_spoof": "0.85", "pred_label": "1"},
            ],
        )
        write_csv(
            self.results_dir / "predictions_baseline_2021.csv",
            fieldnames=["utt_id", "label", "score_spoof", "pred_label"],
            rows=[
                {"utt_id": "utt-d", "label": "1", "score_spoof": "0.95", "pred_label": "1"},
                {"utt_id": "utt-e", "label": "0", "score_spoof": "0.15", "pred_label": "0"},
                {"utt_id": "utt-f", "label": "1", "score_spoof": "0.55", "pred_label": "0"},
                {"utt_id": "utt-g", "label": "0", "score_spoof": "0.20", "pred_label": "0"},
            ],
        )
        write_csv(
            self.results_dir / "predictions_optimized_2019.csv",
            fieldnames=["utt_id", "label", "score_spoof", "pred_label"],
            rows=[
                {"utt_id": "utt-h", "label": "1", "score_spoof": "0.90", "pred_label": "1"},
                {"utt_id": "utt-i", "label": "0", "score_spoof": "0.70", "pred_label": "1"},
            ],
        )
        write_csv(
            self.results_dir / "predictions_optimized_2021.csv",
            fieldnames=["utt_id", "label", "score_spoof", "pred_label"],
            rows=[
                {"utt_id": "utt-j", "label": "1", "score_spoof": "0.90", "pred_label": "1"},
                {"utt_id": "utt-k", "label": "0", "score_spoof": "0.25", "pred_label": "0"},
                {"utt_id": "utt-l", "label": "1", "score_spoof": "0.60", "pred_label": "1"},
            ],
        )

    def run_script(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "scripts/normalize_legacy_results.py",
                "--results-dir",
                str(self.results_dir),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_normalize_legacy_results_creates_expected_layout_and_schema(self) -> None:
        self.write_legacy_fixture()

        result = self.run_script()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("SUMMARY created=12 skipped=0 warnings=0", result.stdout)

        baseline_dir = self.results_dir / "cnn_mfcc"
        optimized_dir = self.results_dir / "lcnn_lfcc"
        self.assertTrue((baseline_dir / "train_log.csv").exists())
        self.assertTrue((optimized_dir / "train_log.csv").exists())
        self.assertTrue((baseline_dir / "predictions_2019.csv").exists())
        self.assertTrue((baseline_dir / "predictions_2021.csv").exists())
        self.assertTrue((optimized_dir / "predictions_2019.csv").exists())
        self.assertTrue((optimized_dir / "predictions_2021.csv").exists())

        train_fields, train_rows = read_csv(baseline_dir / "train_log.csv")
        self.assertEqual(
            train_fields,
            ["profile", "model", "feature", "epoch", "train_loss", "val_loss", "val_accuracy", "val_eer", "lr"],
        )
        self.assertEqual([row["epoch"] for row in train_rows], ["1", "2"])

        result_2019_fields, result_2019_rows = read_csv(baseline_dir / "results_2019.csv")
        self.assertEqual(result_2019_fields, RESULTS_2019_FIELDS)
        self.assertEqual(result_2019_rows[0]["num_samples"], "3")

        result_2021_fields, result_2021_rows = read_csv(baseline_dir / "results_2021.csv")
        self.assertEqual(result_2021_fields, RESULTS_2021_FIELDS)
        self.assertEqual(result_2021_rows[0]["num_samples"], "4")
        self.assertAlmostEqual(float(result_2021_rows[0]["accuracy"]), 0.75)

        comparison_fields, comparison_rows = read_csv(baseline_dir / "eer_comparison.csv")
        self.assertEqual(comparison_fields, EER_COMPARISON_FIELDS)
        self.assertEqual(comparison_rows[0]["checkpoint"], "outputs/checkpoints/baseline_cnn_mfcc.pth")

    def test_normalize_legacy_results_skips_existing_destination_files(self) -> None:
        self.write_legacy_fixture()
        existing_path = self.results_dir / "cnn_mfcc" / "results_2019.csv"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("sentinel\n", encoding="utf-8")

        result = self.run_script()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(existing_path.read_text(encoding="utf-8"), "sentinel\n")
        self.assertIn("SKIPPED", result.stdout)
        self.assertIn(str(existing_path), result.stdout)
        self.assertIn("SUMMARY created=11 skipped=1 warnings=0", result.stdout)

    def test_normalize_legacy_results_skips_ambiguous_profile_mapping(self) -> None:
        write_csv(
            self.results_dir / "results_2019.csv",
            fieldnames=["profile", "model", "feature", "accuracy", "eer", "checkpoint"],
            rows=[
                {
                    "profile": "baseline",
                    "model": "cnn",
                    "feature": "mfcc",
                    "accuracy": "0.91",
                    "eer": "0.10",
                    "checkpoint": "outputs/checkpoints/baseline_cnn_mfcc.pth",
                },
            ],
        )
        write_csv(
            self.results_dir / "train_log.csv",
            fieldnames=[
                "profile",
                "model",
                "feature",
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_eer",
                "lr",
            ],
            rows=[
                {
                    "profile": "baseline",
                    "model": "lcnn",
                    "feature": "lfcc",
                    "epoch": "1",
                    "train_loss": "0.10",
                    "val_loss": "0.12",
                    "val_accuracy": "0.90",
                    "val_eer": "0.08",
                    "lr": "0.001",
                },
            ],
        )

        result = self.run_script()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Ambiguous mapping for profile 'baseline'", result.stdout)
        self.assertIn("SUMMARY created=0 skipped=0 warnings=1", result.stdout)
        self.assertFalse((self.results_dir / "cnn_mfcc").exists())
        self.assertFalse((self.results_dir / "lcnn_lfcc").exists())


if __name__ == "__main__":
    unittest.main()
