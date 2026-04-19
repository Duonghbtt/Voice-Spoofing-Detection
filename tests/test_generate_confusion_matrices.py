from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_confusion_matrices import generate_confusion_matrices, load_confusion_data


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class GenerateConfusionMatricesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.results_root = self.root / "outputs" / "results"
        self.figures_root = self.root / "outputs" / "figures"
        self.results_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def experiment_dir(self, combo: str) -> Path:
        path = self.results_root / combo
        path.mkdir(parents=True, exist_ok=True)
        return path

    def output_png(self, combo: str) -> Path:
        return self.figures_root / combo / "confusion_matrix.png"

    def run_generator(self, *, force: bool = False) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            generate_confusion_matrices(self.results_root, self.figures_root, force=force)
        return buffer.getvalue()

    def test_generates_confusion_matrix_and_prefers_predictions_2019(self) -> None:
        combo = "cnn_mfcc"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "predictions_2019.csv",
            fieldnames=["label", "pred_label"],
            rows=[
                {"label": "0", "pred_label": "0"},
                {"label": "1", "pred_label": "1"},
            ],
        )
        write_csv(
            experiment_dir / "predictions_2021.csv",
            fieldnames=["label", "pred_label"],
            rows=[
                {"label": "0", "pred_label": "1"},
                {"label": "0", "pred_label": "1"},
                {"label": "1", "pred_label": "0"},
            ],
        )

        cleaned, source_path, reason = load_confusion_data(experiment_dir)

        self.assertIsNone(reason)
        self.assertIsNotNone(cleaned)
        self.assertEqual(source_path, experiment_dir / "predictions_2019.csv")
        self.assertEqual(cleaned.to_dict("records"), [{"y_true": 0, "y_pred": 0}, {"y_true": 1, "y_pred": 1}])

        output = self.run_generator()

        png_path = self.output_png(combo)
        self.assertIn(f"[GEN CM] {combo} -> confusion_matrix.png", output)
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 0)

    def test_skips_existing_png_without_force(self) -> None:
        combo = "lcnn_lfcc"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "predictions_2019.csv",
            fieldnames=["label", "pred_label"],
            rows=[{"label": "0", "pred_label": "0"}],
        )
        png_path = self.output_png(combo)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(b"sentinel")

        output = self.run_generator()

        self.assertIn(f"[SKIP CM] {combo} exists", output)
        self.assertEqual(png_path.read_bytes(), b"sentinel")

    def test_regenerates_existing_png_with_force(self) -> None:
        combo = "cnn_lfcc"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "predictions_2019.csv",
            fieldnames=["label", "pred_label"],
            rows=[
                {"label": "0", "pred_label": "0"},
                {"label": "1", "pred_label": "0"},
            ],
        )
        png_path = self.output_png(combo)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(b"sentinel")

        output = self.run_generator(force=True)

        self.assertIn(f"[GEN CM] {combo} -> confusion_matrix.png", output)
        self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG"))

    def test_falls_back_to_predictions_2021_when_2019_is_invalid(self) -> None:
        combo = "resnet18_mfcc"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "predictions_2019.csv",
            fieldnames=["score_spoof"],
            rows=[{"score_spoof": "0.1"}],
        )
        write_csv(
            experiment_dir / "predictions_2021.csv",
            fieldnames=["label", "pred_label"],
            rows=[
                {"label": "bonafide", "pred_label": "spoof"},
                {"label": "spoof", "pred_label": "spoof"},
            ],
        )

        cleaned, source_path, reason = load_confusion_data(experiment_dir)

        self.assertIsNone(reason)
        self.assertEqual(source_path, experiment_dir / "predictions_2021.csv")
        self.assertEqual(cleaned.to_dict("records"), [{"y_true": 0, "y_pred": 1}, {"y_true": 1, "y_pred": 1}])

    def test_falls_back_to_results_2019_with_row_level_labels(self) -> None:
        combo = "resnet18_lfcc"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "results_2019.csv",
            fieldnames=["y_true", "y_pred"],
            rows=[
                {"y_true": "0", "y_pred": "0"},
                {"y_true": "1", "y_pred": "0"},
            ],
        )

        cleaned, source_path, reason = load_confusion_data(experiment_dir)

        self.assertIsNone(reason)
        self.assertEqual(source_path, experiment_dir / "results_2019.csv")
        self.assertEqual(cleaned.to_dict("records"), [{"y_true": 0, "y_pred": 0}, {"y_true": 1, "y_pred": 0}])

    def test_falls_back_to_results_2021_with_row_level_labels(self) -> None:
        combo = "resnet18_spectrogram"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "results_2019.csv",
            fieldnames=["accuracy", "eer"],
            rows=[{"accuracy": "0.91", "eer": "0.12"}],
        )
        write_csv(
            experiment_dir / "results_2021.csv",
            fieldnames=["label", "prediction"],
            rows=[
                {"label": "genuine", "prediction": "fake"},
                {"label": "fake", "prediction": "fake"},
            ],
        )

        cleaned, source_path, reason = load_confusion_data(experiment_dir)

        self.assertIsNone(reason)
        self.assertEqual(source_path, experiment_dir / "results_2021.csv")
        self.assertEqual(cleaned.to_dict("records"), [{"y_true": 0, "y_pred": 1}, {"y_true": 1, "y_pred": 1}])

    def test_skips_missing_predictions_when_only_summary_results_exist(self) -> None:
        combo = "cnn_spectrogram"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "results_2019.csv",
            fieldnames=["profile", "model", "feature", "accuracy", "eer", "num_samples", "checkpoint"],
            rows=[
                {
                    "profile": "custom",
                    "model": "cnn",
                    "feature": "spectrogram",
                    "accuracy": "0.9",
                    "eer": "0.1",
                    "num_samples": "2",
                    "checkpoint": "outputs/checkpoints/example.ckpt",
                }
            ],
        )

        output = self.run_generator()

        self.assertIn(f"[SKIP CM] {combo} missing predictions", output)
        self.assertFalse(self.output_png(combo).exists())

    def test_drops_nan_and_invalid_rows_before_plotting(self) -> None:
        combo = "cleaning_case"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "predictions_2019.csv",
            fieldnames=["label", "pred_label"],
            rows=[
                {"label": "0", "pred_label": "1"},
                {"label": "", "pred_label": "1"},
                {"label": "spoof", "pred_label": "not-a-label"},
                {"label": "genuine", "pred_label": "0"},
            ],
        )

        cleaned, source_path, reason = load_confusion_data(experiment_dir)

        self.assertIsNone(reason)
        self.assertEqual(source_path, experiment_dir / "predictions_2019.csv")
        self.assertEqual(cleaned.to_dict("records"), [{"y_true": 0, "y_pred": 1}, {"y_true": 0, "y_pred": 0}])

        output = self.run_generator()

        self.assertIn(f"[GEN CM] {combo} -> confusion_matrix.png", output)
        self.assertTrue(self.output_png(combo).exists())

    def test_skips_when_no_valid_rows_remain(self) -> None:
        combo = "empty_after_cleaning"
        experiment_dir = self.experiment_dir(combo)
        write_csv(
            experiment_dir / "predictions_2019.csv",
            fieldnames=["label", "pred_label"],
            rows=[
                {"label": "", "pred_label": "1"},
                {"label": "spoof", "pred_label": "not-a-label"},
                {"label": "unknown", "pred_label": "0"},
            ],
        )

        cleaned, source_path, reason = load_confusion_data(experiment_dir)

        self.assertIsNone(cleaned)
        self.assertIsNone(source_path)
        self.assertEqual(reason, "no valid rows")

        output = self.run_generator()

        self.assertIn(f"[SKIP CM] {combo} no valid rows", output)
        self.assertFalse(self.output_png(combo).exists())

    def test_ignores_non_directory_entries_under_results_root(self) -> None:
        (self.results_root / "readme.txt").write_text("ignore me", encoding="utf-8")

        output = self.run_generator()

        self.assertEqual(output, "")
        self.assertFalse(any(self.figures_root.rglob("confusion_matrix.png")))


if __name__ == "__main__":
    unittest.main()
