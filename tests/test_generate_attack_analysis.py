from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_attack_analysis import (
    build_attack_analysis,
    detect_prediction_schema,
    parse_trial_metadata,
    write_outputs,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


class GenerateAttackAnalysisTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.reports_dir = self.root / "outputs" / "reports"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_metadata(self, rows: list[str]) -> Path:
        path = self.root / "data" / "raw" / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def suppress_stdout(self):
        return redirect_stdout(io.StringIO())

    def test_parse_trial_metadata_filters_eval_and_maps_types(self) -> None:
        metadata_path = self.write_metadata(
            [
                "LA_0001 utt_eval_tts pcm tx A07 spoof notrim eval",
                "LA_0001 utt_eval_vc pcm tx A10 spoof notrim eval",
                "LA_0001 utt_eval_bona pcm tx bonafide bonafide notrim eval",
                "LA_0001 utt_progress pcm tx A08 spoof notrim progress",
            ]
        )

        frame = parse_trial_metadata(metadata_path)

        self.assertEqual(frame.columns.tolist(), ["utt_id", "label", "attack_id", "type"])
        self.assertEqual(frame["utt_id"].tolist(), ["utt_eval_tts", "utt_eval_vc", "utt_eval_bona"])
        self.assertEqual(frame["attack_id"].tolist(), ["A07", "A10", ""])
        self.assertEqual(frame["type"].tolist(), ["TTS", "VC", ""])

    def test_detect_prediction_schema_finds_expected_columns_and_continuous_score(self) -> None:
        predictions_path = self.root / "predictions_2021.csv"
        write_csv(
            predictions_path,
            fieldnames=["utt_id", "label", "score_spoof", "pred_label"],
            rows=[
                {"utt_id": "utt-a", "label": 1, "score_spoof": 0.95, "pred_label": 1},
                {"utt_id": "utt-b", "label": 0, "score_spoof": 0.05, "pred_label": 0},
                {"utt_id": "utt-c", "label": 1, "score_spoof": 0.85, "pred_label": 1},
            ],
        )

        with self.suppress_stdout():
            frame, schema = detect_prediction_schema(predictions_path)

        self.assertEqual(frame.columns.tolist(), ["utt_id", "label", "score_spoof", "pred_label"])
        self.assertEqual(
            schema,
            {
                "id_column": "utt_id",
                "label_column": "label",
                "pred_column": "pred_label",
                "score_column": "score_spoof",
            },
        )

    def test_build_attack_analysis_writes_full_report_and_summary(self) -> None:
        metadata_path = self.write_metadata(
            [
                "LA_0001 bona_1 pcm tx bonafide bonafide notrim eval",
                "LA_0001 bona_2 pcm tx bonafide bonafide notrim eval",
                "LA_0001 bona_3 pcm tx bonafide bonafide notrim eval",
                "LA_0001 bona_4 pcm tx bonafide bonafide notrim eval",
                "LA_0001 a07_1 pcm tx A07 spoof notrim eval",
                "LA_0001 a07_2 pcm tx A07 spoof notrim eval",
                "LA_0001 a07_3 pcm tx A07 spoof notrim eval",
                "LA_0001 a07_4 pcm tx A07 spoof notrim eval",
                "LA_0001 a10_1 pcm tx A10 spoof notrim eval",
                "LA_0001 a10_2 pcm tx A10 spoof notrim eval",
                "LA_0001 a10_3 pcm tx A10 spoof notrim eval",
                "LA_0001 a10_4 pcm tx A10 spoof notrim eval",
            ]
        )
        predictions_path = self.root / "predictions_2021.csv"
        write_csv(
            predictions_path,
            fieldnames=["utt_id", "label", "score_spoof", "pred_label"],
            rows=[
                {"utt_id": "bona_1", "label": 1, "score_spoof": 0.95, "pred_label": 1},
                {"utt_id": "bona_2", "label": 1, "score_spoof": 0.90, "pred_label": 1},
                {"utt_id": "bona_3", "label": 1, "score_spoof": 0.85, "pred_label": 1},
                {"utt_id": "bona_4", "label": 1, "score_spoof": 0.80, "pred_label": 1},
                {"utt_id": "a07_1", "label": 0, "score_spoof": 0.10, "pred_label": 0},
                {"utt_id": "a07_2", "label": 0, "score_spoof": 0.20, "pred_label": 0},
                {"utt_id": "a07_3", "label": 0, "score_spoof": 0.05, "pred_label": 0},
                {"utt_id": "a07_4", "label": 0, "score_spoof": 0.15, "pred_label": 0},
                {"utt_id": "a10_1", "label": 0, "score_spoof": 0.95, "pred_label": 1},
                {"utt_id": "a10_2", "label": 0, "score_spoof": 0.85, "pred_label": 1},
                {"utt_id": "a10_3", "label": 0, "score_spoof": 0.20, "pred_label": 0},
                {"utt_id": "a10_4", "label": 0, "score_spoof": 0.10, "pred_label": 0},
            ],
        )

        metadata_df = parse_trial_metadata(metadata_path)
        with self.suppress_stdout():
            predictions_df, schema = detect_prediction_schema(predictions_path)
        analysis_frame, result = build_attack_analysis(metadata_df, predictions_df, "demo_combo", schema)
        write_outputs(analysis_frame, result["summary"], self.reports_dir)

        self.assertEqual(len(analysis_frame), 19)
        self.assertEqual(analysis_frame.columns.tolist(), ["attack_id", "type", "system", "eer", "level", "note", "eer_value"])

        row_a01 = analysis_frame.loc[analysis_frame["attack_id"] == "A01"].iloc[0]
        row_a07 = analysis_frame.loc[analysis_frame["attack_id"] == "A07"].iloc[0]
        row_a10 = analysis_frame.loc[analysis_frame["attack_id"] == "A10"].iloc[0]

        self.assertEqual(row_a01["eer"], "N/A")
        self.assertEqual(row_a01["level"], "N/A")
        self.assertEqual(row_a01["note"], "missing from eval metadata/predictions")
        self.assertEqual(row_a07["eer"], "0.00")
        self.assertEqual(row_a07["level"], "Dễ")
        self.assertEqual(row_a07["note"], "")
        self.assertEqual(row_a10["level"], "Khó")
        self.assertEqual(row_a10["note"], "")

        csv_fields, csv_rows = read_csv(self.reports_dir / "attack_analysis.csv")
        self.assertEqual(csv_fields, ["attack_id", "type", "system", "eer", "level", "note"])
        self.assertEqual(len(csv_rows), 19)
        self.assertEqual(csv_rows[0]["attack_id"], "A01")
        self.assertEqual(csv_rows[0]["eer"], "N/A")

        summary_text = (self.reports_dir / "attack_analysis_summary.txt").read_text(encoding="utf-8")
        self.assertIn("Hardest attack: A10", summary_text)
        self.assertIn("Easiest attack: A07", summary_text)
        self.assertIn("Average TTS EER: 0.00%", summary_text)
        self.assertIn("Conclusion: VC harder to detect", summary_text)

    def test_build_attack_analysis_raises_clear_error_on_key_mismatch(self) -> None:
        metadata_path = self.write_metadata(["LA_0001 meta_only pcm tx A07 spoof notrim eval"])
        predictions_path = self.root / "predictions_2021.csv"
        write_csv(
            predictions_path,
            fieldnames=["utt_id", "label", "score_spoof", "pred_label"],
            rows=[{"utt_id": "pred_only", "label": 1, "score_spoof": 0.95, "pred_label": 1}],
        )

        metadata_df = parse_trial_metadata(metadata_path)
        with self.suppress_stdout():
            predictions_df, schema = detect_prediction_schema(predictions_path)

        with self.assertRaises(ValueError) as context:
            build_attack_analysis(metadata_df, predictions_df, "demo_combo", schema)

        message = str(context.exception)
        self.assertIn("Merge failed", message)
        self.assertIn("prediction key examples", message)
        self.assertIn("metadata key examples", message)

    def test_missing_continuous_score_marks_attack_rows_as_na(self) -> None:
        metadata_path = self.write_metadata(
            [
                "LA_0001 bona_1 pcm tx bonafide bonafide notrim eval",
                "LA_0001 bona_2 pcm tx bonafide bonafide notrim eval",
                "LA_0001 a07_1 pcm tx A07 spoof notrim eval",
                "LA_0001 a07_2 pcm tx A07 spoof notrim eval",
            ]
        )
        predictions_path = self.root / "predictions_2021.csv"
        write_csv(
            predictions_path,
            fieldnames=["utt_id", "label", "pred_label"],
            rows=[
                {"utt_id": "bona_1", "label": 1, "pred_label": 1},
                {"utt_id": "bona_2", "label": 1, "pred_label": 1},
                {"utt_id": "a07_1", "label": 0, "pred_label": 0},
                {"utt_id": "a07_2", "label": 0, "pred_label": 0},
            ],
        )

        metadata_df = parse_trial_metadata(metadata_path)
        with self.suppress_stdout():
            predictions_df, schema = detect_prediction_schema(predictions_path)

        self.assertIsNone(schema["score_column"])

        analysis_frame, result = build_attack_analysis(metadata_df, predictions_df, "demo_combo", schema)
        write_outputs(analysis_frame, result["summary"], self.reports_dir)

        row_a07 = analysis_frame.loc[analysis_frame["attack_id"] == "A07"].iloc[0]
        self.assertEqual(row_a07["eer"], "N/A")
        self.assertEqual(row_a07["level"], "N/A")
        self.assertEqual(row_a07["note"], "missing continuous score")

        summary_text = (self.reports_dir / "attack_analysis_summary.txt").read_text(encoding="utf-8")
        self.assertIn("Hardest attack: N/A", summary_text)
        self.assertIn("Average TTS EER: N/A", summary_text)
        self.assertIn("Conclusion: N/A", summary_text)

    def test_missing_input_files_raise_clear_errors(self) -> None:
        missing_metadata = self.root / "missing_trial_metadata.txt"
        missing_predictions = self.root / "missing_predictions_2021.csv"

        with self.assertRaises(FileNotFoundError) as metadata_error:
            parse_trial_metadata(missing_metadata)
        with self.assertRaises(FileNotFoundError) as prediction_error:
            detect_prediction_schema(missing_predictions)

        self.assertIn(str(missing_metadata), str(metadata_error.exception))
        self.assertIn(str(missing_predictions), str(prediction_error.exception))


if __name__ == "__main__":
    unittest.main()
