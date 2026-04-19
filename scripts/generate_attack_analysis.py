from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream is not None and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except ValueError:
            pass

ATTACK_IDS = [f"A{index:02d}" for index in range(1, 20)]
CSV_COLUMNS = ["attack_id", "type", "system", "eer", "level", "note"]
METADATA_COLUMNS = ["speaker_id", "utt_id", "codec", "transmission", "attack_id", "label", "trim", "subset"]
ID_COLUMN_CANDIDATES = ("utt_id", "file_id", "filename", "file", "utterance_id")
LABEL_COLUMN_CANDIDATES = ("label", "y_true", "target")
PREDICTION_COLUMN_CANDIDATES = ("pred_label", "y_pred", "prediction")
SCORE_COLUMN_CANDIDATES = ("score_spoof", "score", "probability", "prob", "logit", "spoof_score", "spoof_prob")
METADATA_LABEL_MAP = {"bonafide": 0, "spoof": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ASVspoof 2021 LA attack-wise EER report for one combo.")
    parser.add_argument("--combo", required=True, help="Experiment combo name, for example 'lcnn_mfcc'.")
    return parser.parse_args()


def _normalize_utt_id(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    return Path(text).stem


def _attack_type(attack_id: object) -> str:
    text = "" if attack_id is None else str(attack_id).strip().upper()
    if not text or not text.startswith("A") or len(text) != 3 or not text[1:].isdigit():
        return ""
    attack_number = int(text[1:])
    if 1 <= attack_number <= 9:
        return "TTS"
    if 10 <= attack_number <= 19:
        return "VC"
    return ""


def _normalized_column_lookup(columns: pd.Index) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized and normalized not in lookup:
            lookup[normalized] = str(column)
    return lookup


def _find_candidate_column(lookup: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _detect_continuous_score_column(frame: pd.DataFrame, lookup: dict[str, str]) -> str | None:
    for candidate in SCORE_COLUMN_CANDIDATES:
        column = lookup.get(candidate)
        if column is None:
            continue
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        if numeric_values.isna().any():
            continue
        score_array = numeric_values.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(score_array).all():
            continue
        if np.unique(score_array).size <= 2:
            continue
        return column
    return None


def parse_trial_metadata(metadata_path: str | Path) -> pd.DataFrame:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not locate trial metadata file: {path}")

    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=METADATA_COLUMNS, dtype=str)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Trial metadata file is empty: {path}") from error

    frame["utt_id"] = frame["utt_id"].map(_normalize_utt_id)
    frame["label"] = frame["label"].fillna("").astype(str).str.strip().str.lower()
    frame["attack_id"] = frame["attack_id"].fillna("").astype(str).str.strip().str.upper()
    frame["subset"] = frame["subset"].fillna("").astype(str).str.strip().str.lower()

    eval_frame = frame.loc[frame["subset"] == "eval", ["utt_id", "label", "attack_id"]].copy()
    eval_frame.loc[eval_frame["label"] == "bonafide", "attack_id"] = ""
    eval_frame.loc[~eval_frame["attack_id"].isin(ATTACK_IDS), "attack_id"] = ""
    eval_frame["type"] = eval_frame["attack_id"].map(_attack_type)
    eval_frame = eval_frame.drop_duplicates(subset="utt_id", keep="first").reset_index(drop=True)
    return eval_frame


def detect_prediction_schema(predictions_path: str | Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not locate predictions file: {path}")

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Prediction file is empty: {path}") from error

    print(f"Prediction columns: {frame.columns.tolist()}")

    lookup = _normalized_column_lookup(frame.columns)
    id_column = _find_candidate_column(lookup, ID_COLUMN_CANDIDATES)
    if id_column is None:
        raise ValueError(
            "Could not detect utt_id / file_id column in prediction file. "
            f"Available columns: {frame.columns.tolist()}"
        )

    schema = {
        "id_column": id_column,
        "label_column": _find_candidate_column(lookup, LABEL_COLUMN_CANDIDATES),
        "pred_column": _find_candidate_column(lookup, PREDICTION_COLUMN_CANDIDATES),
        "score_column": _detect_continuous_score_column(frame, lookup),
    }

    print(f"Detected id_column: {schema['id_column']}")
    print(f"Detected label_column: {schema['label_column']}")
    print(f"Detected pred_column: {schema['pred_column']}")
    print(f"Detected score_column: {schema['score_column']}")
    return frame, schema


def compute_eer(labels: pd.Series | np.ndarray | list[int], scores: pd.Series | np.ndarray | list[float]) -> float:
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)

    if label_array.size != score_array.size:
        raise ValueError(f"labels and scores must have the same length, got {label_array.size} and {score_array.size}")
    if label_array.size == 0:
        return float("nan")
    if not np.isfinite(score_array).all():
        raise ValueError("scores contains NaN or Inf values")
    if np.unique(label_array).size < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(label_array, score_array, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def _format_eer_percent(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.2f}"


def _format_optional_summary_percent(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.2f}%"


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


def _build_summary(analysis_frame: pd.DataFrame) -> dict[str, Any]:
    valid_rows = analysis_frame.loc[analysis_frame["eer_value"].notna()].copy()
    tts_rows = valid_rows.loc[valid_rows["type"] == "TTS"]
    vc_rows = valid_rows.loc[valid_rows["type"] == "VC"]

    hardest_attack = None
    easiest_attack = None
    hardest_eer = None
    easiest_eer = None
    if not valid_rows.empty:
        hardest_row = valid_rows.sort_values(by="eer_value", ascending=False, kind="stable").iloc[0]
        easiest_row = valid_rows.sort_values(by="eer_value", ascending=True, kind="stable").iloc[0]
        hardest_attack = str(hardest_row["attack_id"])
        easiest_attack = str(easiest_row["attack_id"])
        hardest_eer = float(hardest_row["eer_value"])
        easiest_eer = float(easiest_row["eer_value"])

    average_tts = float(tts_rows["eer_value"].mean()) if not tts_rows.empty else None
    average_vc = float(vc_rows["eer_value"].mean()) if not vc_rows.empty else None

    if average_tts is None or average_vc is None:
        conclusion = "N/A"
    elif np.isclose(average_tts, average_vc):
        conclusion = "TTS and VC equally hard to detect"
    elif average_tts > average_vc:
        conclusion = "TTS harder to detect"
    else:
        conclusion = "VC harder to detect"

    if hardest_attack is None or easiest_attack is None:
        report_paragraph = "Khong du du lieu EER hop le de tom tat attack-wise cho bao cao."
    else:
        report_paragraph = (
            f"Tren tap ASVspoof 2021 LA eval, attack kho detect nhat la {hardest_attack} "
            f"(EER = {hardest_eer:.2f}%), trong khi de detect nhat la {easiest_attack} "
            f"(EER = {easiest_eer:.2f}%). "
        )
        if average_tts is None or average_vc is None:
            report_paragraph += "Chua du du lieu de so sanh trung binh giua TTS va VC."
        elif np.isclose(average_tts, average_vc):
            report_paragraph += (
                f"TTS va VC co do kho tuong duong, voi EER trung binh lan luot la "
                f"{average_tts:.2f}% va {average_vc:.2f}%."
            )
        elif average_tts > average_vc:
            report_paragraph += (
                f"TTS kho hon VC, voi EER trung binh {average_tts:.2f}% so voi {average_vc:.2f}%."
            )
        else:
            report_paragraph += (
                f"VC kho hon TTS, voi EER trung binh {average_vc:.2f}% so voi {average_tts:.2f}%."
            )

    return {
        "hardest_attack": hardest_attack,
        "hardest_eer": hardest_eer,
        "easiest_attack": easiest_attack,
        "easiest_eer": easiest_eer,
        "average_tts": average_tts,
        "average_vc": average_vc,
        "conclusion": conclusion,
        "report_paragraph": report_paragraph,
    }


def build_attack_analysis(
    metadata_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    combo: str,
    schema: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    id_column = schema.get("id_column")
    if not id_column:
        raise ValueError("Prediction schema is missing an ID column, cannot merge.")

    prediction_frame = predictions_df.copy()
    prediction_frame["merge_key"] = prediction_frame[id_column].map(_normalize_utt_id)

    metadata_frame = metadata_df.rename(columns={"label": "metadata_label"}).copy()
    metadata_frame["merge_key"] = metadata_frame["utt_id"].map(_normalize_utt_id)

    merged = prediction_frame.merge(
        metadata_frame[["merge_key", "utt_id", "metadata_label", "attack_id", "type"]],
        on="merge_key",
        how="inner",
        suffixes=("_pred", "_meta"),
    )

    prediction_keys = prediction_frame["merge_key"].loc[prediction_frame["merge_key"].astype(bool)].drop_duplicates()
    metadata_keys = metadata_frame["merge_key"].loc[metadata_frame["merge_key"].astype(bool)].drop_duplicates()
    if merged.empty:
        raise ValueError(
            "Merge failed: no overlapping utt_id/file_id keys between predictions and eval metadata. "
            f"prediction_key_count={len(prediction_keys)}, metadata_key_count={len(metadata_keys)}, "
            f"prediction key examples={prediction_keys.head(5).tolist()}, "
            f"metadata key examples={metadata_keys.head(5).tolist()}"
        )

    merged["metadata_label_binary"] = merged["metadata_label"].map(METADATA_LABEL_MAP)
    if merged["metadata_label_binary"].isna().any():
        unexpected = sorted(merged.loc[merged["metadata_label_binary"].isna(), "metadata_label"].dropna().unique().tolist())
        raise ValueError(f"Unexpected metadata labels after merge: {unexpected}")
    merged["metadata_label_binary"] = merged["metadata_label_binary"].astype(np.int64)

    score_column = schema.get("score_column")
    positive_label_name = None
    score_means: dict[str, float] = {}
    if score_column is not None:
        merged["score_value"] = pd.to_numeric(merged[score_column], errors="coerce")
        if merged["score_value"].isna().any() or not np.isfinite(merged["score_value"].to_numpy(dtype=np.float64, copy=False)).all():
            raise ValueError(f"Detected score column '{score_column}' contains non-finite values after merge.")
        class_means = merged.groupby("metadata_label_binary", sort=True)["score_value"].mean().to_dict()
        if len(class_means) < 2:
            score_column = None
        else:
            positive_label_binary = max(class_means, key=class_means.get)
            positive_label_name = "bonafide" if positive_label_binary == 0 else "spoof"
            score_means = {
                "bonafide": float(class_means.get(0, float("nan"))),
                "spoof": float(class_means.get(1, float("nan"))),
            }
            merged["eer_label"] = (merged["metadata_label_binary"] == positive_label_binary).astype(np.int64)

    bonafide_frame = merged.loc[merged["metadata_label"] == "bonafide"].copy()
    rows: list[dict[str, Any]] = []
    for attack_id in ATTACK_IDS:
        attack_rows = merged.loc[merged["attack_id"] == attack_id].copy()
        note = ""
        eer_value = float("nan")
        level = "N/A"

        if attack_rows.empty:
            note = "missing from eval metadata/predictions"
        elif score_column is None:
            note = "missing continuous score"
        else:
            subset = pd.concat([bonafide_frame, attack_rows], ignore_index=True)
            labels = subset["eer_label"].to_numpy(dtype=np.int64, copy=False)
            scores = subset["score_value"].to_numpy(dtype=np.float64, copy=False)
            num_positive = int((labels == 1).sum())
            num_negative = int((labels == 0).sum())
            if num_positive == 0 or num_negative == 0:
                note = "insufficient bonafide/spoof samples"
            else:
                eer_value = compute_eer(labels, scores) * 100.0
                if np.isfinite(eer_value):
                    level = "Dễ" if eer_value < 10.0 else "Khó"
                else:
                    note = "insufficient bonafide/spoof samples"

        rows.append(
            {
                "attack_id": attack_id,
                "type": _attack_type(attack_id),
                "system": combo,
                "eer": _format_eer_percent(eer_value),
                "level": level,
                "note": note,
                "eer_value": eer_value if np.isfinite(eer_value) else np.nan,
            }
        )

    analysis_frame = pd.DataFrame(rows, columns=CSV_COLUMNS + ["eer_value"])
    summary = _build_summary(analysis_frame)
    debug_info = {
        "prediction_rows": int(len(predictions_df)),
        "metadata_eval_rows": int(len(metadata_df)),
        "merged_rows": int(len(merged)),
        "matched_keys": int(merged["merge_key"].nunique()),
        "prediction_keys": int(prediction_keys.nunique()),
        "metadata_keys": int(metadata_keys.nunique()),
        "score_column": score_column,
        "positive_label_name": positive_label_name,
        "score_means": score_means,
    }
    return analysis_frame, {"summary": summary, "debug": debug_info}


def write_outputs(
    analysis_frame: pd.DataFrame,
    summary: dict[str, Any],
    reports_dir: str | Path,
) -> dict[str, Any]:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    csv_path = reports_path / "attack_analysis.csv"
    summary_path = reports_path / "attack_analysis_summary.txt"

    analysis_frame.loc[:, CSV_COLUMNS].to_csv(csv_path, index=False)

    hardest_attack = (
        f"{summary['hardest_attack']} (EER = {summary['hardest_eer']:.2f}%)"
        if summary["hardest_attack"] is not None and summary["hardest_eer"] is not None
        else "N/A"
    )
    easiest_attack = (
        f"{summary['easiest_attack']} (EER = {summary['easiest_eer']:.2f}%)"
        if summary["easiest_attack"] is not None and summary["easiest_eer"] is not None
        else "N/A"
    )
    lines = [
        f"Hardest attack: {hardest_attack}",
        f"Easiest attack: {easiest_attack}",
        f"Average TTS EER: {_format_optional_summary_percent(summary['average_tts'])}",
        f"Average VC EER: {_format_optional_summary_percent(summary['average_vc'])}",
        f"Conclusion: {summary['conclusion']}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "csv_path": csv_path,
        "summary_path": summary_path,
        "summary_text": "\n".join(lines),
    }


def main() -> None:
    args = parse_args()

    metadata_path = REPO_ROOT / "data" / "raw" / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
    predictions_path = REPO_ROOT / "outputs" / "results" / args.combo / "predictions_2021.csv"
    reports_dir = REPO_ROOT / "outputs" / "reports"

    try:
        metadata_df = parse_trial_metadata(metadata_path)
        predictions_df, schema = detect_prediction_schema(predictions_path)
        analysis_frame, result = build_attack_analysis(metadata_df, predictions_df, args.combo, schema)
        output_info = write_outputs(analysis_frame, result["summary"], reports_dir)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    debug_info = result["debug"]
    print(
        "Merge stats: "
        f"prediction_rows={debug_info['prediction_rows']}, "
        f"metadata_eval_rows={debug_info['metadata_eval_rows']}, "
        f"merged_rows={debug_info['merged_rows']}, "
        f"matched_keys={debug_info['matched_keys']}, "
        f"prediction_keys={debug_info['prediction_keys']}, "
        f"metadata_keys={debug_info['metadata_keys']}"
    )
    if debug_info["score_column"] is None:
        print("Continuous score not available: cannot compute standard EER, CSV rows will show N/A.")
    else:
        bonafide_mean = debug_info["score_means"].get("bonafide")
        spoof_mean = debug_info["score_means"].get("spoof")
        print(
            "Score direction: "
            f"positive_class={debug_info['positive_label_name']}, "
            f"bonafide_mean={bonafide_mean:.6f}, "
            f"spoof_mean={spoof_mean:.6f}"
        )

    print(f"Wrote CSV: {_display_path(output_info['csv_path'])}")
    print(f"Wrote summary: {_display_path(output_info['summary_path'])}")
    print("Preview (first 5 rows):")
    print(analysis_frame.loc[:, CSV_COLUMNS].head(5).to_csv(index=False, lineterminator="\n").strip())
    print("Report summary:")
    print(result["summary"]["report_paragraph"])


if __name__ == "__main__":
    main()
