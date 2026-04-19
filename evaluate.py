from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from src.utils.metrics import save_prediction_csv
from src.utils.runtime import (
    configure_torch_runtime,
    create_spoof_dataloader,
    default_batch_size,
    experiment_output_paths,
    load_checkpoint_bundle,
    resolve_device,
    resolve_num_workers,
    run_evaluation_loop,
    upsert_rows,
)
from src.utils.visualize import save_confusion_matrix_figure


REPO_ROOT = Path(__file__).resolve().parent


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized):
        return None
    return normalized


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_checkpoint_for_json(checkpoint_path: str | Path) -> str:
    resolved_path = Path(checkpoint_path).resolve()
    try:
        return resolved_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def _build_metrics_payload(*, eer: object, accuracy: object, num_samples: object) -> Dict[str, object]:
    return {
        "eer": _safe_float(eer),
        "accuracy": _safe_float(accuracy),
        "num_samples": _safe_int(num_samples),
    }


def _write_metrics_json(
    *,
    result_dir: Path,
    model_name: str,
    feature_name: str,
    checkpoint_path: str | Path,
    checkpoint_bundle: Dict[str, object],
    result_2019: Dict[str, object] | None,
    result_2021: Dict[str, object] | None,
    effective_batch_size: int,
    effective_num_workers: int,
    resolved_device,
) -> Path:
    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "feature": feature_name,
        "checkpoint": _resolve_checkpoint_for_json(checkpoint_path),
        "best_epoch": _safe_int(checkpoint_bundle.get("best_epoch", checkpoint_bundle.get("epoch"))),
        "metrics_2019": (
            _build_metrics_payload(
                eer=result_2019.get("eer"),
                accuracy=result_2019.get("accuracy"),
                num_samples=result_2019.get("num_samples"),
            )
            if result_2019 is not None
            else None
        ),
        "metrics_2021": (
            _build_metrics_payload(
                eer=result_2021.get("eer_2021"),
                accuracy=result_2021.get("accuracy"),
                num_samples=result_2021.get("num_samples"),
            )
            if result_2021 is not None
            else None
        ),
        "config": {
            "batch_size": _safe_int(effective_batch_size),
            "num_workers": _safe_int(effective_num_workers),
            "device": str(resolved_device),
        },
    }
    output_path = result_dir / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    return output_path


def _write_eer_comparison(result_dir: Path, result_2019: Dict[str, object], result_2021: Dict[str, object]) -> None:
    comparison_row = {
        "profile": result_2021["profile"],
        "model": result_2021["model"],
        "feature": result_2021["feature"],
        "eer_2019": result_2021["eer_2019"],
        "eer_2021": result_2021["eer_2021"],
        "generalization_gap": result_2021["generalization_gap"],
        "checkpoint": result_2021["checkpoint"],
    }
    upsert_rows(
        csv_path=result_dir / "eer_comparison.csv",
        rows=[comparison_row],
        key_columns=["checkpoint"],
        sort_columns=["checkpoint"],
    )


def evaluate_checkpoint(
    *,
    checkpoint_path: str,
    data_root: str = "data",
    protocol_root: str | None = None,
    feature_root: str | None = None,
    output_root: str = "outputs",
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
    run_2019: bool = True,
    run_2021: bool = False,
    eval_2021_labels: str | None = None,
    eval_2021_feature_root: str | None = None,
) -> Dict[str, object]:
    resolved_device = resolve_device(device)
    configure_torch_runtime(resolved_device)
    model, checkpoint = load_checkpoint_bundle(checkpoint_path, device="cpu")
    model = model.to(resolved_device)

    profile = checkpoint.get("profile", "custom")
    model_name = str(checkpoint["model_name"])
    feature_name = str(checkpoint["feature_name"])
    target_frames = int(checkpoint.get("target_frames", 128))
    normalize = bool(checkpoint.get("normalize", False))
    hardware_batch_size = default_batch_size(model_name, feature_name, device_name=resolved_device)
    effective_batch_size = batch_size if batch_size is not None else hardware_batch_size
    effective_num_workers = resolve_num_workers(num_workers)
    use_amp = bool(checkpoint.get("amp", False)) and resolved_device.type == "cuda"
    output_paths = experiment_output_paths(output_root, model_name, feature_name)
    result_dir = output_paths["result_dir"]
    figure_dir = output_paths["figure_dir"]
    resolved_checkpoint_path = str(Path(checkpoint_path).resolve())
    metrics_json_path = result_dir / "metrics.json"

    result_2019 = None
    result_2021 = None
    if run_2019:
        eval_loader_2019 = create_spoof_dataloader(
            feature_name=feature_name,
            split="eval",
            target_frames=target_frames,
            batch_size=effective_batch_size,
            data_root=data_root,
            protocol_root=protocol_root,
            feature_root=feature_root,
            normalize=normalize,
            specaugment=False,
            training=False,
            num_workers=effective_num_workers,
            device=resolved_device,
        )
        metrics_2019 = run_evaluation_loop(
            model=model,
            dataloader=eval_loader_2019,
            device=resolved_device,
            criterion=None,
            use_amp=use_amp,
        )
        save_prediction_csv(
            output_path=result_dir / "predictions_2019.csv",
            utt_ids=metrics_2019.utt_ids,
            labels=metrics_2019.labels,
            scores=metrics_2019.scores,
            predictions=metrics_2019.predictions,
        )
        save_confusion_matrix_figure(
            matrix=metrics_2019.confusion,
            output_path=figure_dir / "confusion_matrix_2019.png",
            title=f"{model_name.upper()} + {feature_name.upper()} ASVspoof2019",
        )
        result_2019 = {
            "profile": profile,
            "model": model_name,
            "feature": feature_name,
            "accuracy": metrics_2019.accuracy,
            "eer": metrics_2019.eer,
            "num_samples": int(len(metrics_2019.labels)) if getattr(metrics_2019, "labels", None) is not None else len(metrics_2019.utt_ids),
            "checkpoint": resolved_checkpoint_path,
        }
        upsert_rows(
            csv_path=result_dir / "results_2019.csv",
            rows=[result_2019],
            key_columns=["checkpoint"],
            sort_columns=["checkpoint"],
        )
        metrics_json_path = _write_metrics_json(
            result_dir=result_dir,
            model_name=model_name,
            feature_name=feature_name,
            checkpoint_path=checkpoint_path,
            checkpoint_bundle=checkpoint,
            result_2019=result_2019,
            result_2021=result_2021,
            effective_batch_size=effective_batch_size,
            effective_num_workers=effective_num_workers,
            resolved_device=resolved_device,
        )

    if run_2021:
        if result_2019 is None:
            result_2019 = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                data_root=data_root,
                protocol_root=protocol_root,
                feature_root=feature_root,
                output_root=output_root,
                batch_size=effective_batch_size,
                num_workers=effective_num_workers,
                device=device,
                run_2019=True,
                run_2021=False,
            )["result_2019"]

        eval_loader_2021 = create_spoof_dataloader(
            feature_name=feature_name,
            split="eval_2021",
            target_frames=target_frames,
            batch_size=effective_batch_size,
            data_root=data_root,
            feature_root=eval_2021_feature_root or feature_root,
            normalize=normalize,
            specaugment=False,
            training=False,
            num_workers=effective_num_workers,
            device=resolved_device,
            labels_path=eval_2021_labels,
        )
        metrics_2021 = run_evaluation_loop(
            model=model,
            dataloader=eval_loader_2021,
            device=resolved_device,
            criterion=None,
            use_amp=use_amp,
        )
        save_prediction_csv(
            output_path=result_dir / "predictions_2021.csv",
            utt_ids=metrics_2021.utt_ids,
            labels=metrics_2021.labels,
            scores=metrics_2021.scores,
            predictions=metrics_2021.predictions,
        )
        result_2021 = {
            "profile": profile,
            "model": model_name,
            "feature": feature_name,
            "eer_2021": metrics_2021.eer,
            "accuracy": metrics_2021.accuracy,
            "num_samples": int(len(metrics_2021.labels)) if getattr(metrics_2021, "labels", None) is not None else len(metrics_2021.utt_ids),
            "eer_2019": float(result_2019["eer"]),
            "generalization_gap": float(metrics_2021.eer) - float(result_2019["eer"]),
            "checkpoint": resolved_checkpoint_path,
        }
        upsert_rows(
            csv_path=result_dir / "results_2021.csv",
            rows=[result_2021],
            key_columns=["checkpoint"],
            sort_columns=["checkpoint"],
        )
        _write_eer_comparison(result_dir, result_2019, result_2021)
        metrics_json_path = _write_metrics_json(
            result_dir=result_dir,
            model_name=model_name,
            feature_name=feature_name,
            checkpoint_path=checkpoint_path,
            checkpoint_bundle=checkpoint,
            result_2019=result_2019,
            result_2021=result_2021,
            effective_batch_size=effective_batch_size,
            effective_num_workers=effective_num_workers,
            resolved_device=resolved_device,
        )

    return {
        "profile": profile,
        "model": model_name,
        "feature": feature_name,
        "result_2019": result_2019,
        "result_2021": result_2021,
        "result_dir": result_dir,
        "figure_dir": figure_dir,
        "metrics_json_path": metrics_json_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained voice spoofing detection model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_2019", action="store_true")
    parser.add_argument("--eval_2021", action="store_true")
    parser.add_argument("--feature_root", default=None)
    parser.add_argument("--eval_2021_features", default=None)
    parser.add_argument("--eval_2021_labels", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--protocol_root", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_2019 = args.eval_2019 or not args.eval_2021
    run_2021 = args.eval_2021
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        protocol_root=args.protocol_root,
        feature_root=args.feature_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        run_2019=run_2019,
        run_2021=run_2021,
        eval_2021_labels=args.eval_2021_labels,
        eval_2021_feature_root=args.eval_2021_features,
    )
    print(f"Evaluation complete for {result['profile']} ({result['model']} + {result['feature']}).")


if __name__ == "__main__":
    main()
