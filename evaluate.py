from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from src.utils.metrics import save_prediction_csv
from src.utils.visualize import save_confusion_matrix_figure
from train import create_spoof_dataloader, load_checkpoint_bundle, resolve_device, run_evaluation_loop, upsert_rows


def _load_existing_2019_result(results_path: Path, profile: str, model: str, feature: str, checkpoint_path: str):
    if not results_path.exists():
        return None
    dataframe = pd.read_csv(results_path)
    matched = dataframe[
        (dataframe["profile"] == profile)
        & (dataframe["model"] == model)
        & (dataframe["feature"] == feature)
        & (dataframe["checkpoint"] == checkpoint_path)
    ]
    if matched.empty:
        return None
    return matched.iloc[-1].to_dict()


def _write_eer_comparison(output_root: Path) -> None:
    results_2019_path = output_root / "results" / "results_2019.csv"
    results_2021_path = output_root / "results" / "results_2021.csv"
    if not results_2019_path.exists() or not results_2021_path.exists():
        return

    results_2019 = pd.read_csv(results_2019_path)
    results_2021 = pd.read_csv(results_2021_path)
    merged = results_2021.merge(
        results_2019[["profile", "model", "feature", "eer"]].rename(columns={"eer": "eer_2019"}),
        on=["profile", "model", "feature"],
        how="left",
        suffixes=("", "_dup"),
    )
    if "eer_2019_dup" in merged.columns:
        merged = merged.drop(columns=["eer_2019_dup"])
    comparison = merged[["profile", "model", "feature", "eer_2019", "eer_2021", "generalization_gap"]]
    comparison.to_csv(output_root / "results" / "eer_comparison.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained voice spoofing detection model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_2019", action="store_true")
    parser.add_argument("--eval_2021", action="store_true")
    parser.add_argument("--eval_2021_features", default=None)
    parser.add_argument("--eval_2021_labels", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--protocol_root", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    model, checkpoint = load_checkpoint_bundle(args.checkpoint, device="cpu")
    model = model.to(device)

    profile = checkpoint.get("profile", "custom")
    model_name = checkpoint["model_name"]
    feature_name = checkpoint["feature_name"]
    target_frames = int(checkpoint.get("target_frames", 128))
    normalize = bool(checkpoint.get("normalize", False))
    batch_size = args.batch_size or int(checkpoint.get("batch_size", 64 if model_name == "cnn" else 32))
    use_amp = bool(checkpoint.get("amp", False)) and device.type == "cuda"

    run_2019 = args.eval_2019 or not args.eval_2021
    result_2019 = None

    if run_2019:
        eval_loader_2019 = create_spoof_dataloader(
            feature_name=feature_name,
            split="eval",
            target_frames=target_frames,
            batch_size=batch_size,
            data_root=args.data_root,
            protocol_root=args.protocol_root,
            normalize=normalize,
            specaugment=False,
            training=False,
            num_workers=args.num_workers,
            device=device,
        )
        metrics_2019 = run_evaluation_loop(
            model=model,
            dataloader=eval_loader_2019,
            device=device,
            criterion=None,
            use_amp=use_amp,
        )
        save_prediction_csv(
            output_path=output_root / "results" / f"predictions_{profile}_2019.csv",
            utt_ids=metrics_2019.utt_ids,
            labels=metrics_2019.labels,
            scores=metrics_2019.scores,
            predictions=metrics_2019.predictions,
        )
        save_confusion_matrix_figure(
            matrix=metrics_2019.confusion,
            output_path=output_root / "figures" / f"{profile}_confusion_matrix_2019.png",
            title=f"{profile.title()} ASVspoof2019 Eval Confusion Matrix",
        )
        result_2019 = {
            "profile": profile,
            "model": model_name,
            "feature": feature_name,
            "accuracy": metrics_2019.accuracy,
            "eer": metrics_2019.eer,
            "checkpoint": str(Path(args.checkpoint).resolve()),
        }
        upsert_rows(
            csv_path=output_root / "results" / "results_2019.csv",
            rows=[result_2019],
            key_columns=["profile", "model", "feature"],
            sort_columns=["profile", "model", "feature"],
        )

    if args.eval_2021:
        if result_2019 is None:
            result_2019 = _load_existing_2019_result(
                results_path=output_root / "results" / "results_2019.csv",
                profile=profile,
                model=model_name,
                feature=feature_name,
                checkpoint_path=str(Path(args.checkpoint).resolve()),
            )
            if result_2019 is None:
                eval_loader_2019 = create_spoof_dataloader(
                    feature_name=feature_name,
                    split="eval",
                    target_frames=target_frames,
                    batch_size=batch_size,
                    data_root=args.data_root,
                    protocol_root=args.protocol_root,
                    normalize=normalize,
                    specaugment=False,
                    training=False,
                    num_workers=args.num_workers,
                    device=device,
                )
                metrics_2019 = run_evaluation_loop(
                    model=model,
                    dataloader=eval_loader_2019,
                    device=device,
                    criterion=None,
                    use_amp=use_amp,
                )
                result_2019 = {
                    "profile": profile,
                    "model": model_name,
                    "feature": feature_name,
                    "accuracy": metrics_2019.accuracy,
                    "eer": metrics_2019.eer,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                }
                upsert_rows(
                    csv_path=output_root / "results" / "results_2019.csv",
                    rows=[result_2019],
                    key_columns=["profile", "model", "feature"],
                    sort_columns=["profile", "model", "feature"],
                )

        eval_loader_2021 = create_spoof_dataloader(
            feature_name=feature_name,
            split="eval_2021",
            target_frames=target_frames,
            batch_size=batch_size,
            data_root=args.data_root,
            feature_root=args.eval_2021_features,
            normalize=normalize,
            specaugment=False,
            training=False,
            num_workers=args.num_workers,
            device=device,
            labels_path=args.eval_2021_labels,
        )
        metrics_2021 = run_evaluation_loop(
            model=model,
            dataloader=eval_loader_2021,
            device=device,
            criterion=None,
            use_amp=use_amp,
        )
        save_prediction_csv(
            output_path=output_root / "results" / f"predictions_{profile}_2021.csv",
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
            "eer_2019": float(result_2019["eer"]),
            "generalization_gap": float(metrics_2021.eer) - float(result_2019["eer"]),
            "checkpoint": str(Path(args.checkpoint).resolve()),
        }
        upsert_rows(
            csv_path=output_root / "results" / "results_2021.csv",
            rows=[result_2021],
            key_columns=["profile", "model", "feature"],
            sort_columns=["profile", "model", "feature"],
        )
        _write_eer_comparison(output_root)

    print(f"Evaluation complete for {profile} ({model_name} + {feature_name}).")


if __name__ == "__main__":
    main()
