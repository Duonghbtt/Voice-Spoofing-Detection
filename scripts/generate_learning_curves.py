from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.runtime import experiment_dir_name


AGGREGATE_REQUIRED_COLUMNS = ("model", "feature", "epoch", "train_loss", "val_loss")
REQUIRED_PLOT_COLUMNS = ("epoch", "train_loss", "val_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate learning curve figures from training logs.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate learning_curve.png even when it already exists.",
    )
    return parser.parse_args()


def load_aggregate_groups(aggregate_log: Path) -> dict[str, pd.DataFrame]:
    if not aggregate_log.exists():
        return {}

    try:
        frame = pd.read_csv(aggregate_log)
    except pd.errors.EmptyDataError:
        return {}
    except Exception:
        return {}

    if not set(AGGREGATE_REQUIRED_COLUMNS).issubset(frame.columns):
        return {}

    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for _, row in frame.iterrows():
        try:
            combo = experiment_dir_name(str(row["model"]), str(row["feature"]))
        except Exception:
            continue
        grouped_rows[combo].append(row.to_dict())

    return {
        combo: pd.DataFrame(rows, columns=frame.columns)
        for combo, rows in grouped_rows.items()
    }


def clean_learning_curve_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(REQUIRED_PLOT_COLUMNS)
    if "val_eer" in frame.columns:
        columns.append("val_eer")

    cleaned = frame.loc[:, columns].copy()

    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=list(REQUIRED_PLOT_COLUMNS))
    cleaned = cleaned.loc[cleaned["epoch"] > 0]
    cleaned = cleaned.drop_duplicates(subset="epoch", keep="last")
    cleaned = cleaned.sort_values("epoch", kind="stable").reset_index(drop=True)
    return cleaned


def plot_learning_curve(frame: pd.DataFrame, combo: str, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["epoch"], frame["train_loss"], label="train_loss", linewidth=2)
    ax.plot(frame["epoch"], frame["val_loss"], label="val_loss", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(combo)
    ax.grid(True, alpha=0.3)

    secondary_axis = None
    if "val_eer" in frame.columns:
        eer_frame = frame.loc[frame["val_eer"].notna(), ["epoch", "val_eer"]]
        if not eer_frame.empty:
            secondary_axis = ax.twinx()
            secondary_axis.plot(
                eer_frame["epoch"],
                eer_frame["val_eer"],
                label="val_eer",
                linewidth=2,
                linestyle="--",
            )
            secondary_axis.set_ylabel("val_eer")

    handles, labels = ax.get_legend_handles_labels()
    if secondary_axis is not None:
        secondary_handles, secondary_labels = secondary_axis.get_legend_handles_labels()
        handles.extend(secondary_handles)
        labels.extend(secondary_labels)
    ax.legend(handles, labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    results_root = REPO_ROOT / "outputs" / "results"
    figures_root = REPO_ROOT / "outputs" / "figures"
    aggregate_log = results_root / "train_log.csv"

    aggregate_groups = load_aggregate_groups(aggregate_log)

    if not results_root.exists():
        return

    for experiment_dir in sorted(results_root.iterdir()):
        if not experiment_dir.is_dir():
            continue

        combo = experiment_dir.name
        train_log_path = results_root / combo / "train_log.csv"
        output_png = figures_root / combo / "learning_curve.png"

        try:
            if output_png.exists() and not args.force:
                print(f"[SKIP FIG] {combo} exists")
                continue

            if train_log_path.exists():
                try:
                    frame = pd.read_csv(train_log_path)
                except pd.errors.EmptyDataError:
                    print(f"[SKIP FIG] {combo} empty after cleaning")
                    continue
            elif combo in aggregate_groups:
                frame = aggregate_groups[combo].copy()
            else:
                print(f"[SKIP FIG] {combo} missing train log")
                continue

            if not set(REQUIRED_PLOT_COLUMNS).issubset(frame.columns):
                print(f"[SKIP FIG] {combo} missing required columns")
                continue

            cleaned_frame = clean_learning_curve_frame(frame)
            if cleaned_frame.empty:
                print(f"[SKIP FIG] {combo} empty after cleaning")
                continue

            plot_learning_curve(cleaned_frame, combo, output_png)
            print(f"[GEN FIG] {combo} -> learning_curve.png")
        except Exception:
            print(f"[SKIP FIG] {combo} read error")


if __name__ == "__main__":
    main()
