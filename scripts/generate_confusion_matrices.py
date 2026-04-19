from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMN_PAIRS = (
    ("y_true", "y_pred"),
    ("label", "prediction"),
    ("label", "pred_label"),
)
LABEL_MAP = {
    "0": 0,
    "0.0": 0,
    "1": 1,
    "1.0": 1,
    "bonafide": 0,
    "bona_fide": 0,
    "genuine": 0,
    "spoof": 1,
    "fake": 1,
    0: 0,
    1: 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate confusion matrix figures from existing result CSV files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate confusion_matrix.png even when it already exists.",
    )
    return parser.parse_args()


def parse_binary_label(value: object) -> int:
    if value in LABEL_MAP:
        return LABEL_MAP[value]

    normalized = str(value).strip().lower()
    if normalized in LABEL_MAP:
        return LABEL_MAP[normalized]

    float_value = float(normalized)
    int_value = int(float_value)
    if float_value != int_value or int_value not in (0, 1):
        raise ValueError(f"Expected a binary label, received {value!r}.")
    return int_value


def candidate_source_paths(experiment_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path.exists() and path not in seen:
            candidates.append(path)
            seen.add(path)

    add(experiment_dir / "predictions_2019.csv")
    for path in sorted(experiment_dir.glob("predictions*.csv")):
        add(path)
    add(experiment_dir / "results_2019.csv")
    add(experiment_dir / "results_2021.csv")
    return candidates


def resolve_label_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    for true_column, pred_column in COLUMN_PAIRS:
        if true_column in frame.columns and pred_column in frame.columns:
            return true_column, pred_column
    return None


def clean_confusion_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    label_columns = resolve_label_columns(frame)
    if label_columns is None:
        return None

    true_column, pred_column = label_columns
    cleaned = frame.loc[:, [true_column, pred_column]].copy()
    cleaned.columns = ["y_true", "y_pred"]
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    cleaned = cleaned.dropna(subset=["y_true", "y_pred"])
    if cleaned.empty:
        return pd.DataFrame(columns=["y_true", "y_pred"])

    rows: list[dict[str, int]] = []
    for true_value, pred_value in zip(cleaned["y_true"], cleaned["y_pred"]):
        try:
            rows.append(
                {
                    "y_true": parse_binary_label(true_value),
                    "y_pred": parse_binary_label(pred_value),
                }
            )
        except (TypeError, ValueError):
            continue

    return pd.DataFrame(rows, columns=["y_true", "y_pred"])


def load_confusion_data(experiment_dir: Path) -> tuple[pd.DataFrame | None, Path | None, str | None]:
    saw_supported_columns = False
    saw_no_valid_rows = False

    for source_path in candidate_source_paths(experiment_dir):
        try:
            frame = pd.read_csv(source_path)
        except pd.errors.EmptyDataError:
            continue

        cleaned = clean_confusion_frame(frame)
        if cleaned is None:
            continue

        saw_supported_columns = True
        if cleaned.empty:
            saw_no_valid_rows = True
            continue

        return cleaned, source_path, None

    if saw_no_valid_rows:
        return None, None, "no valid rows"
    if saw_supported_columns:
        return None, None, "no valid rows"
    return None, None, "missing predictions"


def plot_confusion_matrix(frame: pd.DataFrame, combo: str, output_png: Path) -> None:
    labels = sorted(set(frame["y_true"].tolist()) | set(frame["y_pred"].tolist()))
    matrix = confusion_matrix(frame["y_true"], frame["y_pred"], labels=labels)

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        image = ax.imshow(matrix, cmap="Blues")
        tick_labels = [str(label) for label in labels]
        ax.set_xticks(range(len(labels)), labels=tick_labels)
        ax.set_yticks(range(len(labels)), labels=tick_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(combo)

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_png, dpi=200)
    finally:
        plt.close(fig)


def generate_confusion_matrices(results_root: Path, figures_root: Path, force: bool = False) -> None:
    if not results_root.exists():
        return

    for experiment_dir in sorted(results_root.iterdir()):
        if not experiment_dir.is_dir():
            continue

        combo = experiment_dir.name
        output_png = figures_root / combo / "confusion_matrix.png"

        try:
            if output_png.exists() and not force:
                print(f"[SKIP CM] {combo} exists")
                continue

            cleaned_frame, _, reason = load_confusion_data(experiment_dir)
            if cleaned_frame is None:
                print(f"[SKIP CM] {combo} {reason}")
                continue

            plot_confusion_matrix(cleaned_frame, combo, output_png)
            print(f"[GEN CM] {combo} -> confusion_matrix.png")
        except Exception:
            print(f"[SKIP CM] {combo} read error")


def main() -> None:
    args = parse_args()
    results_root = REPO_ROOT / "outputs" / "results"
    figures_root = REPO_ROOT / "outputs" / "figures"
    generate_confusion_matrices(results_root=results_root, figures_root=figures_root, force=args.force)


if __name__ == "__main__":
    main()
