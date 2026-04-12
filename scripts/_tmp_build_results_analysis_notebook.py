from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = []

cells.append(md("""
# Voice Spoofing Detection - Results Analysis

## Muc tieu notebook
Notebook nay tong hop va phan tich ket qua cuoi cung cho vai tro **Nguoi 3 - Evaluator & Analyst** trong do an Voice Spoofing Detection.

- So sanh hieu nang giua **MFCC / LFCC / Spectrogram**.
- So sanh cac mo hinh **CNN / ResNet / LCNN** tren **ASVspoof 2019 eval**.
- Phan tich **generalization gap** khi chuyen tu **ASVspoof 2019** sang **ASVspoof 2021**.
- Phan tich **attack-wise EER** va score distribution de tim ra cac truong hop kho.
- Tao bang va hinh san sang chen vao bao cao cuoi ky.

Notebook duoc thiet ke de **khong crash khi thieu file ket qua**. Neu mot artifact chua ton tai, notebook se in warning than thien va tiep tuc hoat dong.
"""))

cells.append(code(r'''
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

from IPython.display import Markdown, display

plt.style.use("default")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 120)


def guess_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().resolve()]
    candidates.extend(Path.cwd().resolve().parents)
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "outputs").exists() and (candidate / "notebooks").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = guess_project_root()
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_results_analysis.ipynb"


def warn_missing(path: Path) -> None:
    print(f"[Warning] Khong tim thay file: {path}")


def safe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        warn_missing(path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        print(f"[Warning] Khong doc duoc CSV {path}: {exc}")
        return pd.DataFrame()


def safe_read_json(path: str | Path):
    path = Path(path)
    if not path.exists():
        warn_missing(path)
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"[Warning] Khong doc duoc JSON {path}: {exc}")
        return None


def display_if_exists(path: str | Path, title: str | None = None, figsize=(8, 5)) -> None:
    path = Path(path)
    if not path.exists():
        warn_missing(path)
        return
    try:
        image = plt.imread(path)
        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.axis("off")
        plt.title(title or path.name)
        plt.show()
    except Exception as exc:
        print(f"[Warning] Khong hien thi duoc anh {path}: {exc}")


def show_df_overview(name: str, df: pd.DataFrame, preview_rows: int = 5) -> None:
    print(f"\n{name}")
    if df is None or df.empty:
        print("  -> DataFrame rong hoac chua co du lieu.")
        return
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    display(df.head(preview_rows))


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def standardize_results_2019(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()
    aliases = {"eer_2019": "eer", "acc": "accuracy"}
    for src, dst in aliases.items():
        if src in result.columns and dst not in result.columns:
            result = result.rename(columns={src: dst})
    for column in ["profile", "model", "feature"]:
        if column not in result.columns:
            result[column] = "unknown"
        result[column] = result[column].astype(str)
    return _coerce_numeric_columns(result, ["accuracy", "eer", "precision", "recall", "f1", "eer_threshold", "num_samples"])


def standardize_results_2021(df: pd.DataFrame, df_2019: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()
    if "eer" in result.columns and "eer_2021" not in result.columns:
        result = result.rename(columns={"eer": "eer_2021"})
    for column in ["profile", "model", "feature"]:
        if column not in result.columns:
            result[column] = "unknown"
        result[column] = result[column].astype(str)
    result = _coerce_numeric_columns(result, ["accuracy", "eer_2019", "eer_2021", "generalization_gap", "precision", "recall", "f1"])
    if "eer_2019" not in result.columns and df_2019 is not None and not df_2019.empty and "eer" in df_2019.columns:
        result = result.merge(df_2019[["profile", "model", "feature", "eer"]].rename(columns={"eer": "eer_2019"}), on=["profile", "model", "feature"], how="left")
    if "generalization_gap" not in result.columns and {"eer_2019", "eer_2021"}.issubset(result.columns):
        result["generalization_gap"] = result["eer_2021"] - result["eer_2019"]
    return result


def standardize_eer_comparison(df: pd.DataFrame, df_2019: pd.DataFrame | None = None, df_2021: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is not None and not df.empty:
        result = df.copy()
    elif df_2021 is not None and not df_2021.empty:
        result = df_2021.copy()
    else:
        return pd.DataFrame()
    if "eer" in result.columns and "eer_2021" not in result.columns:
        result = result.rename(columns={"eer": "eer_2021"})
    for column in ["profile", "model", "feature"]:
        if column not in result.columns:
            result[column] = "unknown"
        result[column] = result[column].astype(str)
    result = _coerce_numeric_columns(result, ["eer_2019", "eer_2021", "generalization_gap"])
    if "eer_2019" not in result.columns and df_2019 is not None and not df_2019.empty and "eer" in df_2019.columns:
        result = result.merge(df_2019[["profile", "model", "feature", "eer"]].rename(columns={"eer": "eer_2019"}), on=["profile", "model", "feature"], how="left")
    if "eer_2021" not in result.columns and df_2021 is not None and not df_2021.empty and "eer_2021" in df_2021.columns:
        result = result.merge(df_2021[["profile", "model", "feature", "eer_2021"]], on=["profile", "model", "feature"], how="left")
    if "generalization_gap" not in result.columns and {"eer_2019", "eer_2021"}.issubset(result.columns):
        result["generalization_gap"] = result["eer_2021"] - result["eer_2019"]
    columns = [column for column in ["profile", "model", "feature", "eer_2019", "eer_2021", "generalization_gap"] if column in result.columns]
    return result[columns].drop_duplicates().reset_index(drop=True)


def safe_bool_series(series: pd.Series) -> pd.Series:
    mapping = {True: True, False: False, "True": True, "False": False, "true": True, "false": False, 1: True, 0: False}
    return series.map(mapping).fillna(False).astype(bool)


def parse_result_filename(path: Path, prefix: str) -> dict:
    stem = path.stem
    info = {"file": path.name, "path": path}
    remainder = stem[len(prefix) + 1 :] if stem.startswith(prefix + "_") else stem
    parts = remainder.split("_")
    if parts and parts[-1].isdigit():
        info["dataset"] = parts[-1]
        parts = parts[:-1]
    info["tag"] = "_".join(parts) if parts else "unknown"
    return info


print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"RESULTS_DIR:  {RESULTS_DIR}")
print(f"FIGURES_DIR:  {FIGURES_DIR}")
print(f"Notebook:     {NOTEBOOK_PATH}")
print(f"So file ket qua hien co: {len(list(RESULTS_DIR.glob('*')))}")
print(f"So file figure hien co: {len(list(FIGURES_DIR.glob('*')))}")
'''))
cells.append(md("""
## Load Experiment Results
Doc cac bang ket qua tong hop, bang generalization, metrics JSON va preview de kiem tra cac cot du lieu truoc khi phan tich.
"""))

cells.append(code(r'''
results_2019_raw = safe_read_csv(RESULTS_DIR / "results_2019.csv")
results_2021_raw = safe_read_csv(RESULTS_DIR / "results_2021.csv")
eer_comparison_raw = safe_read_csv(RESULTS_DIR / "eer_comparison.csv")

results_2019 = standardize_results_2019(results_2019_raw)
results_2021 = standardize_results_2021(results_2021_raw, df_2019=results_2019)
eer_comparison = standardize_eer_comparison(eer_comparison_raw, df_2019=results_2019, df_2021=results_2021)

metrics_records = []
for metrics_path in sorted(RESULTS_DIR.glob("metrics_*.json")):
    payload = safe_read_json(metrics_path)
    if isinstance(payload, dict):
        payload["source_file"] = metrics_path.name
        metrics_records.append(payload)
metrics_df = pd.DataFrame(metrics_records)

show_df_overview("results_2019", results_2019)
show_df_overview("results_2021", results_2021)
show_df_overview("eer_comparison", eer_comparison)
show_df_overview("metrics_json", metrics_df)
'''))

cells.append(md("""
## Ket Qua Tren ASVspoof 2019
Phan nay tap trung vao hieu nang tren tap eval 2019, dac biet la **EER** va **accuracy**. EER cang thap cang tot.
"""))

cells.append(code(r'''
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))

if results_2019.empty:
    print("Chua co du lieu results_2019.csv de phan tich.")
else:
    ordered = results_2019.sort_values(by=["eer", "accuracy"], ascending=[True, False]).reset_index(drop=True)
    report_cols = [column for column in ["profile", "model", "feature", "accuracy", "precision", "recall", "f1", "eer", "eer_threshold", "num_samples", "checkpoint"] if column in ordered.columns]
    report_table = ordered[report_cols].copy()
    format_dict = {col: "{:.4f}" for col in report_table.select_dtypes(include=[np.number]).columns}
    display(report_table.style.format(format_dict).highlight_min(subset=[col for col in ["eer"] if col in report_table.columns], color="#d8f3dc").highlight_max(subset=[col for col in ["accuracy"] if col in report_table.columns], color="#ffe5d9"))

    if {"feature", "model", "eer"}.issubset(ordered.columns):
        pivot_eer = ordered.pivot_table(index="feature", columns="model", values="eer", aggfunc="min")
        print("\nPivot EER (lower is better)")
        display(pivot_eer.style.format("{:.4f}").highlight_min(axis=None, color="#d8f3dc"))

    if {"feature", "model", "accuracy"}.issubset(ordered.columns):
        pivot_acc = ordered.pivot_table(index="feature", columns="model", values="accuracy", aggfunc="max")
        print("\nPivot Accuracy (higher is better)")
        display(pivot_acc.style.format("{:.4f}").highlight_max(axis=None, color="#ffe5d9"))

    best_eer_row = ordered.loc[ordered["eer"].idxmin()] if ordered["eer"].notna().any() else None
    best_acc_row = ordered.loc[ordered["accuracy"].idxmax()] if "accuracy" in ordered.columns and ordered["accuracy"].notna().any() else None
    best_feature = ordered.groupby("feature", dropna=False)["eer"].mean().sort_values().index[0] if {"feature", "eer"}.issubset(ordered.columns) else None
    best_model = ordered.groupby("model", dropna=False)["eer"].mean().sort_values().index[0] if {"model", "eer"}.issubset(ordered.columns) else None

    if best_eer_row is not None:
        print(f"\nTo hop co EER thap nhat: {best_eer_row['profile']} | {best_eer_row['model']} + {best_eer_row['feature']} | EER={best_eer_row['eer']:.4f}")
    if best_acc_row is not None:
        print(f"To hop co accuracy cao nhat: {best_acc_row['profile']} | {best_acc_row['model']} + {best_acc_row['feature']} | ACC={best_acc_row['accuracy']:.4f}")
    if best_feature is not None:
        print(f"Dac trung co EER trung binh tot nhat: {best_feature}")
    if best_model is not None:
        print(f"Mo hinh co EER trung binh tot nhat: {best_model}")
'''))

cells.append(md("""
## Generalization Gap: ASVspoof 2019 -> 2021
Phan nay so sanh kha nang tong quat hoa. Neu **generalization gap** lon, mo hinh co the da hoc nhieu artifact dac thu cua tap 2019 thay vi hoc dac trung gian mao ben vung.
"""))

cells.append(code(r'''
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))
results_2021 = standardize_results_2021(safe_read_csv(RESULTS_DIR / "results_2021.csv"), df_2019=results_2019)
eer_comparison = standardize_eer_comparison(safe_read_csv(RESULTS_DIR / "eer_comparison.csv"), df_2019=results_2019, df_2021=results_2021)

if eer_comparison.empty:
    print("Chua co du lieu de phan tich generalization gap.")
else:
    gap_table = eer_comparison.sort_values(by="generalization_gap", ascending=False).reset_index(drop=True)
    display(gap_table.style.format({"eer_2019": "{:.4f}", "eer_2021": "{:.4f}", "generalization_gap": "{:.4f}"}).background_gradient(subset=["generalization_gap"], cmap="Reds"))

    gap_table = gap_table.copy()
    gap_table["model_feature"] = gap_table["model"].astype(str) + " + " + gap_table["feature"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].bar(gap_table["model_feature"], gap_table["generalization_gap"], color="#c44e52")
    axes[0].set_title("Generalization Gap by Model + Feature")
    axes[0].set_ylabel("EER_2021 - EER_2019")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(gap_table))
    width = 0.38
    axes[1].bar(x - width / 2, gap_table["eer_2019"], width=width, label="EER 2019", color="#4c72b0")
    axes[1].bar(x + width / 2, gap_table["eer_2021"], width=width, label="EER 2021", color="#dd8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gap_table["model_feature"], rotation=45)
    axes[1].set_title("EER 2019 vs EER 2021")
    axes[1].set_ylabel("EER")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()

    most_stable = gap_table.loc[gap_table["generalization_gap"].idxmin()]
    most_degraded = gap_table.loc[gap_table["generalization_gap"].idxmax()]
    display(Markdown("\n".join([
        "### Nhan xet nhanh",
        f"- Mo hinh generalize tot nhat hien tai: **{most_stable['model']} + {most_stable['feature']}** (gap = {most_stable['generalization_gap']:.4f}).",
        f"- Mo hinh giam manh nhat khi chuyen sang 2021: **{most_degraded['model']} + {most_degraded['feature']}** (gap = {most_degraded['generalization_gap']:.4f}).",
        "- Neu gap lon, mo hinh co kha nang dang hoc cac artifact dac thu cua ASVspoof 2019 hon la cac mau spoof ben vung tren du lieu unseen.",
    ])))
'''))
cells.append(md("""
## Attack-wise Error Analysis
Notebook se tu tim cac file `attack_wise_eer_*.csv`. Neu co nhieu file, uu tien phan tich file tuong ung voi profile co EER 2019 tot nhat; neu khong co, chon file moi nhat.
"""))

cells.append(code(r'''
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))
attack_files = sorted(RESULTS_DIR.glob("attack_wise_eer_*.csv"))

attack_meta_rows = []
attack_frames = []
for attack_path in attack_files:
    info = parse_result_filename(attack_path, prefix="attack_wise_eer")
    df = safe_read_csv(attack_path)
    if df.empty:
        continue
    df = df.copy()
    df["source_file"] = attack_path.name
    df["profile_hint"] = info.get("tag", "unknown")
    df["dataset"] = info.get("dataset", "unknown")
    attack_frames.append(df)
    attack_meta_rows.append({"file": attack_path.name, "profile_hint": info.get("tag", "unknown"), "dataset": info.get("dataset", "unknown"), "num_rows": len(df), "mtime": attack_path.stat().st_mtime})

attack_file_summary = pd.DataFrame(attack_meta_rows)
show_df_overview("attack file summary", attack_file_summary)

selected_attack_df = pd.DataFrame()
selected_attack_label = None
if attack_frames:
    attack_all = pd.concat(attack_frames, ignore_index=True)
    if not results_2019.empty and "profile" in results_2019.columns:
        best_profile = results_2019.sort_values("eer", ascending=True).iloc[0]["profile"]
        preferred = attack_all[attack_all["profile_hint"] == str(best_profile)]
        if not preferred.empty:
            selected_attack_df = preferred.copy()
            selected_attack_label = f"profile={best_profile}"
    if selected_attack_df.empty:
        latest_file = max(attack_files, key=lambda path: path.stat().st_mtime)
        selected_attack_df = attack_all[attack_all["source_file"] == latest_file.name].copy()
        selected_attack_label = f"latest={latest_file.name}"
    if not selected_attack_df.empty and not results_2019.empty:
        selected_attack_df = selected_attack_df.merge(results_2019[["profile", "model", "feature"]].rename(columns={"profile": "profile_hint"}), on="profile_hint", how="left")
    print(f"Tap attack-wise duoc chon: {selected_attack_label}")
    display(selected_attack_df.head())
else:
    print("Chua co file attack_wise_eer_*.csv. Hay chay evaluate voi --per_attack de bo sung phan nay.")
'''))

cells.append(code(r'''
attack_files = sorted(RESULTS_DIR.glob("attack_wise_eer_*.csv"))
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))

selected_attack_df = pd.DataFrame()
if attack_files:
    attack_frames = []
    for attack_path in attack_files:
        info = parse_result_filename(attack_path, prefix="attack_wise_eer")
        df = safe_read_csv(attack_path)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = attack_path.name
        df["profile_hint"] = info.get("tag", "unknown")
        attack_frames.append(df)
    if attack_frames:
        attack_all = pd.concat(attack_frames, ignore_index=True)
        if not results_2019.empty:
            best_profile = results_2019.sort_values("eer", ascending=True).iloc[0]["profile"]
            selected_attack_df = attack_all[attack_all["profile_hint"] == str(best_profile)].copy()
        if selected_attack_df.empty:
            latest_file = max(attack_files, key=lambda path: path.stat().st_mtime)
            selected_attack_df = attack_all[attack_all["source_file"] == latest_file.name].copy()

if selected_attack_df.empty:
    print("Khong co du lieu attack-wise hop le de phan tich.")
else:
    attack_df = selected_attack_df.copy()
    if "valid" in attack_df.columns:
        attack_df = attack_df[safe_bool_series(attack_df["valid"])].copy()
    if attack_df.empty:
        print("Tat ca dong attack-wise hien tai deu invalid hoac rong.")
    else:
        attack_df["eer"] = pd.to_numeric(attack_df["eer"], errors="coerce")
        attack_df = attack_df.dropna(subset=["eer"]).sort_values(by="eer", ascending=False).reset_index(drop=True)
        display(attack_df.style.format({"eer": "{:.4f}", "eer_threshold": "{:.4f}"}))

        plt.figure(figsize=(10, max(4, 0.45 * len(attack_df))))
        plt.barh(attack_df["attack_id"], attack_df["eer"], color="#c44e52")
        plt.gca().invert_yaxis()
        plt.xlabel("EER")
        plt.ylabel("Attack ID")
        plt.title("Attack-wise EER (sorted from hardest to easiest)")
        plt.grid(axis="x", alpha=0.25)
        plt.show()

        hardest = attack_df.head(5)
        easiest = attack_df.tail(5).sort_values(by="eer", ascending=True)
        print("\nTop 5 attack kho nhat")
        display(hardest[[col for col in ["attack_id", "eer", "num_spoof", "num_samples", "model", "feature"] if col in hardest.columns]])
        print("\nTop 5 attack de nhat")
        display(easiest[[col for col in ["attack_id", "eer", "num_spoof", "num_samples", "model", "feature"] if col in easiest.columns]])
'''))

cells.append(md("""
## Score Distribution Analysis
Phan nay doc `predictions_*.csv` de xem score cua bonafide va spoof co tach biet ro khong. Neu score hai lop con chong lan nhieu, mo hinh van de nham lan o cac mau kho.
"""))

cells.append(code(r'''
prediction_files = sorted(RESULTS_DIR.glob("predictions_*.csv"))
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))

prediction_catalog_rows = []
for pred_path in prediction_files:
    info = parse_result_filename(pred_path, prefix="predictions")
    prediction_catalog_rows.append({"file": pred_path.name, "profile_hint": info.get("tag", "unknown"), "dataset": info.get("dataset", "unknown"), "mtime": pred_path.stat().st_mtime})
prediction_catalog = pd.DataFrame(prediction_catalog_rows)
show_df_overview("prediction file catalog", prediction_catalog)

selected_prediction_path = None
if not results_2019.empty:
    best_profile = results_2019.sort_values("eer", ascending=True).iloc[0]["profile"]
    candidate = RESULTS_DIR / f"predictions_{best_profile}_2019.csv"
    if candidate.exists():
        selected_prediction_path = candidate
if selected_prediction_path is None:
    eval_2019_files = [path for path in prediction_files if path.stem.endswith("_2019")]
    if eval_2019_files:
        selected_prediction_path = max(eval_2019_files, key=lambda path: path.stat().st_mtime)
    elif prediction_files:
        selected_prediction_path = max(prediction_files, key=lambda path: path.stat().st_mtime)

if selected_prediction_path is None:
    print("Chua co file predictions_*.csv de phan tich score distribution.")
else:
    prediction_df = safe_read_csv(selected_prediction_path)
    print(f"Dang phan tich prediction file: {selected_prediction_path.name}")
    show_df_overview("selected prediction dataframe", prediction_df)
'''))

cells.append(code(r'''
prediction_files = sorted(RESULTS_DIR.glob("predictions_*.csv"))
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))
selected_prediction_path = None

if not results_2019.empty:
    best_profile = results_2019.sort_values("eer", ascending=True).iloc[0]["profile"]
    candidate = RESULTS_DIR / f"predictions_{best_profile}_2019.csv"
    if candidate.exists():
        selected_prediction_path = candidate
if selected_prediction_path is None:
    eval_2019_files = [path for path in prediction_files if path.stem.endswith("_2019")]
    if eval_2019_files:
        selected_prediction_path = max(eval_2019_files, key=lambda path: path.stat().st_mtime)
    elif prediction_files:
        selected_prediction_path = max(prediction_files, key=lambda path: path.stat().st_mtime)

if selected_prediction_path is None:
    print("Khong co prediction file hop le.")
else:
    prediction_df = safe_read_csv(selected_prediction_path)
    required_columns = {"label", "score_spoof", "pred_label"}
    if not required_columns.issubset(prediction_df.columns):
        print(f"Prediction file thieu cot can thiet: {sorted(required_columns - set(prediction_df.columns))}")
    else:
        prediction_df = prediction_df.copy()
        prediction_df["label"] = pd.to_numeric(prediction_df["label"], errors="coerce")
        prediction_df["score_spoof"] = pd.to_numeric(prediction_df["score_spoof"], errors="coerce")
        prediction_df = prediction_df.dropna(subset=["label", "score_spoof"])

        bonafide_scores = prediction_df.loc[prediction_df["label"] == 0, "score_spoof"].to_numpy()
        spoof_scores = prediction_df.loc[prediction_df["label"] == 1, "score_spoof"].to_numpy()
        if bonafide_scores.size == 0 or spoof_scores.size == 0:
            print("Khong du du lieu bonafide/spoof de ve phan phoi diem.")
        else:
            plt.figure(figsize=(9, 5))
            bins = np.linspace(0.0, 1.0, 31)
            plt.hist(bonafide_scores, bins=bins, alpha=0.70, density=True, label="bonafide", color="#4c72b0")
            plt.hist(spoof_scores, bins=bins, alpha=0.60, density=True, label="spoof", color="#c44e52")
            plt.xlabel("Spoof score")
            plt.ylabel("Density")
            plt.title(f"Score distribution - {selected_prediction_path.name}")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.show()

            hist_bona, bin_edges = np.histogram(bonafide_scores, bins=bins, density=True)
            hist_spoof, _ = np.histogram(spoof_scores, bins=bins, density=True)
            score_overlap = np.minimum(hist_bona, hist_spoof).sum() * np.diff(bin_edges).mean()
            print(f"Uoc luong muc do chong lan giua 2 phan phoi: {score_overlap:.4f}")

            if "attack_id" in prediction_df.columns:
                spoof_only = prediction_df[prediction_df["label"] == 1].copy()
                attack_stats = spoof_only.groupby("attack_id", dropna=False).agg(mean_score=("score_spoof", "mean"), median_score=("score_spoof", "median"), num_samples=("score_spoof", "size")).reset_index()
                bonafide_mean = float(bonafide_scores.mean())
                attack_stats["distance_to_bonafide_mean"] = (attack_stats["mean_score"] - bonafide_mean).abs()
                print("\nTop attack co score gan bonafide nhat")
                display(attack_stats.sort_values(by="distance_to_bonafide_mean", ascending=True).head(5))
                print("\nTop attack de tach nhat theo score spoof trung binh")
                display(attack_stats.sort_values(by="mean_score", ascending=False).head(5))
            else:
                print("Prediction file hien tai chua co cot attack_id, bo qua phan tich score theo attack.")
'''))
cells.append(md("""
## Confusion Matrix / Figure Viewer
Notebook se tu tim va hien thi cac hinh da duoc tao trong `outputs/figures/`, bao gom confusion matrix, ROC curve, histogram score va attack-wise EER neu co.
"""))

cells.append(code(r'''
figure_patterns = [
    "*_confusion_matrix_2019.png",
    "*_confusion_matrix_2021.png",
    "*_roc_2019.png",
    "*_roc_2021.png",
    "*_score_hist_2019.png",
    "*_score_hist_2021.png",
    "*_attack_wise_eer_2019.png",
]

available = []
for pattern in figure_patterns:
    available.extend(sorted(FIGURES_DIR.glob(pattern)))

if not available:
    print("Chua co figure phu hop trong outputs/figures/. Notebook van hoat dong, ban co the sinh them bang evaluate.py --save_figures.")
else:
    print(f"Tim thay {len(available)} figure de hien thi.")
    for figure_path in available:
        display_if_exists(figure_path, title=figure_path.name, figsize=(9, 5))
'''))

cells.append(md("""
## Auto-generated Textual Insights
Phan duoi day tu dong rut ra mot so nhan xet tu du lieu hien co. Neu thieu file, notebook se ghi ro rang du lieu nao chua san sang.
"""))

cells.append(code(r'''
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))
results_2021 = standardize_results_2021(safe_read_csv(RESULTS_DIR / "results_2021.csv"), df_2019=results_2019)
eer_comparison = standardize_eer_comparison(safe_read_csv(RESULTS_DIR / "eer_comparison.csv"), df_2019=results_2019, df_2021=results_2021)
attack_files = sorted(RESULTS_DIR.glob("attack_wise_eer_*.csv"))
prediction_files = sorted(RESULTS_DIR.glob("predictions_*.csv"))

insights = []
if results_2019.empty:
    insights.append("Chua du du lieu de ket luan ve ket qua ASVspoof 2019.")
else:
    best_2019 = results_2019.sort_values(by="eer", ascending=True).iloc[0]
    insights.append(f"To hop dat EER thap nhat tren ASVspoof 2019 la {best_2019['model']} + {best_2019['feature']} (profile={best_2019['profile']}), EER={best_2019['eer']:.4f}.")
    if "accuracy" in results_2019.columns and results_2019["accuracy"].notna().any():
        best_acc = results_2019.sort_values(by="accuracy", ascending=False).iloc[0]
        insights.append(f"To hop dat accuracy cao nhat la {best_acc['model']} + {best_acc['feature']} voi accuracy={best_acc['accuracy']:.4f}.")
    feature_rank = results_2019.groupby("feature")["eer"].mean().sort_values()
    if not feature_rank.empty:
        insights.append(f"Theo EER trung binh, dac trung dung dau hien tai la {feature_rank.index[0]}.")

if eer_comparison.empty:
    insights.append("Chua du du lieu de ket luan ve generalization gap 2019 -> 2021.")
else:
    most_stable = eer_comparison.sort_values(by="generalization_gap", ascending=True).iloc[0]
    most_degraded = eer_comparison.sort_values(by="generalization_gap", ascending=False).iloc[0]
    insights.append(f"Mo hinh generalize tot nhat hien tai la {most_stable['model']} + {most_stable['feature']} voi generalization gap={most_stable['generalization_gap']:.4f}.")
    insights.append(f"Mo hinh bi giam manh nhat tren 2021 la {most_degraded['model']} + {most_degraded['feature']} voi gap={most_degraded['generalization_gap']:.4f}.")
    insights.append("Neu gap lon, dieu nay goi y mo hinh co the dang hoc dataset-specific artifacts thay vi dac trung spoof ben vung.")

if attack_files:
    frames = []
    for attack_path in attack_files:
        df = safe_read_csv(attack_path)
        if df.empty:
            continue
        if "valid" in df.columns:
            df = df[safe_bool_series(df["valid"])].copy()
        if not df.empty:
            df["eer"] = pd.to_numeric(df["eer"], errors="coerce")
            df = df.dropna(subset=["eer"])
            frames.append(df)
    if frames:
        attack_all = pd.concat(frames, ignore_index=True).sort_values(by="eer", ascending=False)
        if not attack_all.empty:
            hardest = attack_all.iloc[0]
            easiest = attack_all.sort_values(by="eer", ascending=True).iloc[0]
            insights.append(f"Attack kho nhat hien tai la {hardest['attack_id']} voi EER={hardest['eer']:.4f}.")
            insights.append(f"Attack de nhat hien tai la {easiest['attack_id']} voi EER={easiest['eer']:.4f}.")
else:
    insights.append("Chua co attack-wise EER, hay chay evaluate.py voi --per_attack neu muon phan tich attack chi tiet.")

if prediction_files:
    chosen_prediction = prediction_files[0]
    prediction_df = safe_read_csv(chosen_prediction)
    if not prediction_df.empty and {"label", "score_spoof"}.issubset(prediction_df.columns):
        prediction_df["label"] = pd.to_numeric(prediction_df["label"], errors="coerce")
        prediction_df["score_spoof"] = pd.to_numeric(prediction_df["score_spoof"], errors="coerce")
        prediction_df = prediction_df.dropna(subset=["label", "score_spoof"])
        bona = prediction_df.loc[prediction_df["label"] == 0, "score_spoof"].to_numpy()
        spoof = prediction_df.loc[prediction_df["label"] == 1, "score_spoof"].to_numpy()
        if bona.size > 0 and spoof.size > 0:
            bins = np.linspace(0.0, 1.0, 31)
            hist_bona, bin_edges = np.histogram(bona, bins=bins, density=True)
            hist_spoof, _ = np.histogram(spoof, bins=bins, density=True)
            overlap = np.minimum(hist_bona, hist_spoof).sum() * np.diff(bin_edges).mean()
            level = "lon" if overlap >= 0.35 else "trung binh" if overlap >= 0.20 else "nho"
            insights.append(f"Muc do chong lan score giua bonafide va spoof duoc uoc luong la {overlap:.4f} ({level}).")
else:
    insights.append("Chua co prediction CSV de phan tich score distribution.")

if insights:
    display(Markdown("### Tong hop nhan xet\n" + "\n".join(["- " + item for item in insights])))
else:
    print("Chua co insight nao do du lieu dau vao dang trong.")
'''))

cells.append(md("""
## Export-ready Tables for Report
Ba bang duoi day duoc lam sach va lam tron de de copy sang Word / Google Docs cho phan bao cao cuoi ky.
"""))

cells.append(code(r'''
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))
results_2021 = standardize_results_2021(safe_read_csv(RESULTS_DIR / "results_2021.csv"), df_2019=results_2019)
eer_comparison = standardize_eer_comparison(safe_read_csv(RESULTS_DIR / "eer_comparison.csv"), df_2019=results_2019, df_2021=results_2021)
attack_files = sorted(RESULTS_DIR.glob("attack_wise_eer_*.csv"))

if not results_2019.empty:
    main_cols = [col for col in ["profile", "model", "feature", "accuracy", "precision", "recall", "f1", "eer"] if col in results_2019.columns]
    main_report_table = results_2019[main_cols].sort_values(by=["eer", "accuracy"], ascending=[True, False]).reset_index(drop=True)
    num_cols = main_report_table.select_dtypes(include=[np.number]).columns
    main_report_table[num_cols] = main_report_table[num_cols].round(4)
    print("Bang ket qua chinh 2019")
    display(main_report_table)
else:
    print("Chua co bang ket qua 2019.")

if not eer_comparison.empty:
    generalization_report_table = eer_comparison.sort_values(by="generalization_gap", ascending=True).reset_index(drop=True)
    num_cols = generalization_report_table.select_dtypes(include=[np.number]).columns
    generalization_report_table[num_cols] = generalization_report_table[num_cols].round(4)
    print("\nBang generalization 2019 vs 2021")
    display(generalization_report_table)
else:
    print("Chua co bang generalization.")

if attack_files:
    frames = []
    for attack_path in attack_files:
        df = safe_read_csv(attack_path)
        if df.empty:
            continue
        if "valid" in df.columns:
            df = df[safe_bool_series(df["valid"])].copy()
        if not df.empty:
            frames.append(df)
    if frames:
        selected_attack = pd.concat(frames, ignore_index=True)
        selected_attack["eer"] = pd.to_numeric(selected_attack["eer"], errors="coerce")
        selected_attack = selected_attack.dropna(subset=["eer"]).sort_values(by="eer", ascending=False).reset_index(drop=True)
        attack_report_table = selected_attack.head(5)[[col for col in ["attack_id", "eer", "num_spoof", "num_samples"] if col in selected_attack.columns]].copy()
        num_cols = attack_report_table.select_dtypes(include=[np.number]).columns
        attack_report_table[num_cols] = attack_report_table[num_cols].round(4)
        print("\nBang top attack kho nhat")
        display(attack_report_table)
    else:
        print("Khong co bang attack-wise hop le.")
else:
    print("Chua co file attack-wise EER.")
'''))

cells.append(code(r'''
results_2019 = standardize_results_2019(safe_read_csv(RESULTS_DIR / "results_2019.csv"))
eer_comparison = standardize_eer_comparison(safe_read_csv(RESULTS_DIR / "eer_comparison.csv"), df_2019=results_2019)
attack_files = sorted(RESULTS_DIR.glob("attack_wise_eer_*.csv"))

summary_lines = ["## Data-driven Conclusion"]
if not results_2019.empty:
    best_2019 = results_2019.sort_values(by="eer", ascending=True).iloc[0]
    summary_lines.append(f"- Mo hinh / dac trung tot nhat tren ASVspoof 2019 hien tai la **{best_2019['model']} + {best_2019['feature']}** (profile={best_2019['profile']}), voi **EER = {best_2019['eer']:.4f}**.")
else:
    summary_lines.append("- Chua du du lieu de ket luan mo hinh / dac trung tot nhat tren ASVspoof 2019.")

if not eer_comparison.empty:
    best_generalizer = eer_comparison.sort_values(by="generalization_gap", ascending=True).iloc[0]
    worst_generalizer = eer_comparison.sort_values(by="generalization_gap", ascending=False).iloc[0]
    summary_lines.append(f"- Mo hinh generalize tot nhat hien tai la **{best_generalizer['model']} + {best_generalizer['feature']}** voi **generalization gap = {best_generalizer['generalization_gap']:.4f}**.")
    summary_lines.append(f"- Mo hinh giam manh nhat tren 2021 la **{worst_generalizer['model']} + {worst_generalizer['feature']}** voi gap = {worst_generalizer['generalization_gap']:.4f}.")
else:
    summary_lines.append("- Chua du du lieu de ket luan ve generalization gap.")

hardest_attack_line = "- Chua du du lieu attack-wise de xac dinh attack kho nhat."
if attack_files:
    frames = []
    for attack_path in attack_files:
        df = safe_read_csv(attack_path)
        if df.empty:
            continue
        if "valid" in df.columns:
            df = df[safe_bool_series(df["valid"])].copy()
        if not df.empty:
            df["eer"] = pd.to_numeric(df["eer"], errors="coerce")
            df = df.dropna(subset=["eer"])
            frames.append(df)
    if frames:
        attack_all = pd.concat(frames, ignore_index=True).sort_values(by="eer", ascending=False)
        if not attack_all.empty:
            hardest = attack_all.iloc[0]
            hardest_attack_line = f"- Attack kho nhat hien tai la **{hardest['attack_id']}** voi **EER = {hardest['eer']:.4f}**."
summary_lines.append(hardest_attack_line)
summary_lines.append("- Cho phan Discussion, nen nhan manh moi lien he giua EER tren 2019, generalization gap tren 2021, va cac attack kho de phan tich xem mo hinh dang hoc dac trung spoof ben vung hay chi hoc artifact dac thu cua dataset.")

display(Markdown("\n".join(summary_lines)))
'''))

cells.append(md("""
## Conclusion
Notebook nay duoc thiet ke de phuc vu trich xuat bang, hinh va nhan xet cho bao cao cuoi ky. Khi cap nhat them ket qua moi bang `src/evaluate.py`, ban chi can rerun notebook de:

- cap nhat bang ket qua 2019 va 2021,
- cap nhat generalization gap,
- cap nhat attack-wise EER,
- cap nhat cac figure confusion matrix / ROC / score histogram,
- lay ra cac bullet nhan xet data-driven de chen vao phan Results va Discussion.

Neu mot phan hien chua co du lieu, hay sinh them artifact bang `evaluate.py --save_figures --per_attack` roi rerun notebook.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "version": "3.9",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path("notebooks/03_results_analysis.ipynb")
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"WROTE {path}")
