#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon"]
GROUP_CANDIDATES = [
    "split",
    "horizon",
    "transition_type",
    "net_direction",
    "absolute_growth_bin",
    "new_growth_rate_quantile",
    "absolute_change_rate_quantile",
    "delta_days_bin",
]


def parse_list(text: str | None) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def read_json_or_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return pd.DataFrame(payload)
    return pd.read_csv(path)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "patient" in out.columns and "patient_id" not in out.columns:
        out = out.rename(columns={"patient": "patient_id"})
    if "subject_id" in out.columns and "patient_id" not in out.columns:
        out = out.rename(columns={"subject_id": "patient_id"})
    for col in ["input_idx", "target_idx", "horizon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "patient_id" in out.columns:
        out["patient_id"] = out["patient_id"].astype(str)
    return out


def infer_method_name(path: Path) -> str:
    stem = path.stem
    for suffix in ["_per_sample", "_samples", "_selected_samples"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def load_model_tables(files: Iterable[str], dirs: Iterable[str], methods: Iterable[str]) -> pd.DataFrame:
    rows = []

    for raw in files:
        if ":" in raw:
            name, path_s = raw.split(":", 1)
            path = Path(path_s)
        else:
            path = Path(raw)
            name = infer_method_name(path)
        cur = read_json_or_csv(path)
        cur = normalize_keys(cur)
        cur["method"] = name
        rows.append(cur)

    for raw_dir in dirs:
        root = Path(raw_dir)
        for method in methods:
            candidates = [
                root / f"{method}_per_sample.json",
                root / f"{method}_per_sample.csv",
                root / f"{method}_selected_samples.csv",
            ]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                continue
            cur = read_json_or_csv(path)
            cur = normalize_keys(cur)
            cur["method"] = method
            rows.append(cur)

    if not rows:
        raise FileNotFoundError("No model per-sample files were found.")

    out = pd.concat(rows, ignore_index=True)
    if "dice" not in out.columns:
        dice_like = [c for c in out.columns if c.endswith("dice") or c == "mean_dice"]
        if len(dice_like) == 1:
            out = out.rename(columns={dice_like[0]: "dice"})
    if "dice" not in out.columns:
        raise ValueError("Model per-sample tables must contain a dice column or one unambiguous '*dice' column.")
    out["dice"] = pd.to_numeric(out["dice"], errors="coerce")
    out = out.dropna(subset=KEY_COLS + ["dice"])
    return out


def load_transition_samples(path: Path) -> pd.DataFrame:
    df = normalize_keys(pd.read_csv(path))
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Transition samples are missing key columns: {missing}")
    return df


def summarize(df: pd.DataFrame, group_cols: List[str], dice_col: str = "dice") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        df = df.assign(_overall="overall")
        cols = ["_overall"]

    def q25(x: pd.Series) -> float:
        return float(np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.25))

    def q75(x: pd.Series) -> float:
        return float(np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.75))

    out = (
        df.groupby(cols, dropna=False, observed=True)
        .agg(
            count=(dice_col, "size"),
            n_patients=("patient_id", "nunique"),
            mean_dice=(dice_col, "mean"),
            median_dice=(dice_col, "median"),
            q25_dice=(dice_col, q25),
            q75_dice=(dice_col, q75),
            std_dice=(dice_col, "std"),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def summarize_gap(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty or "gap_vs_locf" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        df = df.assign(_overall="overall")
        cols = ["_overall"]
    out = (
        df.groupby(cols, dropna=False, observed=True)
        .agg(
            count=("gap_vs_locf", "size"),
            n_patients=("patient_id", "nunique"),
            mean_model_dice=("dice", "mean"),
            mean_locf_dice=("locf_dice_ref", "mean"),
            mean_gap_vs_locf=("gap_vs_locf", "mean"),
            median_gap_vs_locf=("gap_vs_locf", "median"),
            win_rate_vs_locf=("gap_vs_locf", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
        )
        .reset_index()
    )
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def add_locf_reference(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    if "locf_dice" in out.columns:
        out["locf_dice_ref"] = pd.to_numeric(out["locf_dice"], errors="coerce")
    else:
        out["locf_dice_ref"] = np.nan

    locf_rows = out[out["method"].str.lower().isin(["locf", "last_observation_carried_forward"])]
    if not locf_rows.empty:
        locf_ref = locf_rows[KEY_COLS + ["dice"]].rename(columns={"dice": "locf_dice_from_model"})
        out = out.merge(locf_ref, on=KEY_COLS, how="left")
        out["locf_dice_ref"] = out["locf_dice_ref"].fillna(out["locf_dice_from_model"])
        out = out.drop(columns=["locf_dice_from_model"])
    out["gap_vs_locf"] = out["dice"] - out["locf_dice_ref"]
    return out


def plot_transition_audit(enriched: pd.DataFrame, out_dir: Path, min_count: int = 2) -> List[str]:
    import matplotlib.pyplot as plt

    outputs = []
    if "transition_type" in enriched.columns:
        part = summarize(enriched, ["method", "transition_type"])
        part = part[part["count"] >= min_count].copy()
        if not part.empty:
            order = (
                part.groupby("transition_type")["mean_dice"]
                .mean()
                .sort_values(ascending=True)
                .index.tolist()
            )
            methods = part["method"].drop_duplicates().tolist()
            fig_h = max(3.2, 0.38 * len(order) + 1.2)
            fig, ax = plt.subplots(figsize=(8.2, fig_h))
            y = np.arange(len(order))
            width = 0.72 / max(1, len(methods))
            for i, method in enumerate(methods):
                vals = []
                labels = []
                for t in order:
                    row = part[(part["method"] == method) & (part["transition_type"] == t)]
                    if row.empty:
                        vals.append(np.nan)
                        labels.append("")
                    else:
                        vals.append(float(row.iloc[0]["mean_dice"]))
                        labels.append(f"n={int(row.iloc[0]['count'])}, p={int(row.iloc[0]['n_patients'])}")
                offset = (i - (len(methods) - 1) / 2) * width
                ax.barh(y + offset, vals, height=width * 0.92, label=method)
                for yy, val, lab in zip(y + offset, vals, labels):
                    if np.isfinite(val):
                        ax.text(val + 0.01, yy, f"{val:.2f} {lab}", va="center", fontsize=7)
            ax.set_yticks(y)
            ax.set_yticklabels([x.replace("_", " ") for x in order])
            ax.set_xlim(0, min(1.02, max(0.75, float(np.nanmax(part["mean_dice"]) + 0.18))))
            ax.set_xlabel("Mean Dice")
            ax.set_title("Forecasting performance by transition type")
            ax.grid(axis="x", alpha=0.25)
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            path = out_dir / "model_dice_by_transition_type.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            outputs.append(str(path))

        gap = enriched[~enriched["method"].str.lower().eq("locf") & enriched["gap_vs_locf"].notna()].copy()
        gap_summary = summarize_gap(gap, ["method", "transition_type"])
        gap_summary = gap_summary[gap_summary["count"] >= min_count].copy()
        if not gap_summary.empty:
            order = (
                gap_summary.groupby("transition_type")["mean_gap_vs_locf"]
                .mean()
                .sort_values(ascending=True)
                .index.tolist()
            )
            methods = gap_summary["method"].drop_duplicates().tolist()
            fig_h = max(3.2, 0.38 * len(order) + 1.2)
            fig, ax = plt.subplots(figsize=(8.2, fig_h))
            y = np.arange(len(order))
            width = 0.72 / max(1, len(methods))
            for i, method in enumerate(methods):
                vals = []
                labels = []
                for t in order:
                    row = gap_summary[(gap_summary["method"] == method) & (gap_summary["transition_type"] == t)]
                    if row.empty:
                        vals.append(np.nan)
                        labels.append("")
                    else:
                        vals.append(float(row.iloc[0]["mean_gap_vs_locf"]))
                        labels.append(f"win={float(row.iloc[0]['win_rate_vs_locf']):.2f}")
                offset = (i - (len(methods) - 1) / 2) * width
                ax.barh(y + offset, vals, height=width * 0.92, label=method)
                for yy, val, lab in zip(y + offset, vals, labels):
                    if np.isfinite(val):
                        x_text = val + (0.004 if val >= 0 else -0.004)
                        ha = "left" if val >= 0 else "right"
                        ax.text(x_text, yy, f"{val:+.3f} {lab}", va="center", ha=ha, fontsize=7)
            ax.axvline(0, color="black", linewidth=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels([x.replace("_", " ") for x in order])
            ax.set_xlabel("Mean Dice gap versus LOCF")
            ax.set_title("Model gain/loss relative to LOCF by transition type")
            ax.grid(axis="x", alpha=0.25)
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            path = out_dir / "model_gap_vs_locf_by_transition_type.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            outputs.append(str(path))

    return outputs


def write_report(path: Path, overall: pd.DataFrame, gap_overall: pd.DataFrame, figures: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Forecasting Model Transition Audit\n\n")
        f.write("This audit joins model per-sample Dice outputs to transition-taxonomy descriptors using patient/session/horizon keys.\n\n")
        f.write("## Overall Model Performance\n\n")
        f.write(overall.to_markdown(index=False) if not overall.empty else "_No overall rows._")
        f.write("\n\n## Model Gap Versus LOCF\n\n")
        f.write(gap_overall.to_markdown(index=False) if not gap_overall.empty else "_No non-LOCF model gap rows._")
        f.write("\n\n## Figures\n\n")
        for fig in figures:
            f.write(f"- `{fig}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit forecasting model performance by transition descriptors.")
    parser.add_argument("--transition_samples_csv", type=str, required=True)
    parser.add_argument("--per_sample_files", type=str, default="", help="Comma-separated paths or name:path entries.")
    parser.add_argument("--baseline_output_dirs", type=str, default="", help="Comma-separated output dirs containing <method>_per_sample.json/csv.")
    parser.add_argument("--methods", type=str, default="locf,resunet_mask,resunet_image_mask,unet_mask,unet_image_mask")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_count_for_plot", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transition = load_transition_samples(Path(args.transition_samples_csv))
    models = load_model_tables(parse_list(args.per_sample_files), parse_list(args.baseline_output_dirs), parse_list(args.methods))
    enriched = models.merge(transition, on=KEY_COLS, how="inner", suffixes=("", "_transition"))
    if enriched.empty:
        raise ValueError("No rows remained after merging model outputs with transition samples.")
    enriched = add_locf_reference(enriched)

    overall = summarize(enriched, ["method"])
    by_groups = []
    for group in GROUP_CANDIDATES:
        if group in enriched.columns:
            cur = summarize(enriched, ["method", group])
            if not cur.empty:
                cur.insert(0, "grouping", group)
                by_groups.append(cur.rename(columns={group: "group_value"}))
    by_group = pd.concat(by_groups, ignore_index=True, sort=False) if by_groups else pd.DataFrame()

    non_locf = enriched[~enriched["method"].str.lower().eq("locf") & enriched["gap_vs_locf"].notna()].copy()
    gap_overall = summarize_gap(non_locf, ["method"])
    gap_groups = []
    for group in GROUP_CANDIDATES:
        if group in non_locf.columns:
            cur = summarize_gap(non_locf, ["method", group])
            if not cur.empty:
                cur.insert(0, "grouping", group)
                gap_groups.append(cur.rename(columns={group: "group_value"}))
    gap_by_group = pd.concat(gap_groups, ignore_index=True, sort=False) if gap_groups else pd.DataFrame()

    enriched.to_csv(output_dir / "forecasting_model_transition_audit_samples.csv", index=False)
    overall.to_csv(output_dir / "forecasting_model_transition_audit_overall.csv", index=False)
    by_group.to_csv(output_dir / "forecasting_model_transition_audit_by_group.csv", index=False)
    gap_overall.to_csv(output_dir / "forecasting_model_transition_audit_gap_overall.csv", index=False)
    gap_by_group.to_csv(output_dir / "forecasting_model_transition_audit_gap_by_group.csv", index=False)
    figures = plot_transition_audit(enriched, output_dir, min_count=args.min_count_for_plot)
    write_report(output_dir / "forecasting_model_transition_audit_report.md", overall, gap_overall, figures)

    summary = {
        "n_model_rows": int(len(models)),
        "n_transition_rows": int(len(transition)),
        "n_merged_rows": int(len(enriched)),
        "n_patients": int(enriched["patient_id"].nunique()),
        "methods": sorted(enriched["method"].astype(str).unique().tolist()),
        "output_dir": str(output_dir),
        "figures": figures,
    }
    with (output_dir / "forecasting_model_transition_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
