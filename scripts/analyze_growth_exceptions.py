#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEY_COLS = ["patient_id", "input_idx", "target_idx", "horizon", "delta_days"]


def load_per_sample_tables(output_dir: Path, methods: Iterable[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        path = output_dir / f"{method}_per_sample.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload:
            out = dict(row)
            out["method"] = method
            rows.append(out)
    if not rows:
        raise FileNotFoundError(f"No per-sample JSONs found in {output_dir}")
    return pd.DataFrame(rows)


def build_dice_pairwise(per_sample: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    locf = per_sample[per_sample["method"] == "locf"][KEY_COLS + ["dice"]].rename(columns={"dice": "locf_dice"})
    rows = []
    for method in methods:
        if method == "locf":
            continue
        cur = per_sample[per_sample["method"] == method][KEY_COLS + ["dice"]].rename(columns={"dice": "model_dice"})
        pair = cur.merge(locf, on=KEY_COLS, how="inner")
        pair["model"] = method
        pair["dice_gap_vs_locf"] = pair["model_dice"] - pair["locf_dice"]
        rows.append(pair)
    if not rows:
        return pd.DataFrame(columns=KEY_COLS + ["model", "model_dice", "locf_dice", "dice_gap_vs_locf"])
    return pd.concat(rows, ignore_index=True)


def build_ranking_pairwise(ranking: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    dist = ranking[ranking["method"] == "distance_to_input_mask"][
        KEY_COLS + ["growth_average_precision", "growth_recall_at_growth_volume"]
    ].rename(
        columns={
            "growth_average_precision": "distance_ap",
            "growth_recall_at_growth_volume": "distance_recall_at_growth_volume",
        }
    )
    rows = []
    for method in methods:
        if method == "locf":
            continue
        cur = ranking[ranking["method"] == method][
            KEY_COLS + ["growth_average_precision", "growth_recall_at_growth_volume"]
        ].rename(
            columns={
                "growth_average_precision": "model_ap",
                "growth_recall_at_growth_volume": "model_recall_at_growth_volume",
            }
        )
        pair = cur.merge(dist, on=KEY_COLS, how="inner")
        pair["model"] = method
        pair["ap_gap_vs_distance"] = pair["model_ap"] - pair["distance_ap"]
        pair["recall_gap_vs_distance"] = (
            pair["model_recall_at_growth_volume"] - pair["distance_recall_at_growth_volume"]
        )
        rows.append(pair)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def assign_exception_flags(
    df: pd.DataFrame,
    min_ap_gap: float,
    min_dice_gap: float,
) -> pd.DataFrame:
    out = df.copy()
    out["exception_reasons"] = ""

    reason_sets = []
    for _, row in out.iterrows():
        reasons = []
        growth_bin = str(row.get("new_growth_bin", ""))
        ap_gap = row.get("ap_gap_vs_distance")
        dice_gap = row.get("dice_gap_vs_locf")

        if growth_bin == "high" and pd.notna(ap_gap) and ap_gap < 0:
            reasons.append("high_growth_model_ranks_below_distance")
        if growth_bin == "high" and pd.notna(dice_gap) and dice_gap < 0:
            reasons.append("high_growth_model_dice_below_locf")
        if growth_bin == "low" and pd.notna(ap_gap) and ap_gap > min_ap_gap:
            reasons.append("low_growth_model_ranks_above_distance")
        if growth_bin == "low" and pd.notna(dice_gap) and dice_gap > min_dice_gap:
            reasons.append("low_growth_model_dice_above_locf")
        if pd.notna(ap_gap) and pd.notna(dice_gap) and ap_gap > min_ap_gap and dice_gap < -min_dice_gap:
            reasons.append("ranking_good_but_dice_bad")
        if pd.notna(ap_gap) and pd.notna(dice_gap) and ap_gap < -min_ap_gap and dice_gap > min_dice_gap:
            reasons.append("dice_good_but_ranking_bad")

        reason_sets.append(";".join(reasons) if reasons else "none")

    out["exception_reasons"] = reason_sets
    out["is_exception_case"] = out["exception_reasons"] != "none"
    return out


def summarize_reason_counts(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df[df["is_exception_case"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["exception_reason", "count", "fraction"])
    exploded = (
        flagged[["exception_reasons"]]
        .assign(exception_reason=lambda x: x["exception_reasons"].str.split(";"))
        .explode("exception_reason")
    )
    out = exploded.groupby("exception_reason").size().reset_index(name="count")
    out["fraction"] = out["count"] / max(1, len(flagged))
    return out.sort_values("count", ascending=False)


def summarize_groups(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    flagged = df[df["is_exception_case"]].copy()
    if flagged.empty:
        return pd.DataFrame()
    available = [c for c in group_cols if c in flagged.columns]
    return (
        flagged.groupby(available, dropna=False, observed=True)
        .agg(
            count=("is_exception_case", "size"),
            mean_ap_gap_vs_distance=("ap_gap_vs_distance", "mean"),
            mean_dice_gap_vs_locf=("dice_gap_vs_locf", "mean"),
            mean_relative_new_growth=("relative_new_growth", "mean"),
            mean_growth_volume_vox=("growth_volume_vox", "mean"),
        )
        .reset_index()
        .sort_values(available)
    )


def write_report(path: Path, reason_counts: pd.DataFrame, group_summary: pd.DataFrame, top_cases: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Growth Exception Audit\n\n")
        f.write("This audit flags cases that challenge the current growth-aware interpretation.\n\n")
        f.write("## Exception Reason Counts\n\n")
        f.write(reason_counts.to_markdown(index=False) if not reason_counts.empty else "No exception cases found.")
        f.write("\n\n## Exception Group Summary\n\n")
        f.write(group_summary.to_markdown(index=False) if not group_summary.empty else "No grouped exception summary.")
        f.write("\n\n## Top Exception Cases\n\n")
        f.write(top_cases.to_markdown(index=False) if not top_cases.empty else "No top exception cases.")
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit high-growth learned-model failures and low-growth learned-model successes."
    )
    parser.add_argument("--growth_eval_dir", type=str, required=True)
    parser.add_argument("--baseline_output_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="unet_image_mask,resunet_image_mask")
    parser.add_argument("--min_ap_gap", type=float, default=0.05)
    parser.add_argument("--min_dice_gap", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    growth_eval_dir = Path(args.growth_eval_dir)
    baseline_output_dir = Path(args.baseline_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    per_sample = load_per_sample_tables(baseline_output_dir, ["locf"] + methods)
    dice_pair = build_dice_pairwise(per_sample, methods)
    ranking = pd.read_csv(growth_eval_dir / "growth_ranking_metrics.csv")
    rank_pair = build_ranking_pairwise(ranking, methods)
    features = pd.read_csv(growth_eval_dir / "growth_sample_features.csv")

    merged = rank_pair.merge(dice_pair, on=KEY_COLS + ["model"], how="left")
    feature_cols = [
        "tier",
        "new_growth_bin",
        "abs_change_bin",
        "net_growth_bin",
        "relative_new_growth",
        "relative_abs_change",
        "growth_volume_vox",
        "input_volume_vox",
        "target_volume_vox",
        "locf_dice_from_masks",
    ]
    merged = merged.merge(features[KEY_COLS + feature_cols], on=KEY_COLS, how="left")
    audited = assign_exception_flags(merged, min_ap_gap=args.min_ap_gap, min_dice_gap=args.min_dice_gap)

    reason_counts = summarize_reason_counts(audited)
    group_summary = summarize_groups(audited, ["model", "new_growth_bin", "tier", "horizon"])
    top_cols = [
        "model",
        "patient_id",
        "input_idx",
        "target_idx",
        "horizon",
        "tier",
        "new_growth_bin",
        "growth_volume_vox",
        "relative_new_growth",
        "model_ap",
        "distance_ap",
        "ap_gap_vs_distance",
        "model_dice",
        "locf_dice",
        "dice_gap_vs_locf",
        "exception_reasons",
    ]
    top_cases = (
        audited[audited["is_exception_case"]][top_cols]
        .assign(abs_ap_gap=lambda x: x["ap_gap_vs_distance"].abs())
        .sort_values(["new_growth_bin", "abs_ap_gap"], ascending=[True, False])
        .drop(columns=["abs_ap_gap"])
        .head(40)
    )

    audited.to_csv(output_dir / "growth_exception_samples.csv", index=False)
    reason_counts.to_csv(output_dir / "growth_exception_reason_counts.csv", index=False)
    group_summary.to_csv(output_dir / "growth_exception_group_summary.csv", index=False)
    top_cases.to_csv(output_dir / "growth_exception_top_cases.csv", index=False)
    write_report(output_dir / "growth_exception_audit_report.md", reason_counts, group_summary, top_cases)

    print(
        json.dumps(
            {
                "growth_eval_dir": str(growth_eval_dir),
                "baseline_output_dir": str(baseline_output_dir),
                "methods": methods,
                "n_rows": int(len(audited)),
                "n_exception_rows": int(audited["is_exception_case"].sum()),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
