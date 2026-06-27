#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def assign_transition_regime(df: pd.DataFrame, stable_eps: float) -> pd.DataFrame:
    out = df.copy()
    growth = out["relative_growth_rate"].astype(float)
    out["transition_regime"] = "stable"
    out.loc[growth > stable_eps, "transition_regime"] = "growing"
    out.loc[growth < -stable_eps, "transition_regime"] = "shrinking"
    return out


def summarize_numeric(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols)[value_cols]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    out.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        for col in out.columns.to_flat_index()
    ]
    return out


def summarize_regimes(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = (
        df.groupby(group_cols + ["transition_regime"])
        .size()
        .reset_index(name="count")
        .sort_values(group_cols + ["transition_regime"])
    )
    counts["fraction"] = counts["count"] / counts.groupby(group_cols)["count"].transform("sum")
    return counts


def save_regime_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    order = ["shrinking", "stable", "growing"]
    pivot = (
        df.pivot_table(index="group_label", columns="transition_regime", values="fraction", aggfunc="mean")
        .fillna(0.0)
    )
    for col in order:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = None
    colors = {"shrinking": "#CC79A7", "stable": "#999999", "growing": "#D55E00"}
    for col in order:
        vals = pivot[col].values
        ax.bar(pivot.index, vals, bottom=bottom, label=col, color=colors[col])
        bottom = vals if bottom is None else bottom + vals
    ax.set_ylabel("Fraction of transitions")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    patient_summary: pd.DataFrame,
    session_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Real-Data Bridge Report\n\n")
        f.write("## Patient-Level Summary\n\n")
        f.write(patient_summary.to_markdown(index=False))
        f.write("\n\n## Session-Level Summary\n\n")
        f.write(session_summary.to_markdown(index=False))
        f.write("\n\n## Transition-Level Summary\n\n")
        f.write(transition_summary.to_markdown(index=False))
        f.write("\n\n## Transition Regime Summary\n\n")
        f.write(regime_summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge synthetic and real longitudinal regime structure using audit outputs.")
    parser.add_argument("--synthetic_audit_root", type=str, required=True)
    parser.add_argument("--real_audit_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--synthetic_group", type=str, default="C", help="Synthetic tier to compare most directly to real data.")
    parser.add_argument("--stable_eps", type=float, default=0.10, help="Threshold for labeling stable transition growth.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    syn_pat = pd.read_csv(Path(args.synthetic_audit_root) / "patients.csv")
    syn_sess = pd.read_csv(Path(args.synthetic_audit_root) / "sessions.csv")
    syn_trans = pd.read_csv(Path(args.synthetic_audit_root) / "transitions.csv")

    real_pat = pd.read_csv(Path(args.real_audit_root) / "patients.csv")
    real_sess = pd.read_csv(Path(args.real_audit_root) / "sessions.csv")
    real_trans = pd.read_csv(Path(args.real_audit_root) / "transitions.csv")

    syn_pat = syn_pat[syn_pat["tier"].astype(str) == str(args.synthetic_group)].copy()
    syn_sess = syn_sess[syn_sess["tier"].astype(str) == str(args.synthetic_group)].copy()
    syn_trans = syn_trans[syn_trans["tier"].astype(str) == str(args.synthetic_group)].copy()

    syn_pat["group_label"] = f"SYN-{args.synthetic_group}"
    syn_sess["group_label"] = f"SYN-{args.synthetic_group}"
    syn_trans["group_label"] = f"SYN-{args.synthetic_group}"
    real_pat["group_label"] = "REAL"
    real_sess["group_label"] = "REAL"
    real_trans["group_label"] = "REAL"

    both_pat = pd.concat([syn_pat, real_pat], ignore_index=True)
    both_sess = pd.concat([syn_sess, real_sess], ignore_index=True)
    both_trans = pd.concat([syn_trans, real_trans], ignore_index=True)
    both_trans = assign_transition_regime(both_trans, stable_eps=args.stable_eps)

    patient_summary = summarize_numeric(
        both_pat,
        ["group_label"],
        ["n_sessions", "followup_days", "mean_interval_days", "treatment_on_any"],
    )
    session_summary = summarize_numeric(
        both_sess,
        ["group_label"],
        ["volume_vox", "elongation_ratio", "compactness_proxy"],
    )
    transition_summary = summarize_numeric(
        both_trans,
        ["group_label"],
        ["delta_days", "delta_volume_vox", "relative_growth_rate"],
    )
    regime_summary = summarize_regimes(both_trans, ["group_label"])

    patient_summary.to_csv(out_dir / "bridge_patient_summary.csv", index=False)
    session_summary.to_csv(out_dir / "bridge_session_summary.csv", index=False)
    transition_summary.to_csv(out_dir / "bridge_transition_summary.csv", index=False)
    regime_summary.to_csv(out_dir / "bridge_transition_regime_summary.csv", index=False)
    write_report(out_dir / "real_bridge_report.md", patient_summary, session_summary, transition_summary, regime_summary)

    save_regime_plot(regime_summary, out_dir / "transition_regime_comparison.png", f"Transition Regime Comparison: SYN-{args.synthetic_group} vs REAL")

    print(f"Saved real-data bridge analysis to: {out_dir}")


if __name__ == "__main__":
    main()
