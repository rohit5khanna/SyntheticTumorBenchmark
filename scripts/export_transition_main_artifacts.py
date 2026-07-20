#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required table: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def get_value(df: pd.DataFrame, dataset: str, col: str) -> float:
    row = df[df["dataset"].astype(str).eq(dataset)]
    if row.empty or col not in row:
        return float("nan")
    return float(row.iloc[0][col])


def region_value(df: pd.DataFrame, region: str, col: str) -> float:
    row = df[df["region"].astype(str).eq(region)]
    if row.empty or col not in row:
        return float("nan")
    return float(row.iloc[0][col])


def patient_spread_text(patient_spread: pd.DataFrame, dataset: str, region: str) -> str:
    row = patient_spread[
        patient_spread["dataset"].astype(str).eq(dataset) & patient_spread["region"].astype(str).eq(region)
    ]
    if row.empty:
        return ""
    r = row.iloc[0]
    n = int(r["n"])
    n_patients = int(r["n_patients"])
    if n == 0 or n_patients == 0:
        return "no transitions"
    transition_word = "transition" if n == 1 else "transitions"
    patient_word = "patient" if n_patients == 1 else "patients"
    return f"{n} {transition_word} across {n_patients} {patient_word}; largest patient share {float(r['max_patient_fraction']):.2f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    try:
        x = float(value) * 100.0
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}%"


def build_compact_table(
    evidence_dir: Path,
    robustness_dir: Path,
    output_dir: Path,
    sailor_name: str,
    srd_name: str,
) -> pd.DataFrame:
    table_dir = evidence_dir / "core_tables"
    overall = read_csv(table_dir / "dataset_overall_comparison.csv")
    transitions = read_csv(table_dir / "transition_type_distribution.csv")
    robust = read_csv(robustness_dir / "transition_label_robust_region_summary.csv")
    spread = read_csv(robustness_dir / "transition_label_default_patient_spread.csv")

    def transition_fraction(dataset: str, transition_type: str) -> float:
        row = transitions[
            transitions["dataset"].astype(str).eq(dataset)
            & transitions["transition_type"].astype(str).eq(transition_type)
        ]
        return float(row.iloc[0]["fraction"]) if not row.empty else 0.0

    rows = [
        {
            "metric": "Transitions / patients",
            sailor_name: f"{int(get_value(overall, sailor_name, 'n_transitions'))} / {int(get_value(overall, sailor_name, 'n_patients'))}",
            srd_name: f"{int(get_value(overall, srd_name, 'n_transitions'))} / {int(get_value(overall, srd_name, 'n_patients'))}",
            "main_message": "Transition-level analysis with patient spread checked for hard SAILOR regions.",
        },
        {
            "metric": "Mean LOCF Dice",
            sailor_name: fmt_float(get_value(overall, sailor_name, "mean_locf_dice")),
            srd_name: fmt_float(get_value(overall, srd_name, "mean_locf_dice")),
            "main_message": "Persistence is stronger on SRD, but this should be read as operating-range context rather than a leaderboard.",
        },
        {
            "metric": "Mean relative absolute change",
            sailor_name: fmt_float(get_value(overall, sailor_name, "mean_relative_absolute_change")),
            srd_name: fmt_float(get_value(overall, srd_name, "mean_relative_absolute_change")),
            "main_message": "SAILOR has substantially higher observed transition burden.",
        },
        {
            "metric": "Mixed growth/loss transitions",
            sailor_name: fmt_pct(transition_fraction(sailor_name, "mixed_growth_loss")),
            srd_name: fmt_pct(transition_fraction(srd_name, "mixed_growth_loss")),
            "main_message": (
                f"SAILOR > SRD in {fmt_pct(region_value(robust, 'mixed_growth_loss', 'fraction_gap_positive_rate'), 0)} "
                f"of threshold settings; {patient_spread_text(spread, sailor_name, 'mixed_growth_loss')}"
            ),
        },
        {
            "metric": "Distant/non-boundary growth",
            sailor_name: fmt_pct(get_value(overall, sailor_name, "distant_growth_rate")),
            srd_name: fmt_pct(get_value(overall, srd_name, "distant_growth_rate")),
            "main_message": (
                f"SAILOR > SRD in {fmt_pct(region_value(robust, 'distant_growth_present', 'fraction_gap_positive_rate'), 0)} "
                f"of threshold settings; {patient_spread_text(spread, sailor_name, 'distant_growth_present')}"
            ),
        },
        {
            "metric": "Persistence-dominant transitions",
            sailor_name: fmt_pct(transition_fraction(sailor_name, "persistence_dominant")),
            srd_name: fmt_pct(transition_fraction(srd_name, "persistence_dominant")),
            "main_message": "SRD remains more persistence-friendly across tested thresholds.",
        },
        {
            "metric": "High-change transitions",
            sailor_name: fmt_pct(region_value(spread[spread["dataset"].astype(str).eq(sailor_name)], "high_absolute_change", "fraction")),
            srd_name: fmt_pct(region_value(spread[spread["dataset"].astype(str).eq(srd_name)], "high_absolute_change", "fraction")),
            "main_message": (
                "Defined as relative absolute change >= 2.0; "
                f"{patient_spread_text(spread, sailor_name, 'high_absolute_change')}"
            ),
        },
    ]

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "main_transition_summary_table.csv", index=False)
    with (output_dir / "main_transition_summary_table.md").open("w", encoding="utf-8") as f:
        f.write(out.to_markdown(index=False))
        f.write("\n")
    return out


def ordered_transition_types(transitions: pd.DataFrame) -> List[str]:
    preferred = [
        "mixed_growth_loss",
        "growth_dominant",
        "loss_dominant",
        "boundary_growth_dominant",
        "persistence_dominant",
        "distant_growth_present",
        "moderate_mixed_change",
    ]
    existing = list(dict.fromkeys(transitions["transition_type"].astype(str)))
    return [x for x in preferred if x in existing] + [x for x in existing if x not in preferred]


def write_figure(evidence_dir: Path, output_dir: Path, sailor_name: str, srd_name: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transitions = read_csv(evidence_dir / "core_tables" / "transition_type_distribution.csv")
    order = ordered_transition_types(transitions)
    datasets = [sailor_name, srd_name]
    colors = {sailor_name: "#2E7D32", srd_name: "#546E7A"}

    def vals(col: str, dataset: str) -> List[float]:
        out = []
        for transition_type in order:
            row = transitions[
                transitions["dataset"].astype(str).eq(dataset)
                & transitions["transition_type"].astype(str).eq(transition_type)
            ]
            out.append(float(row.iloc[0][col]) if not row.empty else 0.0)
        return out

    labels = [x.replace("_", "\n") for x in order]
    x = np.arange(len(order))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharex=True)

    for i, dataset in enumerate(datasets):
        offset = (i - 0.5) * width
        axes[0].bar(x + offset, vals("fraction", dataset), width=width, label=dataset, color=colors[dataset], alpha=0.9)
        axes[1].bar(
            x + offset,
            vals("mean_locf_dice", dataset),
            width=width,
            label=dataset,
            color=colors[dataset],
            alpha=0.9,
        )

    axes[0].set_title("A. Transition-type distribution")
    axes[0].set_ylabel("Fraction of transitions")
    axes[0].set_ylim(0, max(0.8, float(transitions["fraction"].max()) * 1.15))
    axes[1].set_title("B. LOCF Dice by transition type")
    axes[1].set_ylabel("Mean LOCF Dice")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Transition composition and persistence difficulty", y=1.02, fontsize=13)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "main_transition_type_and_locf_figure.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_caption(output_dir: Path, table: pd.DataFrame, figure_path: str) -> None:
    caption = (
        "Figure: Transition composition and persistence difficulty. "
        "Panel A compares the fraction of transitions assigned to each descriptive transition type. "
        "Panel B reports mean LOCF Dice within the same transition types. "
        "The main visual point is that SAILOR contains mixed growth/loss transitions absent from SRD, "
        "whereas SRD more cleanly separates persistence, growth, and loss axes. "
        "The full robustness audit tests threshold sensitivity and patient concentration."
    )
    with (output_dir / "main_transition_artifacts_notes.md").open("w", encoding="utf-8") as f:
        f.write("# Main Transition Artifacts\n\n")
        f.write("## Recommended Main-Paper Use\n\n")
        f.write("Use one compact table and one two-panel figure for the transition-evidence part.\n\n")
        f.write("## Table\n\n")
        f.write("`main_transition_summary_table.csv` / `main_transition_summary_table.md`\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n\n")
        f.write("Table note: LOCF = last observation carried forward. Relative absolute change is `(new-growth volume + apparent-loss volume) / input tumor volume`. Categorical rows report transition fractions. Core-loss and loss-dominant descriptors are retained as audit-level secondary evidence because their threshold robustness was weaker.\n\n")
        f.write("## Figure\n\n")
        f.write(f"`{figure_path}`\n\n")
        f.write("Suggested caption:\n\n")
        f.write(caption)
        f.write("\n\n")
        f.write("## Writing Note\n\n")
        f.write(
            "In the main text, avoid listing every robustness table. Instead state that mixed growth/loss and "
            "distant-growth gaps survived all tested threshold settings, while core-loss/loss-dominant descriptors "
            "were less stable and are treated as secondary.\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact main-paper transition evidence artifacts.")
    parser.add_argument("--evidence_package_dir", type=str, required=True)
    parser.add_argument("--robustness_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sailor_name", type=str, default="SAILOR")
    parser.add_argument("--srd_name", type=str, default="SRD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence_dir = Path(args.evidence_package_dir)
    robustness_dir = Path(args.robustness_dir)
    output_dir = Path(args.output_dir)
    table = build_compact_table(evidence_dir, robustness_dir, output_dir, args.sailor_name, args.srd_name)
    figure_path = write_figure(evidence_dir, output_dir, args.sailor_name, args.srd_name)
    write_caption(output_dir, table, figure_path)
    payload = {
        "evidence_package_dir": str(evidence_dir),
        "robustness_dir": str(robustness_dir),
        "output_dir": str(output_dir),
        "outputs": {
            "main_transition_summary_table_csv": str(output_dir / "main_transition_summary_table.csv"),
            "main_transition_summary_table_md": str(output_dir / "main_transition_summary_table.md"),
            "main_transition_type_and_locf_figure_png": figure_path,
            "main_transition_artifacts_notes_md": str(output_dir / "main_transition_artifacts_notes.md"),
        },
    }
    with (output_dir / "main_transition_artifacts_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
