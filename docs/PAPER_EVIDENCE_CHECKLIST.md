# Paper Evidence Checklist

## Purpose

This file is the practical companion to the paper draft.

It answers:

- which tables we should include,
- which figures we should include,
- which runs they come from,
- and which evidence is core versus optional.

## Core narrative

The current paper narrative should be built around:

1. final generalized synthetic benchmark (`v3-lite`)
2. regime dependence
3. cross-regime transfer
4. case-level driver analysis
5. corrected `SAILOR` sanity validation

## Core tables

### Table 1: Final compact synthetic benchmark

Purpose:

- establish the main model ranking on the final synthetic benchmark

Content:

- `LOCF`
- `UNet-image+mask`
- `ResUNet-image+mask`
- overall mean Dice
- standard deviation
- sample count

Source:

- `fixed_dataset_v3_lite_generalized`
- run family: `v3lite_compact_h123_s42`

Status:

- ready from recorded results

### Table 2: Horizon-wise synthetic benchmark breakdown

Purpose:

- show how performance changes from horizon `1` to `3`

Content:

- rows: methods
- columns: horizons `1`, `2`, `3`
- mean Dice and optionally std

Source:

- `fixed_dataset_v3_lite_generalized`
- run family: `v3lite_compact_h123_s42`

Status:

- ready from recorded results

### Table 3: Tier-wise synthetic benchmark breakdown

Purpose:

- show regime dependence directly

Content:

- rows: methods
- columns: tiers `A`, `B`, `C`
- mean Dice and optionally std

Source:

- `fixed_dataset_v3_lite_generalized`
- run family: `v3lite_compact_h123_s42`

Status:

- ready from recorded results

### Table 4: Cross-regime transfer matrix

Purpose:

- show that tiers behave as distinct regimes rather than a single difficulty axis

Content:

- train tier vs eval tier matrix
- at minimum for `ResUNet`
- optional `LOCF` comparison in caption or supplement

Source:

- `biophys1`
- `cross_regime_transfer_overall.csv`
- `cross_regime_transfer_locf.csv`

Status:

- ready

### Table 5: Corrected `SAILOR` sanity validation

Purpose:

- anchor the synthetic story with a restrained real-data check

Content:

- overall corrected means for:
  - `LOCF`
  - `ResUNet-mask`
  - `ResUNet-image+mask`
- horizon-wise means for `1`, `2`, `3`

Source:

- processed `SAILOR`
- corrected rerun only

Status:

- ready from recorded results

## Core figures

### Figure 1: Dice by horizon across methods

Purpose:

- visualize horizon-wise behavior on the synthetic benchmark

Preferred source artifact:

- `dice_by_horizon_all_methods.png`

Likely generated from:

- `scripts/export_regime_figures.py`

Best source directory:

- the regime-analysis report directory corresponding to the final synthetic benchmark narrative

Status:

- likely available already for earlier benchmark variants
- may need regeneration for the exact final benchmark if not already saved in that form

### Figure 2: Win rate by tier (`ResUNet` vs `LOCF`)

Purpose:

- make regime dependence immediately visible

Preferred source artifact:

- `win_rate_by_tier.png`

Source:

- regime-analysis export

Status:

- available for prior benchmark analysis
- acceptable if clearly labeled as `biophys1` evidence rather than `v3-lite`

### Figure 3: Win rate by future growth bin

Purpose:

- show that learned gains increase with stronger future growth

Preferred source artifact:

- `win_rate_by_future_growth_bin.png`

Source:

- regime-analysis export

Status:

- available

### Figure 4: Cross-regime transfer heatmap

Purpose:

- visually summarize transfer asymmetry

Preferred source artifact:

- transfer heatmap exported from `scripts/export_transfer_figures.py`

Source:

- `cross_regime_transfer_overall.csv`

Status:

- available or straightforward to regenerate

## Optional figures

### Optional Figure A: Win rate by recent growth bin

Why optional:

- useful, but slightly less central than future-growth or tier views

Artifact:

- `win_rate_by_recent_growth_bin.png`

### Optional Figure B: Win rate by input volume bin

Why optional:

- helps the case-analysis story
- may fit better in supplement if space is tight

Artifact:

- `win_rate_by_input_volume_bin.png`

### Optional Figure C: Real-bridge comparison

Why optional:

- useful for benchmark-design discussion
- not essential to the main forecasting results section

Source:

- `multitier_real_bridge_v3lite`
- `real_bridge_report.md`

## Supplementary evidence

These are good to keep, but not necessary in the main paper body:

1. `UNETR` compact comparison
2. plain CNN comparison
3. early benchmark variants (`v1`, medium, `biophys1`) except where needed for transfer or driver analysis
4. seed sweeps beyond the minimal robustness check
5. detailed case-type tables

## Recommended main-text package

If space is limited, the strongest compact package is:

- Table 1: final compact synthetic benchmark
- Table 2: tier-wise synthetic breakdown
- Table 3: corrected `SAILOR` horizon-wise sanity table
- Figure 1: Dice by horizon
- Figure 2: win rate by tier
- Figure 3: cross-regime transfer heatmap

## Recommended supplement package

- full horizon-wise synthetic table
- full case-type summaries
- win rate by growth / volume bins
- additional model comparisons
- benchmark design / real-bridge audit tables

## Caption guidance

Across all figures and tables, we should keep captions honest:

- clearly distinguish synthetic from real data
- clearly distinguish `biophys1` from final `v3-lite`
- explicitly note that `SAILOR` is a sanity validation, not a tuned benchmark endpoint

## Final caution

The main paper should not mix evidence from different benchmark generations carelessly.

Safe pattern:

- use `v3-lite` for the final synthetic benchmark story
- use `biophys1` for transfer and case-driver analysis if needed, but label it explicitly
- use corrected `SAILOR` only as a constrained external check
