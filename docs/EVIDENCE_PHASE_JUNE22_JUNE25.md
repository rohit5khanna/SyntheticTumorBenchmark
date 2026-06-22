# Evidence Phase Plan: June 22 to June 25

## Goal

Use the next 3-4 days as a focused evidence-collection phase.

By the end of June 25, the project should have:

- one pilot benchmark result package,
- one medium-scale benchmark result package,
- a small set of paper-ready tables,
- a small set of paper-ready figures,
- enough evidence to spend June 26-30 primarily on writing and literature review.

## Positioning

The sprint benchmark results collected so far should be treated as **pilot evidence**.

Their purpose is to:

- validate the pipeline,
- reveal the benchmark story,
- identify what is worth scaling.

The medium-scale benchmark should be treated as the **main paper evidence**.

## Main Questions for the Evidence Phase

1. Does the benchmark story survive scaling beyond the sprint dataset?
2. Does `LOCF` remain a strong short-term baseline on a larger frozen dataset?
3. Do the difficulty regimes (`Tier A`, `Tier B`, `Tier C`) remain meaningfully separated?
4. Is `UNet-image+mask` still the more stable learned baseline?
5. Does a slightly longer horizon improve learned-model competitiveness without overturning the short-term framing?

## Datasets to Use

### Dataset 1: Sprint Benchmark

Already completed.

Purpose:

- pilot evidence,
- debugging,
- quick hypothesis generation.

### Dataset 2: Medium Benchmark

Config:

- `configs/benchmark_medium.yaml`

Purpose:

- main benchmark evidence for the paper,
- larger sample counts than the sprint run,
- same task structure and same modeling pipeline.

## Exact Experiment Set

### E1. Medium dataset generation

Generate one frozen medium-scale dataset using:

- `configs/benchmark_medium.yaml`

Deliverables:

- dataset artifact path
- generation summary
- manifest and split files

### E2. Medium baseline reference run

Run:

- `LOCF`
- `UNet-mask`
- `UNet-image+mask`

Suggested training budget:

- `epochs=12`

Deliverables:

- `all_baselines_summary.json`
- per-sample outputs
- checkpoint files

### E3. Medium breakdown analysis

For the medium dataset, compute:

- overall summary table
- per-tier table
- per-horizon table
- tier + horizon table
- worst-case table

Deliverables:

- CSV tables
- narrative notes in experiment log

### E4. Medium seed stability sweep

Run:

- `5` seeds on the medium dataset at `epochs=12`

Recommended seeds:

- `7,21,42,123,999`

Purpose:

- enough to assess stability on the larger dataset without overspending time

Deliverables:

- seed sweep summary JSON
- per-seed summary table
- stability interpretation

### E5. Medium horizon extension

Run one additional medium-dataset experiment with:

- `horizons=1,2,3`

Purpose:

- test whether learned models gain relative value at modestly longer horizons on the larger dataset

Deliverables:

- horizon comparison table
- short discussion note for the paper

## What to Store for Every Experiment

For every experiment, preserve:

1. command used
2. config used
3. output directory
4. summary JSON
5. per-sample JSON
6. table exports
7. one short interpretation paragraph

All interpretations should also be copied into:

- `docs/EXPERIMENT_LOG.md`

## Tables We Need by June 25

1. Core baseline table
   - `LOCF`, `UNet-mask`, `UNet-image+mask`
   - sprint dataset
   - medium dataset

2. Tier breakdown table
   - `Tier A`, `Tier B`, `Tier C`

3. Horizon breakdown table
   - `h=1`, `h=2`, and where run, `h=3`

4. Seed stability table
   - mean, std, min, max across seeds

5. Failure-case summary table
   - worst cases by Dice
   - tier membership

## Figures We Need by June 25

1. Mean Dice by method and tier
2. Mean Dice by method and horizon
3. Seed variability plot for learned baselines
4. Optional: qualitative failure-case panel from easy vs hard tiers

## Daily Plan

### June 22

- finalize pilot interpretation
- generate medium benchmark
- start medium baseline reference run

### June 23

- finish medium baseline run
- compute medium breakdown tables
- update experiment log

### June 24

- run medium seed sweep
- compute stability summary
- start horizon `1,2,3` run on medium dataset if time allows

### June 25

- finalize tables and figures
- write short result interpretations for each major experiment
- stop experiment expansion unless a critical gap remains

## Stop Conditions

Stop running new experiments once these are true:

1. medium-scale baseline table is complete
2. medium-scale tier breakdown is complete
3. medium-scale seed stability summary is complete
4. one horizon-extension result is complete

At that point, the project should move into writing-first mode.

## Writing Phase Input Package

By the start of June 26, we want a folder of evidence containing:

- frozen configs used
- output directories
- summary tables
- exported CSVs
- selected figures
- experiment log entries

That package should be sufficient to draft:

- abstract
- methods
- results
- limitations
- discussion
