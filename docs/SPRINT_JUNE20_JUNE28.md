# Sprint Plan: June 20 to June 28

## Goal

Turn `SyntheticTumorBenchmark` into a frozen, reproducible benchmark artifact with:

- one fixed dataset release,
- one official baseline table,
- one or two targeted benchmark analyses beyond the baseline,
- enough material for a DMS submission draft or, at minimum, a durable long-term benchmark release.

## Scope Decision

For this 8-day sprint, the benchmark itself is the main product.

That means:

- `yes`: dataset generation, frozen config, baselines, reporting, benchmark analysis
- `yes`: one focused experimental story about difficulty tiers / augmentation / forecasting behavior
- `no`: full SAILOR integration unless it is already nearly ready
- `no`: ambitious PDE-to-real calibration pipeline
- `no`: broad privacy/fidelity framework unless it directly supports one core result

## Working Paper Angle

If we submit to DMS, the paper should be framed as:

**A reproducible synthetic longitudinal tumor forecasting benchmark with controlled difficulty tiers and baseline analyses.**

The strongest near-term claim is not:

- "we solved real tumor forecasting"

The strongest near-term claim is:

- "we built a controlled benchmark for longitudinal tumor forecasting and show how standard baselines behave across increasing simulation difficulty."

Optional secondary claim:

- "targeted synthetic augmentation or tier-aware training improves hard-case performance."

## Success Criteria

Minimum success by June 28:

1. A fixed benchmark dataset has been generated with locked config and seed.
2. `LOCF`, `UNet-mask`, and `UNet-image+mask` have been run on the same frozen dataset.
3. Results are saved in one benchmark summary table with per-sample outputs.
4. At least one additional benchmark analysis is complete.
5. The repo docs explain exactly how to reproduce the dataset and baseline results.

Stretch success:

1. Add one new benchmark metric or breakdown that strengthens the benchmark story.
2. Add one small new experiment such as tier-aware evaluation or augmentation sensitivity.
3. Draft a 4-6 page paper skeleton with figures and tables already filled in.

## Primary Deliverables

By the end of the sprint, the repo should contain:

1. A frozen dataset artifact path and release note.
2. A frozen baseline summary JSON.
3. A benchmark results table for the paper.
4. At least one figure showing benchmark difficulty structure or baseline failure patterns.
5. Updated docs describing dataset generation, protocol, and reference results.

## Daily Plan

### Day 1: Freeze the benchmark

Deliverables:

- confirm environment and dependencies
- generate one fixed dataset using `configs/benchmark_v1.yaml`
- save dataset path and generation summary
- run `LOCF` first to validate the pipeline end-to-end

Decision:

- if dataset generation fails or is too slow, shrink dataset size immediately for sprint mode and preserve the original config as the full release target

### Day 2: Run official baselines

Deliverables:

- run `scripts/run_all_baselines.py` on the fixed dataset
- produce `all_baselines_summary.json`
- inspect per-sample outputs and verify split/protocol consistency

Decision:

- if UNet runtime is too slow, reduce epochs for the sprint reference run and document it clearly as `benchmark_v0p1_sprint`

### Day 3: Build benchmark reporting

Deliverables:

- create a clean summary table from baseline outputs
- add per-tier breakdown: `Tier A`, `Tier B`, `Tier C`
- add horizon breakdown: `h=1`, `h=2`

Why this matters:

- this is the first place the benchmark becomes a research artifact rather than just a codebase

### Day 4: Add one stronger benchmark diagnostic

Choose one:

- per-patient failure table
- relative volume difference metric
- growth-rate-stratified evaluation
- session-gap sensitivity using `delta_days`

Recommended choice:

- add `RVD` plus per-tier/per-horizon reporting

Reason:

- it strengthens the evaluation without forcing a new model

### Day 5: Run one focused experiment

Choose exactly one:

- tier-aware training vs pooled training
- training on easier tiers and testing on harder tiers
- synthetic augmentation ratio sensitivity
- ablation: `mask` vs `image+mask` under each tier

Recommended choice:

- `mask` vs `image+mask` with tier-wise and horizon-wise analysis

Reason:

- the code already supports this, so it is likely to finish in time

### Day 6: Convert outputs into paper evidence

Deliverables:

- one benchmark table
- one breakdown table
- one main figure
- one failure-case figure or panel

Paper-ready figure ideas:

- Dice by method and tier
- Dice by method and horizon
- qualitative examples from easy vs hard tiers

### Day 7: Tighten the repo and write

Deliverables:

- update README with frozen benchmark commands
- update docs with benchmark release notes
- create a short paper outline with filled claims, methods, results, and limitations

### Day 8: Submission packaging

Deliverables:

- write abstract, intro, methods, and results draft
- finalize tables and figures
- write honest limitations and future work
- decide whether the DMS version is strong enough to submit

## Exact Experiment Priority

Work in this order and do not skip ahead:

1. Frozen dataset generation
2. Official baseline run
3. Per-tier and per-horizon benchmark table
4. One added metric or diagnostic
5. One focused experiment
6. Paper packaging

Anything beyond that is optional.

## Recommended Core Experiment Set

These are the experiments most likely to finish and still tell a useful story:

1. `LOCF` vs `UNet-mask` vs `UNet-image+mask`
2. Per-tier breakdown: `A`, `B`, `C`
3. Per-horizon breakdown: `1`, `2`
4. Added metric: `RVD`
5. One focused study:
   easier-tier training vs full-tier training, or `mask` vs `image+mask` behavior by tier

## Non-Goals for This Sprint

Do not let these consume the sprint unless everything above is already done:

- SAILOR loader
- real-data calibration
- large architecture search
- privacy paper-quality evaluation suite
- full synthetic-to-real transfer study

These can still be good long-term directions, but they are not the fastest route to a finished artifact by June 28.

## Team Split for 4 People

### Lead

- own sprint board, daily priorities, result review, paper narrative

### Person 2

- dataset generation, frozen artifact management, config and manifest validation

### Person 3

- baseline execution, metric extraction, results logging

### Person 4

- figures, tables, doc cleanup, reproducibility notes

If only one or two people are active, collapse the work to:

1. benchmark freeze
2. baseline run
3. result tables
4. one focused experiment

## Daily Standup Questions

Every day, answer only these:

1. What artifact became reproducible today?
2. What result table or figure became paper-usable today?
3. What is the single blocker for tomorrow?

## Go / No-Go Standard for DMS

Submit if by June 27 you have:

1. one frozen dataset release,
2. one official baseline table,
3. one additional experiment with a clear takeaway,
4. one coherent claim about benchmark difficulty and baseline behavior.

Do not submit if the sprint ends with:

1. only code changes and no reference results,
2. many incomplete experiments,
3. no clear benchmark narrative.

## Immediate First Three Actions

1. Generate the fixed dataset.
2. Run `LOCF`, then run `run_all_baselines.py`.
3. Create the first benchmark summary table from the outputs before touching any new feature work.
