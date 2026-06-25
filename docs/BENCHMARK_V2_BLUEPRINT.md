# Benchmark v2 Blueprint

This document defines the next-stage redesign of `SyntheticTumorBenchmark`.

The point of Benchmark v2 is not to chase one real dataset too closely. The point is to build a more principled and useful longitudinal tumor forecasting benchmark that:

- remains synthetic and controllable,
- reflects key challenge axes of tumor growth,
- supports short-term forecasting research,
- and stands up better to realism and difficulty audits.

## Why v2 Is Needed

The current benchmark already does several things well:

- it provides a complete end-to-end longitudinal generation pipeline,
- it supports forecasting with multiple past sessions,
- it includes a useful tier ladder,
- and it already exposes a meaningful forecasting phenomenon:
  - immediate forecasting is highly persistence-dominated,
  - yet stronger learned models can outperform `LOCF` under some synthetic regimes.

However, the first audit also showed important limitations:

- session counts are too short,
- follow-up windows are too short,
- growth-rate behavior does not match real data well,
- tumor size scale is not physically calibrated,
- and the tier system is helpful but still too coarse to serve as a benchmark specification by itself.

So v2 should not just be "more data." It should be "better-defined data."

## Core Design Principle

Benchmark v2 should be organized around **challenge axes**, with tiers acting as readable benchmark regimes built from those axes.

This keeps the benchmark useful for data mining and forecasting research:

- we can stress-test what models fail on,
- we can trace performance to specific data properties,
- and we can avoid a vague realism claim.

## Scientific Goal

The central task remains:

`Given k prior longitudinal sessions, forecast the next tumor state over short horizons.`

The benchmark should be especially strong for:

- immediate next-step forecasting,
- short-horizon forecasting under high persistence,
- identifying regimes where simple carry-forward baselines fail,
- and quantifying when learned models provide meaningful value.

## Proposed Challenge Axes

### 1. Temporal Depth

What should vary:

- number of sessions per patient,
- total follow-up duration,
- inter-scan interval variability,
- missing or irregular visits.

Why it matters:

- real longitudinal data is not uniformly sampled,
- short-term forecasting difficulty depends strongly on how much trajectory history is available,
- and forecasting with `k=1`, `k=3`, and `k>3` should be distinguishable.

### 2. Growth Regime

What should vary:

- slow / stable tumors,
- moderate growth tumors,
- aggressive growth tumors,
- treatment-perturbed trajectories,
- and partial-response or rebound patterns if feasible.

Why it matters:

- the current benchmark likely under-represents realistic growth diversity,
- and `LOCF` is strongest exactly when the process is highly persistent and weakly changing.

### 3. Spatial Complexity

What should vary:

- isotropic vs anisotropic expansion,
- single-lobe vs irregular spatial spread,
- shape elongation and deformation,
- local heterogeneity in diffusion and proliferation.

Why it matters:

- this is where Tier `C` is supposed to matter most,
- and it is one of the main bridges between idealized and more realistic tumor evolution.

### 4. Tumor Burden / Scale

What should vary:

- baseline tumor size,
- size at forecast start,
- relative change compared to occupied brain region,
- and size-conditioned difficulty bins.

Why it matters:

- very small absolute changes can make learned methods look worse than persistence,
- and models may behave differently for tiny, moderate, and large lesions.

### 5. Observation Realism

What should vary:

- image noise,
- bias field strength,
- boundary ambiguity,
- mask perturbation or partial annotation error,
- and modality informativeness.

Why it matters:

- forecasting methods are often tested under cleaner inputs than what downstream use would see,
- and robustness to noisy observation is a genuine data mining question.

### 6. Treatment Effects

What should vary:

- timing of treatment onset,
- strength of treatment response,
- delayed response,
- and post-treatment stabilization or rebound.

Why it matters:

- treatment can break naive persistence assumptions,
- and treatment-aware longitudinal prediction is important in the literature.

## Proposed Benchmark Structure

Rather than replacing the current tiers, v2 should refine them into interpretable regimes.

### Regime A: Controlled Persistence

Purpose:

- establish the persistence-dominated floor,
- quantify when `LOCF` is hard to beat,
- and expose the simplest immediate forecasting setting.

Expected properties:

- smoother growth,
- lower shape complexity,
- weaker treatment perturbation,
- shorter and cleaner histories.

### Regime B: Mechanistic Smooth Growth

Purpose:

- capture more realistic reaction-diffusion style evolution while staying structured.

Expected properties:

- moderate anisotropy,
- moderate heterogeneity,
- cleaner growth dynamics than the hardest regime,
- medium temporal depth.

### Regime C: Heterogeneous Short-Horizon Challenge

Purpose:

- serve as the main synthetic frontier for short-term forecasting.

Expected properties:

- stronger anisotropy,
- more irregular growth,
- more trajectory diversity,
- more treatment perturbation,
- and more ambiguous next-step change.

### Optional Regime D: Robustness / Stress-Test

Only if time permits.

Purpose:

- isolate observation noise and missingness,
- not necessarily realism,
- but explicit robustness testing.

Expected properties:

- scan irregularity,
- noisier inputs,
- imperfect observation or masks,
- and stronger mismatch between latent growth and visible appearance.

## What v2 Should Improve Immediately

These are the first practical redesign targets.

### Priority 1. Increase longitudinal depth

Target direction:

- increase sessions per patient,
- increase follow-up duration,
- allow more irregular time gaps.

Reason:

- the audit showed the current benchmark is too shallow compared with real longitudinal cohorts.

### Priority 2. Recalibrate growth-rate regimes

Target direction:

- explicitly sample slow, medium, and aggressive growth groups,
- rather than relying on one broad parameter family.

Reason:

- current `Tier A` is too aggressive on average,
- current `Tier B/C` are too slow on average,
- and realistic difficulty comes from a distribution of regimes.

### Priority 3. Strengthen size-conditioned challenge cases

Target direction:

- intentionally include tiny-change next-step cases,
- moderate-change cases,
- and clinically noticeable growth cases.

Reason:

- this directly targets the persistence problem and makes the benchmark scientifically sharper.

### Priority 4. Make treatment trajectories more informative

Target direction:

- introduce clearer pre-treatment, early-response, and post-treatment patterns.

Reason:

- treatment transitions are one of the most realistic sources of nontrivial forecasting difficulty.

### Priority 5. Separate realism from difficulty

Target direction:

- audit every regime on both realism proxies and forecasting difficulty.

Reason:

- a harder benchmark is not automatically a more realistic benchmark,
- and we should not blur those two claims.

## Evaluation Tasks for v2

The benchmark should support multiple related tasks, even if the paper focuses on one.

### Primary task

- next-step mask forecasting from `k` prior sessions

### Secondary task

- horizon-conditioned short-term forecasting:
  - `h=1`
  - `h=2`
  - `h=3`

### Optional auxiliary tasks

- growth/no-growth classification,
- size-delta regression,
- uncertainty calibration,
- voxelwise probability-field prediction.

That last task is especially interesting because it aligns with the idea that future growth can be represented as a spatial probability field rather than only a hard mask.

## Metrics v2 Should Emphasize

### Core metrics

- Dice
- relative volume difference
- volume change error

### Slice / patient reporting

- per-patient tables
- per-regime summaries
- per-horizon summaries

### Uncertainty / probabilistic metrics

If we later add probabilistic outputs:

- Brier score
- calibration curves
- negative log-likelihood for voxelwise probabilities

## Minimal Realism Audit for Every New Dataset Release

Every new benchmark release should be accompanied by:

1. cohort statistics table
2. per-regime statistics table
3. shape proxy table
4. growth dynamics table
5. synthetic-vs-real comparison against at least one processed cohort
6. limitations note

This should become part of the benchmark release process, not a one-off analysis.

## What the Next Paper Can Honestly Claim

If we follow this blueprint, a rigorous paper can eventually claim:

- short-horizon tumor forecasting is strongly persistence-dominated,
- benchmark design materially changes what conclusions appear true,
- some learned models only help in specific synthetic regimes,
- and benchmark realism must be audited explicitly before drawing stronger clinical conclusions.

That is a real data-mining contribution because it is about:

- benchmark design,
- controlled longitudinal evaluation,
- model-vs-data interaction,
- and the structure of failure modes in forecasting.

## Immediate Next Build Plan

### Phase 1. Freeze benchmark v1 evidence

- keep current v1 results as a documented baseline
- do not over-claim realism

### Phase 2. Design benchmark v2 parameter families

- define temporal regimes
- define growth-rate regimes
- define spatial-complexity regimes
- define treatment regimes

### Phase 3. Generate pilot v2 cohorts

- small pilot for debugging
- medium pilot for audit
- larger benchmark only after audit passes basic checks

### Phase 4. Re-run forecasting baselines

- `LOCF`
- direct `U-Net`
- residual `U-Net`
- stronger CNN variants
- optional PDE / mechanistic comparator

### Phase 5. Real-data bridge

- validate against processed `SAILOR`
- optionally validate against a second cohort if available

## Decision Rule Before More Large Experiments

Before spending more compute on benchmark-scale model comparisons, we should first answer:

1. what exact regimes do we want v2 to contain,
2. what statistics should each regime target,
3. and what minimal realism thresholds do we expect before we trust those runs?

If we cannot answer those questions, more model runs are not the bottleneck.

## Bottom Line

Benchmark v1 gave us a useful and honest result:

- persistence is a serious short-horizon baseline,
- stronger learned models can beat it in some synthetic regimes,
- but benchmark realism is still incomplete.

Benchmark v2 should convert that result into a more defensible research asset:

- a structured synthetic benchmark for longitudinal tumor forecasting,
- centered on short-term prediction,
- with explicit challenge axes,
- clear realism audits,
- and interpretable regime-based evaluation.
