# Benchmark v3-Lite Plan

This document defines the next benchmark revision after the `biophys1` evidence phase.

The goal is **not** to tune the synthetic benchmark to one real dataset.
The goal is to use real-data comparison as a reality check, identify broad missing regime families, and improve the benchmark in a principled and general way.

## Current Evidence Summary

The current project now has four important findings:

1. Short-horizon tumor forecasting is strongly regime-dependent.
2. `LOCF` remains highly competitive in stable persistence-heavy cases.
3. `ResUNet` provides clear value in active-growth, larger-volume cases.
4. The current synthetic benchmark does **not** match the real longitudinal regime mix well.

The real-data bridge particularly showed:

- `Tier A` is closest to real data on stability fraction.
- `Tier B` is the closest overall compromise tier.
- `Tier C` is too growth-heavy to be treated as the most realistic tier.

This means the old interpretation:

- `A = easy`
- `B = medium`
- `C = most realistic`

should be replaced with:

- `A = stability / persistence regime`
- `B = mixed transferable regime`
- `C = aggressive-growth stress regime`

That reinterpretation is scientifically stronger and more consistent with the evidence.

## What v3-Lite Should Achieve

Benchmark `v3-lite` should improve the synthetic regime space without claiming full realism.

The immediate goals are:

1. increase regime diversity
2. increase treatment prevalence and treatment-affected dynamics
3. increase the share of stable and shrinking transitions
4. increase lesion scale relative to the current benchmark
5. preserve a clear regime structure for short-horizon forecasting analysis

## General Design Principles

### Principle 1. Do not fit to SAILOR specifically

The benchmark should not be adjusted until one real dataset is matched closely.

Instead, use real data only to detect broad gaps:

- too few stable trajectories
- too few shrinking trajectories
- too little treatment prevalence
- tumor burden too small
- follow-up variability too narrow

### Principle 2. Separate regime coverage from realism claims

The benchmark should aim to cover a plausible space of longitudinal tumor behaviors:

- persistence-dominated cases
- mixed-growth cases
- aggressive-growth cases
- treatment-affected cases
- shrinking or regressing cases

This is more useful than claiming one tier is “the realistic one.”

### Principle 3. Keep the benchmark interpretable

The tier system should remain readable.

The benchmark should help answer:

- when persistence is enough
- when learned forecasting helps
- what kinds of tumor trajectories are currently missing

## Proposed v3-Lite Regime Semantics

### Regime A: Stability / Persistence

Purpose:

- preserve the persistence-dominated regime
- keep a realistic fraction of stable short-horizon cases

Desired behavior:

- smoother evolution
- modest growth
- more treatment-affected stabilization
- some shrinking transitions

### Regime B: Mixed Clinical Regime

Purpose:

- serve as the best general short-horizon benchmark regime
- become the main synthetic compromise tier

Desired behavior:

- mixture of stable, growing, and shrinking transitions
- moderate treatment prevalence
- moderate lesion scale
- broader follow-up variability

### Regime C: Aggressive Growth Stress Regime

Purpose:

- preserve a hard active-growth forecasting regime
- test learned models under stronger expansion and heterogeneity

Desired behavior:

- larger lesions
- stronger anisotropy
- higher growth burden
- not necessarily closest to real data

## Gaps That v3-Lite Can Address Immediately

These are changes that can be made **using the current generator controls**.

### 1. Longer and more variable histories

Current gap:

- real data shows longer and more variable follow-up windows

Action:

- increase `n_sessions_max`
- increase `days_interval_max`
- add tier-specific schedule variation

### 2. Higher treatment prevalence

Current gap:

- real data is much more treatment-heavy

Action:

- increase `treatment_patient_prob`
- vary treatment prevalence by regime
- make treatment start earlier in some regimes

### 3. Larger lesion scale

Current gap:

- synthetic lesion volume remains too small relative to real data

Action:

- increase `init_sigma_vox_range`
- increase `init_amp_range`
- lower mask threshold slightly if needed

### 4. More stability and regression

Current gap:

- current benchmark, especially `Tier C`, is too growth-heavy

Action with current controls:

- increase treatment prevalence / treatment strength in `A/B`
- reduce `rho_range` in `A`
- keep `B` moderate rather than aggressively growing

Important limitation:

- the current simulator does **not** explicitly model rebound, response heterogeneity, or subgroup-conditioned shrinkage regimes.

So v3-lite can only partially address this.

## Gaps That Need New Generator Controls

These should be treated as `v3-full` items rather than forced into `v3-lite`.

### 1. Explicit transition-regime control

Needed eventually:

- stable subgroup
- growing subgroup
- shrinking subgroup

Current problem:

- these emerge indirectly from PDE + treatment settings, not by design

### 2. Better treatment-response diversity

Needed eventually:

- early response
- delayed response
- rebound
- partial-response and nonresponse cases

### 3. Size-targeted regime sampling

Needed eventually:

- explicit small / medium / large lesion bins
- rejection sampling or regime-conditioned initialization

### 4. Better real-like heterogeneity

Needed eventually:

- richer anisotropy control
- more varied lesion topology
- more realistic irregular low-compactness failure cases

## Proposed v3-Lite YAML Strategy

Use three regimes with these high-level targets:

### A

- lower `rho`
- lower `Dw`
- higher treatment prevalence
- earlier treatment onset
- longer but calmer histories

Expected effect:

- more stable or weakly changing transitions

### B

- moderate `rho`
- moderate `Dw`
- high treatment prevalence
- moderately larger initial lesions
- broader time intervals

Expected effect:

- mixed regime and likely best real-data compromise

### C

- larger initial lesions
- stronger anisotropy / heterogeneity
- keep active growth
- keep it as stress regime rather than realism proxy

Expected effect:

- strong learned-model challenge

## Recommended Evaluation Questions For v3-Lite

After generating `v3-lite`, do **not** jump straight to model training.

First run:

1. audit summaries
2. transition-regime comparison against `REAL`
3. size-scale comparison
4. treatment prevalence comparison

Then ask:

1. Did stable and shrinking fractions improve?
2. Did lesion scale move upward?
3. Is `B` still the closest overall regime?
4. Did we preserve the forecasting challenge?

## What Would Count As Success

`v3-lite` should be considered successful if:

- no tier is absurdly growth-dominated
- at least one regime approximates a real mixed transition profile better than current `B`
- lesion volumes shift upward meaningfully
- treatment prevalence moves closer to real data
- `A`, `B`, `C` remain interpretable and forecasting-relevant

## Honest Scope

Even if `v3-lite` improves the benchmark, it should still be described honestly as:

- a controlled synthetic longitudinal tumor forecasting benchmark
- with broader regime coverage than `biophys1`
- but not a clinically faithful surrogate for real tumor evolution

That is a strong and defensible claim.
