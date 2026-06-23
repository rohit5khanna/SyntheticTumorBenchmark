# Data Audit Plan

This document defines the data-centered work needed to make `SyntheticTumorBenchmark` scientifically defensible for the DMS submission and useful as a longer-term benchmark project.

The goal is not only to evaluate models **on** the synthetic benchmark, but also to characterize, stress-test, and partially validate the benchmark **itself**.

## Why This Matters

At the current stage, the project already has:

- a synthetic longitudinal forecasting benchmark,
- a meaningful tier ladder (`A -> B -> C`),
- multiple model comparisons,
- and confirmation-stage synthetic results.

However, benchmark quality and benchmark credibility now matter as much as model performance.

To strengthen the paper, we need to answer:

1. What properties does the synthetic dataset actually have?
2. Do the tiers correspond to increasing difficulty and realism in measurable ways?
3. How does the synthetic data compare to processed `SAILOR`?
4. What does the synthetic benchmark capture well, and where does it diverge from real data?

## Audit Structure

The audit is divided into four layers:

1. Synthetic benchmark characterization
2. Tier-ladder validation
3. Synthetic-to-real comparison against `SAILOR`
4. Limitations and intended benchmark claims

## Layer 1: Synthetic Benchmark Characterization

These are the minimum descriptive statistics that should be computed for:

- overall dataset
- `Tier A`
- `Tier B`
- `Tier C`

### Core cohort statistics

- number of patients
- split counts (`train`, `val`, `test`)
- number of sessions per patient
- total forecastable samples by horizon

### Temporal statistics

- inter-scan interval distribution
- total follow-up duration distribution
- treatment prevalence
- treatment start-session distribution

### Tumor-size statistics

- tumor volume per session
- baseline tumor volume distribution
- final tumor volume distribution
- session-to-session absolute delta-volume
- session-to-session relative growth rate

### Geometry / shape proxies

These do not need to be perfect clinical shape metrics. They only need to provide interpretable proxies.

Candidate measures:

- bounding-box size in `x`, `y`, `z`
- elongation ratio:
  - `max(box_dims) / min(box_dims)`
- compactness proxy:
  - volume divided by bounding-box volume
- connected-component count

### Deliverables

1. one dataset characterization table
2. one per-tier dataset table
3. one short paragraph describing what the synthetic benchmark looks like numerically

## Layer 2: Tier-Ladder Validation

The tier system should be shown to mean something measurable.

### What to test

We want evidence that:

- `Tier A` is simpler / more regular,
- `Tier B` is mechanistic but more structured than `Tier C`,
- `Tier C` is more heterogeneous and anisotropic.

### Suggested checks

- variance of tumor volume change by tier
- elongation ratio by tier
- compactness proxy by tier
- connected-component count by tier
- forecast difficulty by tier:
  - `LOCF`
  - strongest direct model
  - strongest residual model

### Goal

Support the interpretation that the tiers form a progression:

- easier and more idealized
- to more mechanistic
- to more realistic and difficult

### Deliverables

1. one tier comparison table
2. one figure showing tier separation on key statistics
3. one paragraph explaining why `Tier C` is the synthetic frontier

## Layer 3: Synthetic-to-Real Comparison (`SAILOR`)

This is the most important credibility upgrade.

The aim is not to prove the synthetic benchmark is fully realistic.
The aim is to show:

- which real-data properties it reflects reasonably,
- which tier is closest to `SAILOR`,
- and which gaps remain.

### Minimum comparison statistics

Compare `SAILOR` against:

- synthetic overall
- `Tier A`
- `Tier B`
- `Tier C`

For:

- sessions per patient
- inter-scan interval distribution
- baseline tumor volume
- future tumor volume
- absolute delta-volume
- relative growth rate
- treatment prevalence / timing if available

### Optional comparison statistics

If feasible without slowing the project too much:

- elongation ratio
- compactness proxy
- connected-component count

### Key question

Does `Tier C` look closer to `SAILOR` than `Tier A/B` on the statistics that matter for forecasting?

If yes, that strengthens the tier-ladder claim.

If mixed, the paper can still report:

- `Tier C` is closer on some statistics,
- while the benchmark remains stylized on others.

That is still scientifically useful.

### Deliverables

1. synthetic vs `SAILOR` comparison table
2. two or three distribution plots
3. one paragraph of benchmark realism interpretation

## Layer 4: Benchmark Limitations and Claims

The paper should be explicit about what the benchmark is and is not.

### Claims we can plausibly make

- the benchmark is useful for controlled short-term forecasting analysis
- the tiers encode increasing synthetic complexity
- `Tier C` is the most realistic synthetic regime among the three
- the benchmark supports comparative method analysis
- the synthetic observations can motivate targeted real-data validation

### Claims we should avoid

- the benchmark is a clinically faithful substitute for real data
- strong synthetic performance implies real-world superiority
- `Tier C` is fully realistic

### Deliverables

1. limitations subsection in the paper
2. revised dataset card
3. benchmark realism paragraph in methods or discussion

## Immediate DMS Priority Order

If time is limited, do the data work in this order:

1. Synthetic benchmark characterization
2. Synthetic vs `SAILOR` basic statistics
3. Tier-ladder validation with shape/growth proxies
4. Optional deeper shape realism analysis

## Minimum Figures

Recommended first-pass figures:

1. sessions-per-patient distribution:
   - synthetic overall
   - `Tier A/B/C`
   - `SAILOR`

2. inter-scan interval distribution:
   - synthetic overall
   - `Tier A/B/C`
   - `SAILOR`

3. tumor-volume or delta-volume distribution:
   - synthetic overall
   - `Tier A/B/C`
   - `SAILOR`

4. tier-ladder difficulty figure:
   - method performance by tier

## Minimum Tables

Recommended first-pass tables:

1. Dataset characterization table
2. Tier comparison table
3. Synthetic vs `SAILOR` realism table
4. Tier-wise forecasting performance table

## How This Supports The Long-Term Benchmark Vision

For DMS, this audit makes the current paper more defensible.

For the longer term, it also lays the foundation for turning `SyntheticTumorBenchmark` into a stronger benchmark artifact by:

- documenting benchmark properties,
- clarifying realism limits,
- identifying which synthetic regimes best align with real data,
- and motivating future benchmark versions with improved realism.

## Practical Next Step

Implement a single audit script or notebook that computes:

- cohort stats,
- temporal stats,
- volume / growth stats,
- simple shape proxies,
- and synthetic vs `SAILOR` comparison tables.

This should become the canonical benchmark-audit pipeline for the project.
