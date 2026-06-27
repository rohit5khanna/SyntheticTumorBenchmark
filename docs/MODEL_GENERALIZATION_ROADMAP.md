# Model Generalization Roadmap

This document defines the next experimental phase after stabilizing the `biophys1` benchmark and establishing the first regime-dependent forecasting result.

The purpose is not to build a model zoo.
The purpose is to test whether the current conclusion is:

- specific to one strong CNN architecture,
- or reflective of broader model-family behavior across tumor-growth regimes.

## Current Starting Point

The current strongest benchmark-result pair is:

- dataset: `fixed_dataset_v2_core_candidate_biophys1`
- baseline reference: `LOCF`
- strongest learned model: `resunet_image_mask`

Current evidence already supports:

- persistence is highly competitive in `Tier A`,
- stronger learned models help much more in `Tier B` and `Tier C`,
- and tumor / trajectory features such as size, recent growth, and future growth strongly explain that shift.

The next question is:

`Do different model families generalize differently across these regimes?`

## Main Research Question

How do architecture family and inductive bias interact with longitudinal tumor-growth regimes?

Sub-questions:

1. Are the current conclusions specific to `ResUNet`, or do they hold across broader model families?
2. Which model families are robust across regimes?
3. Which model families are biased toward particular regimes?
4. Does training on one regime generalize to another?

## Core Experimental Principle

Keep the model set small and representative.

The goal is not broad coverage.
The goal is controlled contrast between:

- persistence
- local CNNs
- stronger residual CNNs
- and one transformer-style global-context model

## Recommended Model Set

### Fixed references

1. `LOCF`
2. `UNet-mask`
3. `UNet-image+mask`
4. `ResUNet-image+mask`
5. `PlainCNN-image+mask`

These already exist and should remain the shared baseline set.

### One additional transformer-style model

Recommended first addition:

- `UNETR` or `SwinUNETR`

Why:

- they represent a distinct global-context architecture family,
- they are recognized in medical imaging literature,
- and one such model is enough to test whether the current story is architecture-family dependent.

### Why not add many more models now

Avoid:

- multiple transformer variants,
- diffusion / flow models at this stage,
- heavy hyperparameter tuning,
- architecture-specific tricks that break comparability.

That would create noise faster than insight.

## Phase Structure

## Phase G1: Freeze the Current Reference Result

Goal:

- preserve the current benchmark and core evidence before expanding scope

Deliverables:

- saved figures from the regime analysis
- frozen summary tables
- log entry noting that the `ResUNet` result is stable across seeds

Status:

- mostly complete

## Phase G2: Architecture Family Comparison On Mixed Training

Goal:

- test whether model families behave differently when trained on the full mixed benchmark

Train on:

- all training samples from `biophys1`

Evaluate on:

- full test split
- `Tier A`
- `Tier B`
- `Tier C`
- `h=1,2,3`

Required models:

- `LOCF`
- `UNet-image+mask`
- `ResUNet-image+mask`
- one transformer-style model

Optional:

- `UNet-mask`
- `PlainCNN-image+mask`

Questions answered:

1. Does the transformer family beat persistence overall?
2. Is it competitive specifically on `Tier B/C`?
3. Does it generalize across regimes as well as `ResUNet`?
4. Is the current conclusion broader than one CNN winner?

### Success criterion

At least one of the following should happen:

- the transformer supports the same regime-dependent story,
- the transformer fails in a way that teaches us something about regime bias,
- or `ResUNet` remains uniquely robust, which is also an interpretable result.

Any of these outcomes can be useful if framed honestly.

## Phase G3: Cross-Regime Transfer

Goal:

- test whether models trained on one growth regime transfer to others

Suggested matrix:

- train on `A`, evaluate on `A/B/C`
- train on `B`, evaluate on `A/B/C`
- train on `C`, evaluate on `A/B/C`

Do this only for:

- `LOCF` reference
- `ResUNet-image+mask`
- transformer-style model

Why this matters:

- mixed-train results show overall regime behavior
- transfer results show whether a model has learned regime-specific shortcuts or broader structure

Key questions:

1. Does training on `Tier C` generalize downward to `A/B` better than the reverse?
2. Is a transformer more robust under regime shift?
3. Is `ResUNet` strong because it is more data-efficient or because it is more regime-specific?

### Warning

This phase is scientifically valuable, but more expensive.
It should come after mixed-train family comparison, not before.

## Phase G4: Input-Side Forecastability / Model-Suitability Analysis

Goal:

- move beyond architecture ranking toward model-choice reasoning

This phase uses input-side features only:

- current tumor volume
- recent growth
- recent stability
- history length
- treatment-at-input
- horizon
- regime / tier
- shape proxies

Main analyses:

1. case-type breakdown
   - persistence-dominated
   - learned-advantage
   - easy-for-all
   - hard-for-all

2. model-suitability summary
   - which observable properties correspond to each case type

3. optional shallow decision model
   - predict case type or model-suitability class from input-side features

This is likely the strongest data-mining extension beyond raw benchmark results.

## Phase G5: Real-Data Bridge

Goal:

- test whether the main logic of the synthetic result appears in real data too

This does **not** need to start as full model retraining on `SAILOR`.

Possible lighter-weight bridge options:

1. descriptive bridge
   - identify whether real trajectories show the same low-growth / high-growth / persistence split

2. baseline sanity check
   - examine whether `LOCF` is especially strong on small, stable real cases

3. limited forecasting bridge
   - evaluate one or two selected models on a small real-data setup if feasible

This phase is the biggest credibility upgrade, but should not block synthetic analysis progress.

## Recommended Order Of Execution

Implementation status update:

- `UNETR` support has been added as the first transformer-family probe.
- tier-aware split filtering has been added to the baseline pipeline.
- `scripts/run_cross_regime_transfer.py` now provides a compact way to execute the `train-tier -> eval-tier` matrix described in Phase `G3`.

### Immediate next order

1. freeze and save current `biophys1` evidence
2. add one transformer-style model
3. run mixed-train per-tier evaluation
4. if results are informative, run selected cross-regime transfer
5. continue the broader tumor-data / forecastability analysis

## Concrete First Architecture Addition

The first new family should be:

- `SwinUNETR` if available without excessive integration cost
- otherwise `UNETR`

Reason:

- both are recognizable,
- both represent transformer-style medical 3D segmentation,
- and one is enough for the first generalization question.

If integration cost becomes too high, do not force it immediately.
In that case, prefer:

- stronger analysis of existing model families,
- then revisit transformer integration as a second-step engineering task.

Current repo status:

- optional `UNETR` support is the first practical transformer-family target
- it should be treated as a compact family-comparison probe rather than a final architecture statement

## Interpretation Rules

To keep conclusions honest:

### If transformer underperforms

Do not conclude:

- "transformers are bad for tumor forecasting"

Instead consider:

- sample size sensitivity
- data efficiency
- regime mismatch
- training-budget mismatch

### If transformer performs similarly to `ResUNet`

Interpret as:

- the regime-dependent result is broader than one residual CNN architecture

### If transformer wins only on some tiers

Interpret as:

- architecture families themselves may be regime-biased

That is a useful result, not a failure.

## Long-Term Research Vision

The longer-term project should eventually integrate three threads:

1. benchmark design
2. forecasting-model comparison
3. tumor-data / regime analysis

The strongest version of the work is not:

- "here is our best model"

It is:

- "here is how longitudinal tumor-growth data structure determines forecasting difficulty, model suitability, and benchmark conclusions."

## Decision Rule Before Expanding Further

Before adding a second transformer or heavier generative models, ask:

1. Did the first family-comparison experiment change our scientific understanding?
2. Did it strengthen or weaken the current regime-dependent claim?
3. Is the next model likely to teach us something new, or only add leaderboard clutter?

If the answer to `3` is "leaderboard clutter," do not add it yet.
