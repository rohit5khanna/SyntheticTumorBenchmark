# Adaptive Residual Experiment Ladder

## Goal

Run a small sequence of feedback-driven experiments for short-term longitudinal tumor forecasting.

The ladder is centered on one modeling family:

- residual forecasting over a strong persistence prior,
- using recent multi-session history,
- starting with `k = 3`.

## Motivation

Current results suggest:

- `LOCF` dominates immediate next-step forecasting,
- learned models become more useful at slightly longer short-term horizons,
- the next reasonable modeling idea is to predict residual change rather than full future masks from scratch.

## Main Candidate Model

`ResidualHistoryUNet`

Definition:

- input: last `k` observed sessions, with `k = 3` initially,
- output: residual correction to a strong prior induced by the most recent observed mask,
- target: future tumor mask at short-term horizons.

## Ladder

### R1: Residual prototype on sprint dataset

Purpose:

- verify implementation,
- check that training is stable,
- get fast feedback before scaling.

Default setup:

- dataset: sprint
- input mode: `image_mask`
- history length: `3`
- horizons: `1,2`
- seed: `42`

Decision gate:

- if clearly broken or worse everywhere than baseline, inspect formulation before scaling
- if helpful on `h=2` or at least competitive overall, move to medium

### R2: Residual prototype on medium dataset

Purpose:

- test the same idea on the main evidence track.

Default setup:

- dataset: medium
- input mode: `image_mask`
- history length: `3`
- horizons: `1,2`
- seed: `42`

Decision gate:

- if it improves over current `UNet-image+mask`, especially at `h=2`, keep going
- if it helps only at `h=1`, that is less interesting
- if it is worse everywhere, stop and revise

### R3: Residual history-length ablation

Purpose:

- determine whether gain comes from residual learning alone or from recent history.

Suggested comparison:

- `k = 1`
- `k = 3`

Default setup:

- dataset: medium
- input mode: `image_mask`
- horizons: `1,2`
- seed: `42`

Decision gate:

- if `k=3` helps clearly over `k=1`, that supports a genuinely longitudinal claim
- if both are similar, residual learning may matter more than added history

### R4: Residual stability check

Purpose:

- determine whether any gain is robust.

Suggested seeds:

- `21`
- `42`
- `123`

Default setup:

- best residual variant from `R2/R3`
- dataset: medium

Decision gate:

- if average gain survives 3 seeds, it is paper-worthy
- if not, report as exploratory

## What To Record For Each Step

1. command used
2. config used
3. baseline it is trying to beat
4. overall summary
5. horizon breakdown
6. tier breakdown
7. interpretation
8. next branch decision

## Success Criteria

The residual-history direction is worth keeping in the paper if it does at least one of the following on the medium dataset:

1. beats `UNet-image+mask` overall
2. beats `UNet-image+mask` at `h=2` or `h=3`
3. improves performance on harder regimes without collapsing elsewhere
4. shows a clear and stable benefit from using recent history
