# Paper Results and Key Claims

## Purpose

This document is a paper-facing condensation of the experiment log.

It is meant to help us write:

- the Results section,
- the Discussion section,
- and the claims / limitations language

without accidentally overstating what the current evidence supports.

## Working Paper Position

Current best framing:

> We study short-horizon longitudinal tumor forecasting under regime heterogeneity. Using a generalized synthetic benchmark with multiple growth regimes and a limited real-data sanity check on `SAILOR`, we show that learned longitudinal models can outperform persistence baselines outside highly stable regimes, while immediate next-step forecasting on real data remains strongly persistence-dominated.

This framing is stronger than a generic "model A beats model B" story because it is about:

- forecasting behavior,
- regime dependence,
- benchmark design,
- and the limits of persistence on different kinds of longitudinal cases.

## Benchmark Context

### Synthetic benchmark

Main final synthetic benchmark:

- dataset: `fixed_dataset_v3_lite_generalized`
- protocol: `fit_sessions=3`, horizons `1,2,3`
- compact model set:
  - `LOCF`
  - `UNet-image+mask`
  - `ResUNet-image+mask`

The three tiers should currently be interpreted as:

- tier `A`: stability / persistence-dominated regime
- tier `B`: mixed clinical compromise regime
- tier `C`: aggressive-growth / stress-test regime

### Real-data sanity check

External sanity evaluation:

- dataset: processed `SAILOR`
- protocol: `fit_sessions=3`, horizons `1,2,3`
- models:
  - `LOCF`
  - `ResUNet-mask`
  - `ResUNet-image+mask`

Important:

- `SAILOR` is not being used here to tune the synthetic benchmark.
- It is being used only as a constrained external validation check.

## Paper-Ready Results

### Result 1: On the final generalized synthetic benchmark, `ResUNet` is the strongest compact model

Overall mean Dice on `fixed_dataset_v3_lite_generalized`:

- `LOCF`: `0.7470`
- `UNet-image+mask`: `0.7608`
- `ResUNet-image+mask`: `0.8148`

Interpretation:

- A plain persistence baseline is strong.
- A standard `UNet` only improves modestly over persistence.
- A residual `UNet` gives a substantially larger gain.

Suggested paper sentence:

> On the generalized `v3-lite` benchmark, persistence remained competitive, but `ResUNet-image+mask` achieved the strongest overall performance, improving mean Dice from `0.747` (`LOCF`) to `0.815`.

### Result 2: Synthetic gains are regime-dependent rather than uniform

Tier-wise mean Dice:

- tier `A`
  - `LOCF`: `0.9017`
  - `ResUNet-image+mask`: `0.9013`
  - `UNet-image+mask`: `0.7740`

- tier `B`
  - `LOCF`: `0.5903`
  - `UNet-image+mask`: `0.6783`
  - `ResUNet-image+mask`: `0.6827`

- tier `C`
  - `LOCF`: `0.7104`
  - `UNet-image+mask`: `0.8325`
  - `ResUNet-image+mask`: `0.8427`

Interpretation:

- Tier `A` behaves like a pure persistence regime.
- Learned models add little in tier `A`.
- Learned models matter much more in tiers `B` and `C`.
- The strongest synthetic value of learning appears in non-stable regimes, not everywhere.

Suggested paper sentence:

> The benefit of learned forecasting was strongly regime-dependent: in the stability-dominated tier `A`, `LOCF` and `ResUNet` were effectively tied, whereas in tiers `B` and `C` learned models provided clear gains over persistence.

### Result 3: On synthetic data, `ResUNet` improves performance across all tested horizons

Horizon-wise mean Dice:

- horizon `1`
  - `LOCF`: `0.8693`
  - `UNet-image+mask`: `0.8435`
  - `ResUNet-image+mask`: `0.9141`

- horizon `2`
  - `LOCF`: `0.7146`
  - `UNet-image+mask`: `0.7561`
  - `ResUNet-image+mask`: `0.8055`

- horizon `3`
  - `LOCF`: `0.6570`
  - `UNet-image+mask`: `0.6827`
  - `ResUNet-image+mask`: `0.7248`

Interpretation:

- On synthetic data, the generalized benchmark no longer makes immediate next-step forecasting exclusively persistence-dominated.
- `ResUNet` wins at every tested horizon.
- This means the synthetic benchmark is capable of expressing cases where learned modeling is useful even near the next step.

Suggested paper sentence:

> Unlike the corrected real-data sanity check, the generalized synthetic benchmark did not remain persistence-dominated at horizon `1`; `ResUNet` was best at horizons `1`, `2`, and `3`.

### Result 4: The synthetic regime story partially transfers to real data, but with a narrower effect

Corrected `SAILOR` overall results:

- `LOCF`: `0.5159`
- `ResUNet-image+mask`: `0.5350`
- `ResUNet-mask`: `0.5400`

Corrected `SAILOR` horizon-wise results:

- horizon `1`
  - `LOCF`: `0.7069`
  - `ResUNet-mask`: `0.7022`
  - `ResUNet-image+mask`: `0.6932`

- horizon `2`
  - `LOCF`: `0.5534`
  - `ResUNet-mask`: `0.5778`
  - `ResUNet-image+mask`: `0.5770`

- horizon `3`
  - `LOCF`: `0.2875`
  - `ResUNet-mask`: `0.3402`
  - `ResUNet-image+mask`: `0.3348`

Interpretation:

- The real-data result supports the core motivation.
- `LOCF` remains extremely strong on the immediate next step.
- Learned models become more useful as the horizon increases.
- Image channels add little extra value over masks in the current real-data setup.

Suggested paper sentence:

> On corrected `SAILOR` evaluation, persistence remained slightly stronger at the immediate next step, whereas `ResUNet` became more favorable at longer horizons, supporting a horizon-dependent view of forecasting difficulty.

### Result 5: Cross-regime transfer supports the regime interpretation rather than a single global difficulty axis

Key `ResUNet` transfer results on `biophys1`:

- train `A` -> eval `A`: `0.8631`
- train `A` -> eval `B`: `0.6857`
- train `A` -> eval `C`: `0.6822`

- train `B` -> eval `A`: `0.7515`
- train `B` -> eval `B`: `0.8697`
- train `B` -> eval `C`: `0.8750`

- train `C` -> eval `A`: `0.6400`
- train `C` -> eval `B`: `0.8374`
- train `C` -> eval `C`: `0.8950`

Interpretation:

- Tier `A` behaves like a specialized persistence regime.
- Tier `B` is the most transferable mixed regime.
- Tier `C` captures aggressive cases and transfers better to `B/C` than to `A`.

Suggested paper sentence:

> Cross-regime transfer suggests that the benchmark tiers capture qualitatively different forecasting regimes rather than a single monotone difficulty scale.

### Result 6: Case analysis suggests when learning helps

From the `LOCF` vs `ResUNet` case-type analysis on `biophys1`:

- `target_wins`: `63 / 119` (`52.9%`)
- `both_easy`: `38 / 119` (`31.9%`)
- `baseline_wins`: `3 / 119` (`2.5%`)

`target_wins` cases tended to have:

- larger input tumor volumes,
- stronger future growth,
- stronger recent growth,
- and concentration in tiers `B` and `C`

Interpretation:

- learned forecasting helps most when the tumor is active enough that pure persistence becomes insufficient
- both methods do well on many easy stable cases
- true `LOCF` wins exist, but they are rare in the final synthetic benchmark

Suggested paper sentence:

> Case-level analysis indicates that learned gains are concentrated in larger and more actively evolving tumors, whereas highly stable cases remain well handled by persistence.

## Claims We Can Defend

The current evidence supports the following claims:

1. A generalized synthetic benchmark can reveal regime-dependent differences between persistence and learned longitudinal forecasting.
2. `LOCF` is not a trivial baseline; it remains extremely strong in stable regimes and at immediate next-step real forecasting.
3. Learned longitudinal models, especially `ResUNet`, can outperform persistence in mixed and aggressive synthetic regimes.
4. The synthetic-to-real story is not all-or-nothing; it partially transfers, but the real effect is smaller and more horizon-dependent.
5. Regime structure matters for understanding forecasting difficulty and model behavior.

## Claims We Should Avoid

The current evidence does **not** support the following stronger claims:

1. That `ResUNet` generally solves short-horizon tumor forecasting on real data.
2. That image channels are broadly necessary or broadly useful in the present setup.
3. That the synthetic benchmark is already a gold-standard or clinically realistic tumor benchmark.
4. That the synthetic tiers map one-to-one to clinical subtypes.
5. That the current real-data evaluation is large enough to settle model ranking conclusively.

## Recommended Results Structure For The Paper

### Results subsection order

1. Final synthetic benchmark table (`v3-lite`)
2. Tier-wise synthetic breakdown
3. Horizon-wise synthetic breakdown
4. Cross-regime transfer analysis
5. Case-type / driver analysis
6. Corrected `SAILOR` sanity validation

This order lets the paper build from:

- benchmark outcome,
- to interpretation,
- to external reality check.

## Minimal Figures and Tables To Keep

### Tables

1. Overall compact benchmark table on `v3-lite`
2. Horizon-wise benchmark table on `v3-lite`
3. Tier-wise benchmark table on `v3-lite`
4. `SAILOR` horizon-wise sanity table
5. Cross-regime transfer matrix

### Figures

1. Dice by horizon across methods
2. Win-rate by tier for `ResUNet` vs `LOCF`
3. Win-rate by future growth bin
4. Transfer heatmap

## Bottom-Line Assessment

This is now a defensible research story if we stay disciplined about scope.

Strongest version:

- we are not claiming universal superiority of deep forecasting over persistence
- we are showing that tumor forecasting behavior is regime-dependent
- and that a persistence baseline can be beaten, but mainly outside the most stable settings

Weakest version to avoid:

- "our model wins on synthetic data, therefore it is broadly better for tumor forecasting"

That weaker version will not survive scrutiny.
