# Draft Results Section

## 4. Results

### 4.1 Final generalized synthetic benchmark

We first evaluated a compact set of forecasting baselines on the generalized `v3-lite` synthetic benchmark, using `3` observed sessions to forecast horizons `1`, `2`, and `3`. The compared methods were `LOCF`, `UNet-image+mask`, and `ResUNet-image+mask`. Overall, persistence remained a strong baseline, but the residual U-Net variant performed best. Mean Dice was `0.7470` for `LOCF`, `0.7608` for `UNet-image+mask`, and `0.8148` for `ResUNet-image+mask`.

This result is important for two reasons. First, it shows that the benchmark is not trivial: a strong persistence baseline remains competitive, and a standard U-Net only improves modestly over it. Second, the larger gain from `ResUNet` suggests that architectural bias toward residual refinement is useful for longitudinal tumor forecasting under the current setup.

### 4.2 Performance differs by growth regime

To understand whether the aggregate gain was uniform, we analyzed performance across the three benchmark tiers. These tiers are best interpreted as distinct forecasting regimes rather than a single difficulty ladder: tier `A` represents a stability- or persistence-dominated regime, tier `B` a mixed regime, and tier `C` an aggressive-growth stress regime.

The regime-wise results were highly structured. In tier `A`, `LOCF` and `ResUNet-image+mask` were effectively tied (`0.9017` vs `0.9013`), while `UNet-image+mask` was notably worse (`0.7740`). In tier `B`, both learned models outperformed persistence, with mean Dice `0.6783` for `UNet-image+mask`, `0.6827` for `ResUNet-image+mask`, and `0.5903` for `LOCF`. In tier `C`, the gap widened further: `0.8427` for `ResUNet-image+mask`, `0.8325` for `UNet-image+mask`, and `0.7104` for `LOCF`.

These results indicate that learned forecasting does not provide uniform value across all cases. Instead, the benefit appears concentrated in regimes where pure persistence is no longer sufficient. Stable cases remain well handled by `LOCF`, while mixed and aggressive regimes provide substantially more room for learned improvement.

### 4.3 Horizon-wise behavior on synthetic data

We next examined the effect of forecast horizon. On the generalized synthetic benchmark, `ResUNet-image+mask` was best at all tested horizons. At horizon `1`, mean Dice was `0.9141` for `ResUNet-image+mask`, `0.8693` for `LOCF`, and `0.8435` for `UNet-image+mask`. At horizon `2`, the corresponding values were `0.8055`, `0.7146`, and `0.7561`. At horizon `3`, they were `0.7248`, `0.6570`, and `0.6827`.

This pattern shows that the final synthetic benchmark is no longer dominated by a simple immediate-step persistence effect. In other words, the benchmark contains enough nontrivial short-horizon behavior that a learned model can provide gains even at the next step, at least in synthetic settings.

### 4.4 Cross-regime transfer reveals heterogeneous structure

To test whether the tiers capture qualitatively different forecasting regimes, we ran cross-regime transfer experiments on the `biophys1` benchmark. The transfer matrix showed clear asymmetry. Models trained on tier `A` transferred well to tier `A` itself (`0.8631`) but less well to tiers `B` and `C` (`0.6857`, `0.6822`). Models trained on tier `B` transferred well both within tier `B` (`0.8697`) and to tier `C` (`0.8750`), while retaining reasonable performance on tier `A` (`0.7515`). Models trained on tier `C` performed best within tier `C` (`0.8950`) and also transferred well to tier `B` (`0.8374`), but less well to tier `A` (`0.6400`).

This transfer structure supports the interpretation that the benchmark tiers are not merely arranged along a single easy-to-hard axis. Instead, they encode qualitatively different forecasting conditions. Tier `A` behaves like a specialized persistence regime, tier `B` is the most transferable mixed regime, and tier `C` captures aggressive cases that share more with tier `B` than with tier `A`.

### 4.5 Case-level analysis explains where learning helps

We further analyzed pairwise `LOCF` vs `ResUNet` outcomes on `biophys1` to understand when learned forecasting provides value. Among `119` evaluated cases, `ResUNet` produced clear target wins in `63` cases (`52.9%`), both methods were easy/high-performing in `38` cases (`31.9%`), and `LOCF` produced clear wins in only `3` cases (`2.5%`).

The `ResUNet` win cases were not random. They tended to involve larger input tumor volumes, stronger recent growth, stronger future growth, and concentration in tiers `B` and `C`. By contrast, many of the both-easy cases were stable tier `A` examples where persistence remained sufficient.

This analysis suggests that learned gains are most likely when the tumor is actively evolving, rather than when the forecasting task is dominated by persistence. The value of the learned model therefore appears to depend as much on the data regime as on the model architecture itself.

### 4.6 Corrected real-data sanity validation on `SAILOR`

We finally ran a limited real-data sanity evaluation on processed `SAILOR`. This experiment should be interpreted as an external reality check rather than a full real-data benchmark. After correcting label binarization and tensor-shape handling issues in the real-data pipeline, the valid overall results were `0.5159` for `LOCF`, `0.5350` for `ResUNet-image+mask`, and `0.5400` for `ResUNet-mask`.

The horizon-wise pattern was more revealing than the overall means. At horizon `1`, `LOCF` remained slightly stronger (`0.7069`) than both learned models (`0.7022` for `ResUNet-mask`, `0.6932` for `ResUNet-image+mask`). At horizon `2`, the learned models moved ahead (`0.5778`, `0.5770` vs `0.5534` for `LOCF`), and the same remained true at horizon `3` (`0.3402`, `0.3348` vs `0.2875`).

This result is narrower than the synthetic one, but it is highly informative. It supports the original motivation that immediate next-step tumor forecasting can remain strongly persistence-dominated on real data, while also suggesting that learned models become more useful as the forecast horizon increases. Notably, image channels added little in this setting, since `ResUNet-mask` and `ResUNet-image+mask` performed nearly identically.

### 4.7 Summary of findings

Taken together, the experiments support a regime-dependent view of longitudinal tumor forecasting. On synthetic data, learned models, especially `ResUNet`, can outperform persistence and do so most clearly outside the most stable regime. On real data, the story is more conservative: persistence remains extremely strong at the immediate next step, while learned models offer more benefit at longer horizons. The main contribution is therefore not a blanket claim that a deep model beats `LOCF`, but rather a structured empirical account of when persistence is hard to beat, when learned models help, and how benchmark regime design affects those conclusions.
