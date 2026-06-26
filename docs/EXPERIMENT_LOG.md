# Experiment Log

This file is a living research log for `SyntheticTumorBenchmark`.

Its purpose is to capture:

- what experiment was run,
- why it was run,
- the key results,
- how to interpret those results,
- what decision or next step followed.

The goal is to preserve qualitative research reasoning, not only raw metrics.

## Entry Types

This log uses a few lightweight labels for consistency:

- `Experiment`: a concrete run or analysis block
- `Results Note`: a concise result-focused takeaway
- `Discussion Note`: a conceptual interpretation or motivation
- `Hypothesis Update`: a change in what we think is happening
- `Design Decision`: a choice about what to do next or how to scope the work

## Logging Template

For each new run, record:

1. Date
2. Experiment ID
3. Objective
4. Setup
5. Commands
6. Results
7. Interpretation
8. Decision
9. Next steps

---

## 2026-06-21

### Experiment ID

`EXP-001-benchmark-sprint-freeze`

### Objective

Run the first frozen sprint-scale benchmark end-to-end in Colab to verify that:

- dataset generation works,
- the evaluation protocol works,
- official baseline outputs can be produced,
- we have a first benchmark reference point for the paper sprint.

### Setup

- Environment: Google Colab
- Runtime target: GPU-backed Colab session
- Dataset config: `configs/benchmark_sprint.yaml`
- Dataset root:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_sprint_v1`
- Output root:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/baselines_v0p1_sprint`
- Fit sessions: `3`
- Forecast horizons: `1,2`
- UNet epochs: `4`
- Batch size: `2`
- Seed: `42`

### Commands

Dataset generation:

```bash
python scripts/generate_dataset.py \
  --config configs/benchmark_sprint.yaml \
  --output_root "/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_sprint_v1"
```

Baseline pack:

```bash
python scripts/run_all_baselines.py \
  --dataset_root "/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_sprint_v1" \
  --fit_sessions 3 \
  --horizons 1,2 \
  --train_split train \
  --eval_split test \
  --epochs 4 \
  --batch_size 2 \
  --output_dir "/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/baselines_v0p1_sprint"
```

### Results

Summary from `all_baselines_summary.json`:

- `LOCF`
  - `n_samples`: `15`
  - `mean_dice`: `0.6536`
  - `std_dice`: `0.3644`

- `UNet-mask`
  - `n_train_samples`: `83`
  - `n_eval_samples`: `15`
  - `mean_eval_dice`: `0.2443`
  - `std_eval_dice`: `0.2349`

- `UNet-image+mask`
  - `n_train_samples`: `83`
  - `n_eval_samples`: `15`
  - `mean_eval_dice`: `0.4106`
  - `std_eval_dice`: `0.3868`

### Interpretation

This run already tells us several useful things.

1. The benchmark pipeline is operational.
   - The sprint config generated successfully.
   - The baseline runner completed and produced a combined summary file.
   - The benchmark can now be treated as a real experimental artifact rather than only a codebase.

2. `LOCF` is unexpectedly strong.
   - The naive persistence baseline clearly outperformed both learned baselines in the first run.
   - This suggests that short-horizon forecasting in the current benchmark may be dominated by temporal persistence.

3. The learned models are not yet competitive.
   - `UNet-mask` underperformed heavily.
   - `UNet-image+mask` did better than `UNet-mask`, which suggests the synthetic MRI channels contain useful signal, but not yet enough to beat persistence.

4. The benchmark may already have an interesting research story.
   - A simple baseline is hard to beat.
   - Average performance alone may hide important structure by tier or forecast horizon.
   - The large standard deviations suggest heterogeneous case difficulty.

### Hypothesis Update: What the Results Likely Mean

These results do not necessarily mean the learned models are bad in general.
They more likely indicate one or more of the following:

- `4` epochs may be too little for the UNet baselines.
- The sprint dataset is still small.
- Horizon `1` may strongly favor `LOCF`.
- The benchmark dynamics may currently reward persistence more than shape-change forecasting.

### Decision

Do not scale the models yet.

Before increasing training time or changing architecture, first analyze where `LOCF` wins:

- by horizon,
- by tier,
- by individual sample.

This will tell us whether the benchmark has meaningful hard regimes that the averages are hiding.

### Next Steps

1. Load the per-sample output files for all baselines.
2. Build horizon-wise tables for `h=1` and `h=2`.
3. Build tier-wise tables for `Tier A`, `Tier B`, and `Tier C`.
4. Identify hard cases and outliers.
5. Only after that, decide whether to rerun the UNets with more epochs.

### Open Questions

1. Is `LOCF` winning mainly on short-horizon cases?
2. Does `UNet-image+mask` help more on harder tiers than the overall average suggests?
3. Is the current sprint benchmark too easy for learned forecasting methods?
4. Would a longer horizon or harder subset produce a more discriminative benchmark?

---

## 2026-06-22

### Experiment ID

`EXP-001A-breakdown-analysis`

### Objective

Decompose the first sprint benchmark results by:

- forecast horizon,
- synthetic difficulty tier,
- tier + horizon,
- worst individual samples.

The purpose was to determine whether the overall averages were hiding a more meaningful benchmark pattern.

### Setup

- Source run: `EXP-001-benchmark-sprint-freeze`
- Input files:
  - `locf_per_sample.json`
  - `unet_mask_per_sample.json`
  - `unet_image_mask_per_sample.json`
- Output analyses:
  - horizon breakdown
  - tier breakdown
  - tier + horizon breakdown
  - worst-case sample table

### Key Results

#### Horizon breakdown

- `h=1`
  - `LOCF`: `0.631`
  - `UNet-image+mask`: `0.357`
  - `UNet-mask`: `0.210`

- `h=2`
  - `LOCF`: `0.715`
  - `UNet-image+mask`: `0.559`
  - `UNet-mask`: `0.340`

#### Tier breakdown

- `Tier A`
  - `UNet-image+mask`: `0.771`
  - `LOCF`: `0.675`
  - `UNet-mask`: `0.486`

- `Tier B`
  - `LOCF`: `0.750`
  - `UNet-image+mask`: approximately `0`
  - `UNet-mask`: approximately `0`

- `Tier C`
  - `LOCF`: `0.551`
  - `UNet-image+mask`: `0.306`
  - `UNet-mask`: `0.150`

#### Tier + horizon highlights

- `Tier A, h=1`
  - `UNet-image+mask` slightly outperformed `LOCF`

- `Tier A, h=2`
  - `UNet-image+mask` clearly outperformed `LOCF`

- `Tier B`
  - both learned models essentially collapsed across both horizons

- `Tier C`
  - `LOCF` remained strongest, but `UNet-image+mask` showed non-trivial signal on some cases

#### Worst cases

The worst samples were dominated by:

- `Tier B`
- `Tier C`
- mostly learned-model failures with Dice values near zero

### Interpretation

This breakdown substantially improves the benchmark story.

1. The benchmark is not uniformly easy.
   - `Tier A` is learnable.
   - `Tier B` is pathological for the current learned baselines.
   - `Tier C` is difficult but not completely hopeless.

2. The overall average had hidden an important result.
   - `UNet-image+mask` is not simply "bad."
   - It is actually the best method on `Tier A`, especially at the longer horizon.

3. `Tier B` is the main benchmark stress case right now.
   - Both learned models nearly collapsed to zero Dice.
   - `LOCF` remained strong.
   - This suggests either:
     - the tier is genuinely hard in a way that current learning setup cannot capture, or
     - the training budget is too small for this regime.

4. MRI channels matter.
   - `UNet-image+mask` consistently outperformed `UNet-mask`.
   - This supports keeping image-conditioned forecasting in the benchmark.

5. The benchmark may already support a data-mining-style claim.
   - Aggregate metrics are insufficient.
   - Difficulty-tier-aware evaluation changes the ranking and interpretation of methods.

### Decision

Proceed with one controlled rerun before changing the benchmark design:

- keep the same frozen dataset,
- keep the same protocol,
- increase training budget from `4` epochs to `8`.

The goal is to determine whether the Tier B collapse is due to undertraining or reflects a deeper limitation of the current baselines.

### Next Steps

1. Run `run_all_baselines.py` again with `epochs=8` into a new output directory.
2. Recompute the same four analysis tables on the new run.
3. Compare:
   - overall means,
   - tier breakdowns,
   - Tier B recovery or non-recovery,
   - image-conditioned vs mask-only gap.
4. If `Tier B` still collapses, elevate that as a core benchmark finding rather than trying to hide it.

### Hypothesis Update: Working Hypothesis

Current hypothesis after the breakdown analysis:

- `Tier A` behaves like a learnable forecasting regime.
- `Tier B` behaves like a benchmark failure regime for current learned baselines.
- `Tier C` may serve as an intermediate hard regime.

This is promising because it means the synthetic benchmark may already be separating model behavior by controlled growth complexity, which is exactly the kind of structure a useful benchmark should reveal.

---

## 2026-06-22

### Experiment ID

`EXP-002-baselines-e8`

### Objective

Test whether the weak learned-baseline performance in the first sprint run was mainly due to undertraining.

The experiment kept the dataset and protocol fixed and increased the training budget from `4` epochs to `8`.

### Setup

- Source dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_sprint_v1`
- Output root:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/baselines_v0p2_e8`
- Forecast setup:
  - fit sessions: `3`
  - horizons: `1,2`
- Training setup:
  - epochs: `8`
  - batch size: `2`
  - learning rate: `2e-4`
  - seed: `42`

### Results

Reference comparison against the original `4` epoch run:

- `LOCF`
  - unchanged at `0.6536`

- `UNet-mask`
  - `4` epochs: `0.2443`
  - `8` epochs: `0.3508`
  - absolute gain: `+0.1065`

- `UNet-image+mask`
  - `4` epochs: `0.4106`
  - `8` epochs: `0.4138`
  - absolute gain: `+0.0032`

Training logs showed steady improvement for `UNet-mask` across all `8` epochs.
`UNet-image+mask` improved early and then plateaued near `0.41`.

### Interpretation

This rerun gives a much clearer picture of the baseline landscape.

1. Undertraining was a major issue for `UNet-mask`.
   - The mask-only model improved substantially when given more epochs.
   - The original `4` epoch result had underestimated its capability.

2. `UNet-image+mask` appears to plateau early.
   - It improved little from `4` to `8` epochs.
   - This suggests the model may already be near its current regime limit under this setup.

3. `LOCF` remains the strongest overall baseline.
   - Even after the extra training budget, neither learned model surpassed it in overall mean Dice.

4. The benchmark still looks informative rather than trivial.
   - The learned baselines are sensitive to training budget.
   - Simple persistence remains very hard to beat.
   - This is exactly the kind of behavior a benchmark should surface.

### Decision

Do not change the benchmark yet.

First compute the same breakdown analysis for the `8` epoch run:

- by horizon,
- by tier,
- by tier + horizon,
- worst cases.

That will tell us whether:

- `Tier B` remains a failure regime,
- `UNet-mask` has recovered meaningfully in any tier,
- `UNet-image+mask` still dominates `Tier A`,
- extra training changes the benchmark narrative or only the averages.

### Next Steps

1. Load per-sample outputs from `baselines_v0p2_e8`.
2. Recompute all breakdown tables.
3. Compare `EXP-001A` and `EXP-002` side by side.
4. Decide whether the next experiment should be:
   - `epochs=12`, or
   - a benchmark analysis figure/table pass instead of more training.

---

## 2026-06-22

### Experiment ID

`EXP-002A-breakdown-analysis-e8`

### Objective

Analyze the `8` epoch rerun by horizon, tier, tier + horizon, and worst-case failures to determine whether extra training changes the benchmark interpretation.

### Key Results

#### Horizon breakdown

- `h=1`
  - `LOCF`: `0.631`
  - `UNet-image+mask`: `0.360`
  - `UNet-mask`: `0.304`

- `h=2`
  - `LOCF`: `0.715`
  - `UNet-image+mask`: `0.562`
  - `UNet-mask`: `0.479`

#### Tier breakdown

- `Tier A`
  - `UNet-image+mask`: `0.779`
  - `UNet-mask`: `0.684`
  - `LOCF`: `0.675`

- `Tier B`
  - `LOCF`: `0.750`
  - `UNet-mask`: approximately `0`
  - `UNet-image+mask`: approximately `0`

- `Tier C`
  - `LOCF`: `0.551`
  - `UNet-image+mask`: `0.306`
  - `UNet-mask`: `0.231`

#### Tier + horizon highlights

- `Tier A, h=1`
  - `UNet-image+mask` remains best
  - `UNet-mask` is now close behind

- `Tier A, h=2`
  - both learned models now outperform `LOCF`

- `Tier B`
  - both learned models still collapse at both horizons

- `Tier C, h=2`
  - both learned models produce meaningful gains relative to their earlier run
  - `LOCF` still remains strongest

### Interpretation

This analysis strengthens the benchmark story considerably.

1. Extra training helps, but selectively.
   - `UNet-mask` improved substantially, especially on `Tier A` and some `Tier C` cases.
   - `UNet-image+mask` remained largely stable, suggesting it had already saturated under this setup.

2. `Tier A` is now clearly a learnable regime.
   - Both learned models are competitive there.
   - At `h=2`, both learned models outperform `LOCF`.

3. `Tier B` remains the defining failure regime.
   - Extra training did not rescue either learned baseline.
   - This strongly suggests `Tier B` is exposing a real modeling weakness rather than simple undertraining.

4. `Tier C` behaves like an intermediate hard regime.
   - The learned methods are not collapsing completely.
   - They improve with more training but still trail the persistence baseline.

5. The benchmark now shows meaningful regime separation.
   - easy regime: `Tier A`
   - failure regime: `Tier B`
   - intermediate hard regime: `Tier C`

This is a strong benchmark property and is exactly the type of structure that makes a synthetic testbed valuable.

### Decision

Do not change the dataset yet.

The benchmark is already revealing controlled difficulty structure. The next work should focus on:

- one more modest training-budget test if needed, and
- packaging the tier-based findings into tables and figures.

### Design Decision: Recommended Next Step

The highest-value next experiment is no longer another large redesign.
It is a controlled decision point:

1. either run `epochs=12` once to confirm `Tier B` is still a failure regime,
2. or stop model training and move into benchmark packaging and figure generation.

### Hypothesis Update: Current Working Claim

The current evidence supports the following benchmark-oriented claim:

`SyntheticTumorBenchmark` separates forecasting methods across controlled growth regimes, revealing that:

- simple persistence is a strong baseline overall,
- image-conditioned learning helps on easier regimes,
- some mechanistically harder regimes remain challenging even after additional training.

---

## 2026-06-22

### Experiment ID

`EXP-003-baselines-e12-seeds`

### Objective

Run a longer training-budget test with `12` epochs and two seeds to estimate:

- whether extra training continues to improve the learned baselines,
- whether the model ranking is stable across seeds,
- whether `LOCF` still remains the strongest overall reference baseline.

### Setup

- Source dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_sprint_v1`
- Protocol:
  - fit sessions: `3`
  - horizons: `1,2`
  - train split: `train`
  - eval split: `test`
- Training:
  - epochs: `12`
  - batch size: `2`
  - learning rate: `2e-4`
- Seeds:
  - `42`
  - `123`

### Results

#### Seed 42

- `LOCF`: `0.6536`
- `UNet-mask`: `0.3742`
- `UNet-image+mask`: `0.4189`

Training behavior:

- `UNet-mask` improved steadily through epoch `12`.
- `UNet-image+mask` improved early, peaked around epoch `9`, and then softened slightly.

#### Seed 123

- `LOCF`: `0.6536`
- `UNet-mask`: `0.4663`
- `UNet-image+mask`: `0.3901`

Training behavior:

- `UNet-mask` showed a late jump, reaching its best evaluation Dice at epoch `12`.
- `UNet-image+mask` was less stable and did not surpass the seed `42` result.

### Interpretation

This experiment reveals that seed sensitivity is now a major part of the benchmark story.

1. Extra training still helps, especially for `UNet-mask`.
   - `UNet-mask` improved from:
     - `0.2443` at `4` epochs,
     - to `0.3508` at `8` epochs,
     - to `0.3742` or `0.4663` at `12` epochs depending on seed.

2. The learned-model ranking is not stable across seeds.
   - Seed `42`: `UNet-image+mask` > `UNet-mask`
   - Seed `123`: `UNet-mask` > `UNet-image+mask`

3. `LOCF` still remains the strongest overall baseline.
   - Neither learned model surpassed `0.6536`.
   - This makes `LOCF` a genuinely strong benchmark reference rather than just a trivial placeholder.

4. The benchmark is exposing both regime difficulty and optimization instability.
   - Earlier experiments showed tier-specific separation.
   - This experiment adds evidence that the learned methods are also sensitive to optimization randomness on the small sprint-scale benchmark.

### Decision

Stop increasing epochs for now.

The returns from more training are no longer clean enough to justify chasing larger runs before packaging what we already learned.

The next effort should go into:

- extracting the same tier/horizon breakdowns for both `12`-epoch runs,
- summarizing seed variability,
- turning the existing results into benchmark tables and figures.

### Next Steps

1. Compute tier/horizon breakdowns for:
   - `baselines_v0p3_e12_s42`
   - `baselines_v0p3_e12_s123`
2. Compare seed stability by:
   - overall mean Dice,
   - tier-wise ranking,
   - whether `Tier B` still collapses.
3. Prepare a paper-ready summary table with:
   - `LOCF`
   - `UNet-mask`
   - `UNet-image+mask`
   - results at `4`, `8`, and `12` epochs
4. Start drafting the benchmark claims and limitations section.

### Hypothesis Update: Current Working Claim

The benchmark now appears to support three layers of analysis:

1. **difficulty-tier structure**
   - `Tier A` learnable
   - `Tier B` failure regime
   - `Tier C` intermediate-hard regime

2. **baseline hierarchy**
   - `LOCF` is a strong overall reference
   - learned models can win in easier regimes but not overall

3. **training instability**
   - learned baselines are sensitive to epoch budget and random seed
   - aggregate benchmark claims should therefore include stability caveats

### Discussion Note: Why LOCF Is So Strong

An important interpretation emerging from both this benchmark and related lab experience is that immediate next-step tumor forecasting can strongly favor persistence-based baselines such as `LOCF`.

The likely reason is that, from one session to the next, the tumor mask often changes only slightly. In that setting, a learned model is not really learning the full future tumor shape from scratch. Instead, it must learn a small residual evolution on top of a very strong persistence prior.

This is a difficult setting for learned models because:

- the target is already very close to the most recent observed mask,
- the clinically or visually meaningful change may be small relative to the full volume,
- improvements over `LOCF` may depend on subtle boundary evolution,
- limited sample size makes those residual changes harder to learn robustly.

This helps explain why:

- `LOCF` remains the strongest overall baseline,
- learned models may still be useful in selected regimes or longer horizons,
- immediate next-point overlap metrics alone may understate the value of more expressive models.

### Discussion Note: Short-Term Forecasting as the Right Benchmark Scope

An important refinement to the project framing is that this benchmark is currently best understood as a **short-term longitudinal tumor forecasting benchmark**, not a fully general tumor-trajectory benchmark.

This matters because the current task design is naturally aligned with near-horizon prediction:

- a limited number of observed sessions,
- immediate future target sessions,
- strong temporal persistence,
- high relevance of small residual changes.

Under this framing, the current results make more sense and become more useful:

- lightweight or persistence-based methods are expected to be highly competitive,
- image-conditioned models may help in selected regimes,
- heavier generative or long-range forecasting models may be better reserved for future benchmark extensions over longer horizons.

This also provides a practical application split:

- short-term forecasting methods may be more suitable for frequent clinical monitoring or near-term follow-up support,
- heavier deep generative models may be more appropriate for medium- or long-horizon exploratory studies.

As a result, the benchmark's current contribution is strongest when positioned around:

- immediate next-step or near-term tumor mask forecasting,
- controlled regime-aware comparison,
- understanding when simple persistence is enough and when learned models add value.

### Discussion Note: Residual Forecasting With Multi-Session History

As the benchmark story became clearer, the most promising next modeling direction was refined from two separate ideas into one unified family:

- residual forecasting, and
- multi-session conditioning.

Instead of treating these as unrelated extensions, the better formulation is to define a **residual short-term forecasting baseline with configurable history length**.

The motivating idea is:

1. short-term tumor forecasting is strongly persistence-dominated,
2. the future mask is often close to the most recent observed mask,
3. a learned model may therefore be better framed as learning the **residual change** rather than the full future mask,
4. using more than one recent session can make the task more genuinely longitudinal.

For the current sprint, the discussion settled on:

- start with `k = 3` past sessions,
- do not jump immediately to larger history lengths such as `7` or `10`,
- treat larger `k` as a future extension for richer or longer generated datasets.

The main reasons for choosing `k = 3` now are:

- it matches the current `fit_sessions = 3` benchmark setting,
- it keeps the implementation feasible in the current codebase,
- it preserves sample availability,
- it is enough to test whether recent longitudinal history improves short-term forecasting.

This means the next candidate model is best described as:

- **Residual short-term forecasting with 3-session history**

rather than as two separate experiments.

### Design Decision: Adaptive Residual Experiment Ladder

The next model-development phase should be handled adaptively rather than through a large fixed grid.

The chosen approach is to define a compact residual-history experiment ladder:

- `R1`: residual-history prototype on the sprint dataset,
- `R2`: residual-history prototype on the medium dataset,
- `R3`: history-length ablation (`k=1` vs `k=3`),
- `R4`: small seed-stability check on the best residual variant.

The purpose of this ladder is to let each result determine the next branch rather than committing to unnecessary runs in advance.

The central model family is:

- **Residual short-term forecasting with multi-session history**

starting with:

- input mode: `image_mask`
- history length: `k = 3`
- short-term horizons only

This is intended to test whether:

- learning residual change over a strong persistence prior helps,
- explicitly using recent history improves short-term forecasting,
- any gains survive when moved from sprint to medium evidence tracks.

### Discussion Note: Why Immediate Forecasting Deserves Separate Attention

An important motivation emerged from repeated observations across different modeling settings:

- simple `LOCF` baselines can beat more sophisticated methods at the immediate next-step forecasting task,
- this has been observed not only in the current U-Net benchmark runs, but also anecdotally in related PDE-solver and flow-model work on tumor forecasting.

This suggests that **immediate short-term forecasting is a distinct regime inside the broader tumor-growth forecasting problem**.

The broader literature often studies longitudinal prediction across follow-up periods that may span weeks, months, or longer. In those settings, results are often reported across mixed horizons. However, the immediate next-step case appears qualitatively different because:

- the future mask is often very close to the current one,
- the dominant signal is persistence,
- the true learning target is often a small residual shape change,
- overlap-based metrics such as Dice strongly reward copy-forward behavior when changes are subtle.

This means that short-term immediate forecasting should not automatically be treated as equivalent to general longer-horizon tumor forecasting.

Instead, the current project increasingly supports the following view:

- **immediate short-term tumor forecasting is a persistence-dominated subproblem**
- this subproblem deserves explicit benchmark treatment
- methods should be evaluated not just by overall future-prediction performance, but by how they behave when horizons are short and residual change is small

This is part of the core motivation for the benchmark and for the proposed residual-history modeling direction.

### Discussion Note: Probabilistic Fields vs Deterministic Short-Term Forecasting

Another conceptual direction emerged while interpreting why `LOCF` is so strong for immediate forecasting.

The current forecasting setup evaluates deterministic future mask prediction under overlap metrics such as Dice. However, for very short-term tumor evolution, the true change may be:

- spatially small,
- uncertain at the boundary,
- better described as a likely region or direction of growth rather than a single sharply defined binary outcome.

This suggests a potentially richer framing:

- instead of only predicting a future binary mask,
- one could predict a **probability field**, **uncertainty map**, or even a **growth-oriented spatial field** indicating where change is most likely to occur.

This idea is not disconnected from existing literature. Related work already uses:

- uncertainty-aware future tumor predictions,
- signed-distance-function or neural-field representations of future tumor shape,
- deformation or temporal field modules for modeling future evolution.

What appears less standard, and potentially valuable for this project, is to connect that style of representation specifically to the **short-term persistence-dominated regime**:

- where `LOCF` is hard to beat on binary overlap,
- but deterministic mask prediction may underrepresent the uncertainty or directional ambiguity of immediate growth.

This means a useful longer-term question for the benchmark is not only:

- can a model beat `LOCF` in Dice?

but also:

- can a model produce a better calibrated spatial estimate of where short-term growth is likely to occur?

For the current DMS sprint, this remains a conceptual extension rather than the main implementation target. The immediate priority is still the residual-history baseline. However, this probabilistic-field viewpoint may become an important future direction for both benchmark design and model outputs.

### Results Note: Current Insight Summary

At the present stage of the project, the strongest insights are:

1. `LOCF` is a genuinely strong baseline, not a trivial placeholder.
2. The benchmark behavior depends strongly on horizon.
3. The benchmark behavior depends strongly on synthetic regime/tier.
4. Image-conditioned learned models are generally more stable than mask-only learned models.
5. Small pilot datasets can overstate collapse or instability; medium-scale evidence is more reliable.
6. Immediate next-step forecasting appears qualitatively different from slightly longer short-term forecasting.
7. This makes short-term longitudinal forecasting a valid and interesting data-mining problem in its own right.

---

## 2026-06-22

### Experiment ID

`EXP-004-seed-sweep-e12-25`

### Objective

Quantify optimization randomness more systematically by running the same `12`-epoch baseline setup across `25` seeds on the frozen sprint dataset.

The goal was to determine:

- whether the learned baselines are stable on average,
- whether one learned baseline is consistently stronger,
- whether the strong `LOCF` result remains meaningfully ahead of learned models,
- whether seed sensitivity is a minor nuisance or a central benchmark property.

### Setup

- Dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_sprint_v1`
- Protocol:
  - fit sessions: `3`
  - horizons: `1,2`
  - train split: `train`
  - eval split: `test`
- Training:
  - epochs: `12`
  - batch size: `2`
- Seeds:
  - `1` through `25`

### Aggregate Results

- `LOCF`
  - mean Dice: `0.6536`

- `UNet-mask`
  - seed count: `25`
  - mean Dice across seeds: `0.3964`
  - std across seeds: `0.1434`
  - min: `0.0350`
  - max: `0.6802`

- `UNet-image+mask`
  - seed count: `25`
  - mean Dice across seeds: `0.4108`
  - std across seeds: `0.0567`
  - min: `0.3501`
  - max: `0.6759`

### Interpretation

This is one of the most important benchmark results so far.

1. `LOCF` remains a very strong overall reference.
   - Its mean Dice is still well above the average learned-model performance.
   - This confirms that immediate short-term persistence is a hard baseline to beat.

2. `UNet-image+mask` is more stable than `UNet-mask`.
   - Its average performance is slightly higher.
   - Its seed-to-seed variance is much lower.
   - This suggests image-conditioned forecasting is the more reliable learned baseline under the current setup.

3. `UNet-mask` is much more volatile.
   - It ranges from near-collapse (`0.035`) to near-LOCF-level performance (`0.680`).
   - This shows strong optimization sensitivity and/or data-regime sensitivity.

4. The learned-model ranking is not fixed for every seed.
   - In some seeds, `UNet-mask` performs surprisingly well.
   - In others, it collapses badly.
   - This reinforces that single-seed reporting would be misleading for this benchmark.

5. The benchmark is revealing both persistence effects and optimization instability.
   - This is a meaningful data-mining result, not just noise.
   - A useful benchmark should expose where methods are fragile, not only where they succeed.

### Results Note: Practical Conclusion

For the current short-term forecasting benchmark:

- `LOCF` is the strongest overall baseline,
- `UNet-image+mask` is the more stable learned model,
- `UNet-mask` can sometimes perform very well but is not consistently reliable.

### Decision

We now have enough evidence to justify reporting seed sensitivity explicitly in the paper.

From this point onward, additional training sweeps should be run only if they support a clearly missing claim. The benchmark story is already strong enough to move into figure, table, and writing mode.

### Design Decision: Suggested Next Steps

1. Build a paper-ready summary table covering:
   - `LOCF`
   - `UNet-mask`
   - `UNet-image+mask`
   - results at `4`, `8`, `12` epochs
   - `25`-seed `12`-epoch aggregate statistics

2. Add tier- and horizon-wise analysis for a representative subset or aggregate seed results where feasible.

3. Run one final targeted experiment only if it answers a specific open question, such as:
   - modestly longer horizon (`1,2,3`)
   - delta-days grouped performance

### Hypothesis Update: Working Claim Strengthened by This Sweep

The benchmark supports a short-term forecasting narrative in which:

- simple persistence is a very strong reference point,
- learned models can be competitive but are regime- and optimization-dependent,
- image-conditioned learning is more stable than mask-only learning under the current setup.

---

## 2026-06-22

### Experiment ID

`EXP-005-horizon-extension-h123`

### Objective

Test whether slightly extending the forecasting task from horizons `1,2` to `1,2,3` changes the relative behavior of persistence and learned baselines.

The main question was whether learned methods gain a clearer relative advantage as the forecast target moves farther from the conditioning session.

### Setup

- Dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_sprint_v1`
- Protocol:
  - fit sessions: `3`
  - horizons: `1,2,3`
  - train split: `train`
  - eval split: `test`
- Training:
  - epochs: `12`
  - batch size: `2`
  - seed: `42`
- Output root:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/baselines_v0p4_h123_s42`

### Results

#### Overall summary

- `LOCF`
  - `n_samples`: `18`
  - mean Dice: `0.6487`

- `UNet-mask`
  - `n_eval_samples`: `18`
  - mean Dice: `0.3746`

- `UNet-image+mask`
  - `n_eval_samples`: `18`
  - mean Dice: `0.4209`

#### Horizon breakdown

- `h=1`
  - `LOCF`: `0.6312`
  - `UNet-image+mask`: `0.3556`
  - `UNet-mask`: `0.3032`

- `h=2`
  - `LOCF`: `0.7151`
  - `UNet-image+mask`: `0.5837`
  - `UNet-mask`: `0.5139`

- `h=3`
  - `LOCF`: `0.6242`
  - `UNet-mask`: `0.4508`
  - `UNet-image+mask`: `0.4434`

### Interpretation

This horizon-extension experiment is informative, but it does not overturn the benchmark story.

1. `LOCF` remains strongest at all tested horizons.
   - Even when the horizon is extended modestly to `3`, persistence is still the best-performing overall baseline.

2. Learned models do close the gap at longer horizons.
   - At `h=2` and `h=3`, both learned methods move closer to `LOCF` than they are at `h=1`.
   - This supports the idea that learned forecasting becomes relatively more useful as the target moves farther away from pure persistence.

3. The gain is relative, not absolute.
   - The learned models improve their competitiveness at larger horizons, but they do not yet surpass `LOCF` in this sprint benchmark setting.

4. The benchmark still behaves like a short-term forecasting testbed.
   - Extending the horizon from `1,2` to `1,2,3` is not yet enough to transform the problem into one where learned models dominate.
   - Instead, it reinforces the view that the benchmark currently lives in a near-term forecasting regime with a strong persistence prior.

### Decision

Stop pursuing further modest horizon extensions for now.

This experiment gave the directional answer we needed:

- learned models gain relative ground as horizon increases,
- but the benchmark still remains persistence-dominated in the tested short-term range.

The next work should focus on packaging this result into the paper rather than launching more small horizon variants.

### Next Steps

1. Build a paper-ready table summarizing:
   - `LOCF`
   - `UNet-mask`
   - `UNet-image+mask`
   - performance at `h=1`, `h=2`, `h=3`

2. Add a short discussion framing:
   - immediate next-step forecasting strongly favors persistence,
   - learned methods become relatively more competitive as horizon grows,
   - but the tested benchmark still reflects a short-term monitoring regime.

3. Move toward figure generation and writing unless one clearly missing analysis remains.

---

## 2026-06-22

### Experiment ID

`EXP-006-medium-baselines-v0p1`

### Objective

Move from sprint-scale pilot evidence to medium-scale benchmark evidence using a larger frozen dataset while keeping the same short-term forecasting protocol and baseline suite.

### Setup

- Dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_medium_v1`
- Config:
  `configs/benchmark_medium.yaml`
- Protocol:
  - fit sessions: `3`
  - horizons: `1,2`
  - train split: `train`
  - eval split: `test`
- Training:
  - epochs: `12`
  - batch size: `2`
  - seed: `42`
- Output root:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/medium_baselines_v0p1`

### Results

- `LOCF`
  - `n_samples`: `30`
  - mean Dice: `0.7087`
  - std Dice: `0.2103`

- `UNet-mask`
  - `n_train_samples`: `150`
  - `n_eval_samples`: `30`
  - mean Dice: `0.6273`
  - std Dice: `0.2073`

- `UNet-image+mask`
  - `n_train_samples`: `150`
  - `n_eval_samples`: `30`
  - mean Dice: `0.6788`
  - std Dice: `0.2445`

### Interpretation

This is a strong scaling result.

1. The core benchmark story survives scaling.
   - `LOCF` remains the strongest overall baseline.
   - `UNet-image+mask` remains the stronger learned model.
   - `UNet-mask` remains competitive but weaker on average.

2. The learned models are substantially stronger on the medium dataset than on the sprint pilot.
   - This suggests the sprint-scale runs were indeed too small to stand alone as main paper evidence.
   - The larger dataset gives a much more credible experimental basis.

3. The gap to `LOCF` narrows meaningfully at medium scale.
   - `UNet-image+mask` now approaches the persistence baseline much more closely.
   - This makes the short-term forecasting comparison more interesting and more publication-worthy.

4. The benchmark is no longer only a pilot.
   - With `150` training samples and `30` evaluation samples, the evidence is still modest, but much less fragile than the sprint setting.

### Decision

Use the medium dataset as the main evidence track going forward.

The sprint dataset should now be treated as:

- pilot evidence,
- debugging evidence,
- fast hypothesis-generation evidence.

### Next Steps

1. Compute per-tier and per-horizon breakdowns on the medium dataset.
2. Check whether the same `Tier A / Tier B / Tier C` structure survives scaling.
3. If it does, prioritize:
   - medium seed sweep,
   - medium horizon-extension,
   - then residual-history modeling work.

---

## 2026-06-22

### Experiment ID

`EXP-006A-medium-breakdown-analysis`

### Objective

Analyze the medium-scale benchmark results by:

- forecast horizon,
- synthetic difficulty tier,
- overall regime behavior after scaling.

### Key Results

#### Horizon breakdown

- `h=1`
  - `LOCF`: `0.7621`
  - `UNet-image+mask`: `0.6044`
  - `UNet-mask`: `0.5758`

- `h=2`
  - `UNet-image+mask`: `0.7903`
  - `UNet-mask`: `0.7047`
  - `LOCF`: `0.6286`

#### Tier breakdown

- `Tier A`
  - `UNet-image+mask`: `0.7597`
  - `UNet-mask`: `0.7329`
  - `LOCF`: `0.6397`

- `Tier B`
  - `LOCF`: `0.7151`
  - `UNet-image+mask`: `0.6341`
  - `UNet-mask`: `0.5831`

- `Tier C`
  - `LOCF`: `0.9307`
  - `UNet-image+mask`: `0.5296`
  - `UNet-mask`: `0.3905`

### Interpretation

This medium-scale breakdown meaningfully sharpens the benchmark story.

1. The benchmark is now clearly horizon-dependent.
   - At `h=1`, persistence remains strongest by a clear margin.
   - At `h=2`, both learned models outperform `LOCF`, with `UNet-image+mask` taking a substantial lead.

2. The benchmark is still regime-dependent, but differently than in the sprint pilot.
   - `Tier A` remains clearly learnable and now both learned models outperform `LOCF`.
   - `Tier B` is no longer a total learned-model collapse regime at medium scale.
   - `Tier C` is strongly persistence-dominated, though the sample count there is small and should be interpreted cautiously.

3. Scaling changed the interpretation in an important way.
   - The sprint dataset made `Tier B` appear pathological for learned models.
   - The medium dataset suggests that this earlier finding was at least partly a small-data or underpowered-benchmark effect.
   - This reinforces why the medium dataset should be treated as the main evidence track.

4. The short-term forecasting framing is strengthened.
   - Immediate next-step forecasting (`h=1`) strongly favors `LOCF`.
   - Slightly farther short-term forecasting (`h=2`) allows learned models, especially image-conditioned ones, to add clear value.

### Results Note: Practical Conclusion

The medium benchmark now supports a much more nuanced and publication-worthy claim:

- persistence is strongest for very immediate forecasting,
- learned models become advantageous at slightly longer short-term horizons,
- image-conditioned forecasting is the strongest learned baseline,
- benchmark conclusions depend strongly on both horizon and regime.

### Caution

`Tier C` currently has only `4` evaluation samples in this run, so any claims about that tier should be treated as directional rather than definitive until more evidence is collected.

### Next Steps

1. Run a medium-scale seed sweep at `epochs=12`.
2. Run a medium-scale `horizons=1,2,3` extension.
3. Use the medium dataset as the base for residual-history modeling experiments.

---

## 2026-06-22

### Experiment ID

`EXP-006B-medium-seed-sweep-e12-5`

### Objective

Measure seed stability on the medium dataset to determine whether the stronger benchmark story observed at medium scale is robust or still heavily optimization-sensitive.

### Setup

- Dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_medium_v1`
- Protocol:
  - fit sessions: `3`
  - horizons: `1,2`
  - train split: `train`
  - eval split: `test`
- Training:
  - epochs: `12`
  - batch size: `2`
- Seeds:
  - `7, 21, 42, 123, 999`

### Aggregate Results

- `LOCF`
  - mean Dice: `0.7087`

- `UNet-mask`
  - seed count: `5`
  - mean Dice across seeds: `0.6471`
  - std across seeds: `0.0525`
  - min: `0.5960`
  - max: `0.7409`

- `UNet-image+mask`
  - seed count: `5`
  - mean Dice across seeds: `0.6547`
  - std across seeds: `0.0167`
  - min: `0.6295`
  - max: `0.6788`

### Interpretation

This medium-scale seed sweep is a major strengthening result.

1. The medium benchmark is much more stable than the sprint benchmark.
   - Both learned models show far less variance than in the sprint seed sweep.
   - This confirms that the sprint instability was at least partly a small-data effect.

2. `UNet-image+mask` remains the more stable learned baseline.
   - Its average performance is slightly higher than `UNet-mask`.
   - Its seed variance is dramatically lower.

3. `UNet-mask` is still somewhat more variable, but no longer wildly unstable.
   - It can even exceed `LOCF` for some seeds.
   - However, on average it remains below both `LOCF` and `UNet-image+mask`.

4. `LOCF` is still the strongest overall average baseline for `horizons=1,2`.
   - The gap is now much smaller than in the sprint setting.
   - This makes the benchmark comparison scientifically more interesting rather than trivial.

### Results Note: Practical Conclusion

At medium scale, the benchmark supports a more credible and stable short-term forecasting story:

- `LOCF` is still the strongest overall average baseline,
- `UNet-image+mask` is the strongest and most stable learned model,
- `UNet-mask` is competitive and occasionally very strong, but somewhat less reliable.

### Design Decision: Suggested Next Steps

1. Treat medium-scale seed stability evidence as paper-ready.
2. Use the medium dataset for the main benchmark claims.
3. Move the next innovation experiment onto the medium track once it is validated on the sprint track.

### Hypothesis Update: Working Claim Strengthened by This Sweep

At medium scale:

- persistence remains strong,
- image-conditioned learning is stable,
- mask-only learning is viable but somewhat less robust,
- the benchmark is no longer dominated by small-sample noise.

---

## 2026-06-22

### Experiment ID

`EXP-006C-medium-horizon-extension-h123`

### Objective

Test whether the horizon-dependent story observed on the medium dataset continues or strengthens when the task is extended from `horizons=1,2` to `horizons=1,2,3`.

### Setup

- Dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_medium_v1`
- Protocol:
  - fit sessions: `3`
  - horizons: `1,2,3`
  - train split: `train`
  - eval split: `test`
- Training:
  - epochs: `12`
  - batch size: `2`
  - seed: `42`

### Results

#### Overall summary

- `LOCF`
  - `n_samples`: `40`
  - mean Dice: `0.6507`

- `UNet-mask`
  - `n_eval_samples`: `40`
  - mean Dice: `0.6475`

- `UNet-image+mask`
  - `n_eval_samples`: `40`
  - mean Dice: `0.7024`

#### Horizon breakdown

- `h=1`
  - `LOCF`: `0.7621`
  - `UNet-image+mask`: `0.6179`
  - `UNet-mask`: `0.5868`

- `h=2`
  - `UNet-image+mask`: `0.8045`
  - `UNet-mask`: `0.6964`
  - `LOCF`: `0.6286`

- `h=3`
  - `UNet-image+mask`: `0.7319`
  - `UNet-mask`: `0.6979`
  - `LOCF`: `0.4769`

#### Tier breakdown

- `Tier A`
  - `UNet-image+mask`: `0.7314`
  - `UNet-mask`: `0.7196`
  - `LOCF`: `0.5646`

- `Tier B`
  - `UNet-image+mask`: `0.7058`
  - `LOCF`: `0.6828`
  - `UNet-mask`: `0.6226`

- `Tier C`
  - `LOCF`: `0.8991`
  - `UNet-image+mask`: `0.5756`
  - `UNet-mask`: `0.4338`

### Interpretation

This is one of the strongest results in the project so far.

1. The horizon story is now very clear.
   - At `h=1`, `LOCF` is strongly best.
   - At `h=2`, both learned models beat `LOCF`, with `UNet-image+mask` leading clearly.
   - At `h=3`, both learned models beat `LOCF` by an even larger margin.

2. This strongly supports the central short-term forecasting hypothesis.
   - Immediate next-step forecasting is persistence-dominated.
   - Slightly longer short-term horizons increasingly reward learned forecasting.

3. `UNet-image+mask` is now the strongest learned model overall.
   - On `horizons=1,2,3` combined, it exceeds `LOCF` overall mean Dice.
   - This is an important benchmark result and a meaningful nontrivial contribution.

4. Regime dependence still matters.
   - `Tier A` strongly favors learned methods.
   - `Tier B` is now competitive, with image-conditioned learning slightly ahead of `LOCF`.
   - `Tier C` remains strongly persistence-dominated and still looks like the hardest regime for learned forecasting under the current setup.

### Results Note: Practical Conclusion

The medium benchmark now supports a very strong paper-level claim:

- persistence dominates immediate short-term tumor forecasting,
- learned models become clearly advantageous as horizon moves slightly farther out,
- image-conditioned forecasting is the most effective and stable learned baseline in this benchmark.

### Design Decision: Suggested Next Steps

1. Treat the medium horizon-extension result as a core paper figure/table.
2. Use this result as the main motivation for the residual-history modeling experiment.
3. Stop running broad baseline sweeps and move to targeted innovation experiments plus writing support.

### Hypothesis Update: Working Claim Strengthened by This Experiment

The benchmark is now best framed as a short-term longitudinal forecasting testbed in which:

- `h=1` is persistence-dominated,
- `h=2` and `h=3` increasingly reward learned models,
- image-conditioned models provide the strongest learned baseline,
- regime complexity still determines where persistence remains difficult to beat.

### Discussion Note: Probability Fields and Growth-Direction Framing

Another useful conceptual refinement emerged while interpreting the immediate short-term regime.

If `LOCF` is hard to beat at the next-step forecast, one likely reason is that the true change is often very small relative to:

- the full brain volume,
- the current tumor extent,
- and the spatial scale emphasized by overlap metrics such as Dice.

This suggests that immediate forecasting may not always be best understood as only:

- "predict the one correct next binary mask"

but also as:

- "estimate where change is most likely to occur,"
- "estimate how certain that change is,"
- and possibly "estimate the most likely local direction of boundary evolution."

In that sense, a future model could produce outputs such as:

- a voxelwise probability field for future occupancy,
- a boundary uncertainty map,
- a residual change likelihood map,
- or a local growth-direction / displacement-style field near the tumor boundary.

This framing is appealing because it matches the practical difficulty of the task:

- most of the tumor stays the same,
- only a relatively small boundary region may change,
- and the clinically relevant question may be less about exact binary overlap and more about where plausible progression is concentrated.

This does **not** replace the current benchmark direction. For the DMS sprint, the main target remains strong deterministic short-term forecasting baselines, especially residual-history models. But this idea is worth keeping in the project narrative because it offers a principled explanation for why persistence is so dominant and why a pure Dice-based deterministic forecast may understate model value in the immediate regime.

### Design Decision: Scope of This Idea

For now, treat the probability-field / growth-direction idea as:

1. a conceptual interpretation layer for the benchmark,
2. a future modeling direction,
3. and a possible future evaluation extension beyond hard-mask Dice.

Do not expand scope immediately into a new probabilistic modeling project until the current residual short-term experiments are fully characterized.

### Design Decision: SAILOR as Focused Real-Data Validation

An important scope clarification was made once it was confirmed that a processed `SAILOR` dataset is already available.

This changes the longer-term paper package in a useful way:

- the synthetic benchmark remains the main controlled experimental testbed,
- while `SAILOR` can later serve as a focused real-data validation layer.

The intended role of `SAILOR` is **not** to become a second full benchmark track during the current evidence-collection phase. Instead, its best use is to test whether the main short-term forecasting pattern observed on synthetic data also appears on real longitudinal cases.

The specific validation goal would be modest and targeted:

1. run `LOCF` on the real short-term setup,
2. run the strongest learned baseline,
3. if ready, run the residual-history baseline,
4. check whether the same qualitative pattern appears:
   - immediate `h=1` forecasting favors persistence,
   - slightly longer short-term horizons increasingly favor learned models.

This is strategically valuable because it would let the project claim:

- controlled evidence from synthetic data,
- plus limited realism/credibility support from real data.

### Design Decision: Scope Protection for SAILOR

For the current DMS sprint, `SAILOR` should be treated as:

1. a later validation layer,
2. a minimal extension after the synthetic story is clean,
3. not a parallel project that disrupts the current benchmark and residual experiment schedule.

This keeps the paper focused while preserving a strong path to improved external credibility.

---

## 2026-06-23

### Experiment ID

`EXP-007-residual-overnight-ablation`

### Objective

Run a focused overnight residual-model ablation on the medium dataset to answer three questions:

- does residual forecasting remain stronger than the earlier learned baselines,
- does recent history length `k=3` actually help over `k=1`,
- how sensitive is the residual model to the persistence prior and random seed.

### Setup

- Dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_medium_v1`
- Protocol:
  - `fit_sessions = 3`
  - `horizons = 1,2,3`
  - `input_mode = image_mask`
  - `epochs = 12`
  - `batch_size = 2`
- Sweep dimensions:
  - history length: `k in {1, 3}`
  - prior strength: `{2.0, 4.0, 6.0}`
  - seeds:
    - targeted ablations at `seed=42`
    - stability check for `k=3, prior=4.0` with `seeds = {21, 42, 123, 999}`

### Key Results

#### Overall ranking

- `R8_medium_h123_k1_p4_s42`
  - `mean_eval_dice = 0.7986`
- `R10_medium_h123_k3_p4_s123`
  - `mean_eval_dice = 0.7835`
- `R6_medium_h123_k3_p4_s42`
  - `mean_eval_dice = 0.7642`
- `R7_medium_h123_k3_p6_s42`
  - `mean_eval_dice = 0.7616`
- `R9_medium_h123_k3_p4_s21`
  - `mean_eval_dice = 0.7486`
- `R11_medium_h123_k3_p4_s999`
  - `mean_eval_dice = 0.7413`
- `R5_medium_h123_k3_p2_s42`
  - `mean_eval_dice = 0.7199`

#### Comparison to earlier medium baselines

Earlier `h=1,2,3` medium benchmark reference:

- `LOCF`: `0.6507`
- `UNet-mask`: `0.6475`
- `UNet-image+mask`: `0.7024`

All residual variants in this overnight sweep exceeded the earlier `UNet-image+mask` baseline overall, and all strongly exceeded `LOCF` overall.

#### History-length result

At matched prior strength `4.0` and seed `42`:

- `k=1`: `0.7986`
- `k=3`: `0.7642`

This means the current gain does **not** appear to come from adding longer observed history. In the present setup, the stronger effect is the residual formulation itself, and `k=1` actually performed better than `k=3`.

#### Prior-strength result

At `k=3`, `seed=42`:

- `prior=2.0`: `0.7199`
- `prior=4.0`: `0.7642`
- `prior=6.0`: `0.7616`

This suggests:

- a weak persistence prior underperforms,
- moderate-to-strong persistence priors are clearly better,
- `prior=4.0` and `prior=6.0` are similar overall,
- `prior=4.0` looks like the best default among the tested values.

#### Horizon breakdown highlights

Best `k=1, prior=4.0, seed=42` run:

- `h=1`: `0.8150`
- `h=2`: `0.8276`
- `h=3`: `0.7343`

Key implication:

- the residual formulation is not only helping at longer short-term horizons,
- it is now also beating the earlier persistence baseline even at immediate `h=1` on this medium synthetic benchmark.

#### Tier breakdown highlights

Best `k=1, prior=4.0, seed=42` run:

- `Tier A`: `0.7587`
- `Tier B`: `0.8204`
- `Tier C`: `0.8926`

Compared with the earlier medium `UNet-image+mask` baseline:

- `Tier A`: improved
- `Tier B`: improved substantially
- `Tier C`: improved dramatically

Compared with earlier `LOCF`:

- `Tier C` is now nearly matched while remaining strong on `Tier A` and `Tier B`

This is important because earlier learned models struggled badly to compete in the most persistence-dominated or harder-regime settings.

### Interpretation

This overnight sweep materially strengthens the project.

1. The residual idea is real.
   - This is not a tiny fluctuation over the previous learned baseline.
   - The gain over the earlier `UNet-image+mask` medium benchmark is substantial.

2. The core mechanism seems to be residual forecasting over a strong persistence prior.
   - The current evidence does **not** support a strong claim that longer explicit history is the main driver.
   - Right now, `k=1` is actually the strongest result.

3. The short-term forecasting story becomes sharper.
   - Earlier, the benchmark suggested that `h=1` was persistence-dominated.
   - Now, a persistence-aware residual learner appears capable of surpassing naive copy-forward behavior even in that immediate regime on the medium synthetic benchmark.

4. This is a more interesting paper story than simple architecture scaling.
   - The key lesson is not "deeper model beats baseline."
   - The key lesson is that modeling **residual change on top of persistence** is a better inductive bias for immediate and short-term longitudinal tumor forecasting.

### Results Note: Practical Conclusion

The strongest learned direction so far is now:

- persistence-aware residual short-term forecasting,
- with a moderate-to-strong persistence prior,
- and no clear need yet for longer history beyond the latest session.

### Design Decision: Suggested Next Steps

1. Treat the residual formulation as the current lead model family.
2. Reframe the paper contribution more around **persistence-aware residual forecasting** than around multi-session stacking.
3. Use `k=1, prior=4.0` as the provisional best model.
4. Keep `k=3` results in the paper as an important negative/clarifying ablation rather than hiding them.
5. Next analyses should compare:
   - best residual model,
   - `LOCF`,
   - earlier `UNet-image+mask`,
   under identical horizon and tier tables.

### Hypothesis Update: What Now Seems Most True

The most credible current claim is:

- immediate short-term tumor forecasting is persistence-dominated,
- naive `LOCF` is therefore hard to beat,
- but a model that learns **residual correction over a strong persistence prior** can outperform both naive persistence and standard direct-mask forecasting on the synthetic medium benchmark.

### Discussion Note: Why Additional Baselines Still Matter

After the residual improvement became clear, an important concern was raised:

- if the comparison set remains too narrow,
- a reviewer could argue that the current conclusion only demonstrates the weakness of one specific direct `U-Net` family rather than a broader immediate-forecasting phenomenon.

This is a fair concern.

The current evidence strongly supports:

- `LOCF` is a serious baseline,
- direct `U-Net` forecasting can struggle in immediate next-step prediction,
- persistence-aware residual correction is much stronger.

However, this is not yet identical to showing that the persistence problem is broader than one architecture family. To strengthen that claim, the paper should ideally include a small number of **additional baseline families** with different modeling assumptions.

### Design Decision: Baseline Diversity Without Scope Explosion

The project should not expand into a large model zoo.

The right compromise is:

- keep the baseline set small,
- include a few qualitatively different model families,
- avoid large hyperparameter tuning efforts,
- and preserve a single common evaluation protocol.

The reason for adding a few more models is **not** to maximize leaderboard coverage. The reason is to establish that the immediate short-term forecasting difficulty is likely a more general modeling issue, rather than merely a `U-Net` issue.

### Design Decision: Comparator Philosophy

Additional learned baselines should follow these rules:

1. use established models or already-available repos where possible,
2. represent meaningfully different inductive biases,
3. require little or no custom hyperparameter tuning,
4. run under the same immediate forecasting setup,
5. be included only if they improve the scientific argument.

This keeps the paper honest:

- if multiple reasonable direct predictors struggle against `LOCF`,
- then the paper can more credibly argue that immediate next-step forecasting is a persistence-dominated regime,
- rather than overfitting the conclusion to one baseline architecture.

### Design Decision: Provisional Baseline Suite for the Paper Premise

The current preferred comparison set is:

1. `LOCF`
2. one mechanistic `PDE` / diffusion-reaction baseline
3. `UNet-mask`
4. `UNet-image+mask`
5. one or two additional established CNN-style baselines if they are easy to run fairly
6. best persistence-aware residual model
7. optional stronger comparator later:
   - `TaDiff`
   - or a flow-based model

### Discussion Note: Why PDE Baselines Are Useful Here

A PDE or reaction-diffusion solver is especially useful in this project because it serves a different role from the CNN baselines:

- it is mechanistic rather than purely learned,
- it provides a strong reference from prior tumor-growth literature,
- and it can test whether the persistence problem is also visible outside neural forecasting models.

There is also prior anecdotal evidence in this project context that some solver-based approaches have had difficulty beating `LOCF` on short-horizon forecasting. Re-running at least one such baseline under the current protocol would strengthen the premise of the paper considerably.

### Design Decision: What Not To Do

Avoid:

- adding many small CNN variants with cosmetic differences,
- spending significant time rescuing underperforming baselines by tuning,
- turning baseline expansion into a second project,
- or weakening the paper narrative by including models that do not answer a clear question.

The purpose of the extra baselines is to support the central claim:

- immediate next-step tumor forecasting is hard because of persistence,
- and a persistence-aware residual model is a more appropriate response than standard direct forecasting.

---

## 2026-06-23

### Experiment ID

`EXP-008-broader-direct-cnn-baselines`

### Objective

Test whether the earlier immediate-forecasting story was only a `U-Net` artifact or whether it persists across a slightly broader family of direct CNN forecasters.

The key question was:

- is `LOCF` mainly beating one particular direct architecture,
- or is immediate next-step forecasting broadly difficult for standard direct learned predictors?

### Setup

Two new direct CNN variants were added under the same benchmark pipeline:

- `resunet_image_mask`
- `plain_cnn_image_mask`

These were evaluated under the same data loader, loss, optimizer, and split logic as the existing direct baselines.

Two evaluation settings were run on the medium synthetic benchmark:

1. `h=1` only:
   - output root:
     `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/premise_medium_h1_suite`
2. `h=1,2,3` combined:
   - output root:
     `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/context_medium_h123_suite`

Shared settings:

- dataset:
  `/content/drive/MyDrive/synthetic_tumor_benchmark/fixed_dataset_medium_v1`
- `fit_sessions = 3`
- `epochs = 12`
- `batch_size = 2`
- `seed = 42`

### Key Results

#### Immediate next-step only (`h=1`)

Overall means:

- `resunet_image_mask`: `0.7932`
- `LOCF`: `0.7621`
- `unet_image_mask`: `0.7294`
- `unet_mask`: `0.5247`
- `plain_cnn_image_mask`: `0.5179`

Tier breakdown:

- `Tier A`
  - `unet_image_mask`: `0.7788`
  - `resunet_image_mask`: `0.7652`
  - `plain_cnn_image_mask`: `0.7393`
  - `unet_mask`: `0.7306`
  - `LOCF`: `0.7091`

- `Tier B`
  - `resunet_image_mask`: `0.7703`
  - `LOCF`: `0.7401`
  - `unet_image_mask`: `0.6378`
  - `unet_mask`: `0.3584`
  - `plain_cnn_image_mask`: `0.3377`

- `Tier C`
  - `LOCF`: `0.9546`
  - `resunet_image_mask`: `0.9210`
  - `unet_image_mask`: `0.8117`
  - `unet_mask`: `0.3637`
  - `plain_cnn_image_mask`: `0.3481`

#### Context setting (`h=1,2,3`)

Overall means:

- `resunet_image_mask`: `0.7892`
- `unet_image_mask`: `0.7024`
- `LOCF`: `0.6507`
- `unet_mask`: `0.6475`
- `plain_cnn_image_mask`: `0.6247`

### Interpretation

This experiment changes the paper framing in an important way.

1. The earlier result was **not** only a vanilla `U-Net` failure story.
   - A stronger direct CNN family (`ResUNet`) can beat `LOCF` even in the immediate `h=1` regime on the medium synthetic benchmark.

2. At the same time, persistence remains a meaningful challenge.
   - `LOCF` still beats multiple reasonable direct baselines:
     - `unet_mask`
     - `plain_cnn_image_mask`
   - It also remains strongest on `Tier C` for immediate forecasting.

3. The right claim is therefore more nuanced than "direct CNNs fail."
   - Immediate forecasting is not impossible for direct learned models.
   - But performance is highly architecture-sensitive, and persistence remains especially hard to beat in the most copy-forward-dominated regime.

4. The broader paper story is still intact, but it must be stated carefully.
   - `LOCF` is a strong baseline that many reasonable direct methods do not beat.
   - Some stronger architectures can surpass it.
   - Persistence-aware or residual-style inductive biases still appear highly effective, but they are no longer the only evidence that immediate forecasting can beat naive persistence.

### Results Note: Practical Conclusion

The new direct-CNN comparison supports the following more honest framing:

- immediate next-step forecasting is a persistence-heavy regime,
- `LOCF` is a serious and nontrivial baseline,
- weaker direct forecasters often fail to beat it,
- stronger direct CNNs can beat it,
- and the hardest persistence-dominated tier (`Tier C`) remains difficult.

### Design Decision: Paper Framing Adjustment

Do **not** overstate the premise as:

- "standard direct CNNs cannot beat `LOCF`."

Instead, frame it as:

- "`LOCF` is a strong baseline in immediate short-term forecasting, especially in harder persistence-dominated regimes, and model performance is highly sensitive to inductive bias and architecture."

This is more accurate and more defensible.

### Discussion Note: Tier Ladder Interpretation

An important clarification emerged about how the synthetic tiers should be interpreted.

The three benchmark tiers are not just arbitrary subsets. They define a **graduated difficulty ladder**:

- `Tier A`: simple procedural geometric growth
- `Tier B`: isotropic reaction-diffusion growth
- `Tier C`: anisotropic + heterogeneous reaction-diffusion growth

This means the benchmark is best understood as moving from:

- simpler, more idealized cases,
- to more mechanistic synthetic growth,
- to the most heterogeneous and real-like synthetic regime.

Under that interpretation:

- `Tier A` should not be discarded as "too easy"
- `Tier B` should not be treated as merely intermediate noise
- `Tier C` should be treated as the hardest and most important synthetic frontier

The value of the tier structure is precisely that it lets the project show **progressive improvement across increasing realism and difficulty**.

### Hypothesis Update: What Tier Progress Means

Under the new interpretation, progress should be read as a sequence:

1. a method first becomes reliable on `Tier A`,
2. then extends that improvement to `Tier B`,
3. and finally begins to close the gap on `Tier C`.

This is a better way to describe method development than relying only on a single overall average.

It also means that failure to fully beat `LOCF` on `Tier C` is not automatically a negative result. It may simply indicate that the method has not yet crossed the final difficulty frontier.

### Design Decision: Role of Tier C

`Tier C` should now be treated as:

1. the hardest synthetic regime,
2. the most relevant synthetic reference for later `SAILOR` comparison,
3. and the main frontier for future modeling improvements.

This does **not** mean `Tier A` and `Tier B` are unimportant.

Instead:

- `Tier A` shows whether a model can succeed in an idealized regime,
- `Tier B` shows whether that success extends to mechanistic isotropic dynamics,
- `Tier C` tests whether the approach survives anisotropy and heterogeneity.

### Design Decision: How Future Experiments Should Be Chosen

From this point onward, experiment choices should be explained as part of a tier-ladder strategy:

- use `Tier A` and `Tier B` to demonstrate early and intermediate gains,
- use `Tier C` to judge whether those gains actually transfer to the most realistic synthetic regime,
- and later use `SAILOR` as the real-data endpoint of that progression.

This creates a coherent experimental narrative:

- simple synthetic cases,
- then more complex synthetic cases,
- then real data.

### Discussion Note: Current Medium Dataset May Be Too Small for Final Claims

Another important planning conclusion emerged after the tier-ladder interpretation became clearer.

The current medium benchmark has been very useful for:

- model screening,
- identifying the persistence problem,
- testing direct vs residual formulations,
- and understanding how results vary across tiers and horizons.

However, it may still be somewhat small for the final paper-level confirmation stage.

The concern is not that the benchmark is invalid. The concern is that:

- test sample counts are still limited,
- `Tier C` remains especially sample-poor,
- and some conclusions may look more fragile than necessary.

This suggests that the current medium benchmark is better viewed as a **development-scale benchmark** rather than the final confirmation benchmark.

### Design Decision: Increase Dataset Size Moderately

The current direction is to increase overall dataset size moderately, while preserving the same tier structure.

The goal is **not** to create a huge synthetic dataset with thousands of cases.

Instead, the goal is to better mimic a moderate-sized medical forecasting benchmark by increasing the number of synthetic patients per tier while keeping:

- the same generation logic,
- the same tier ladder (`A -> B -> C`),
- and the same evaluation protocol.

The intended benefit is:

- more stable overall means,
- more credible tier-wise conclusions,
- less fragile `Tier C` analysis,
- and a stronger final benchmark confirmation stage.

### Design Decision: Do Not Rerun Everything

It is not necessary to repeat every prior experiment on the larger dataset.

The current medium benchmark should be retained as:

- development evidence,
- hypothesis-forming evidence,
- and model-screening evidence.

The larger benchmark should be used as:

- confirmation evidence for the core paper claims.

This means only the most important model comparisons need to be rerun on the larger dataset.

### Design Decision: Development vs Confirmation Study Structure

The paper can explicitly present the work as a two-stage study:

1. **development stage**
   - moderate synthetic benchmark used for exploration, baseline screening, and hypothesis generation
2. **confirmation stage**
   - larger synthetic benchmark used to verify the main findings more robustly

This is a strength rather than a weakness because it shows:

- efficient use of computation,
- focused narrowing of the experimental space,
- and deliberate confirmation of only the most important findings.

### Design Decision: What To Reconfirm on the Larger Dataset

On the larger synthetic benchmark, priority should be given to rerunning only the headline comparisons:

1. `LOCF`
2. strongest standard direct baseline(s)
3. strongest residual model
4. optional mechanistic `PDE` baseline if available in time

The purpose of the larger benchmark is not to replay all exploratory experiments, but to confirm the main scientific story under a more stable sample size.

### Design Decision: Confirmation Benchmark Config

The next benchmark scale should preserve the exact same generation logic as the current medium benchmark while increasing cohort size to a more stable moderate scale.

The chosen confirmation configuration is:

- config file:
  `configs/benchmark_confirm.yaml`
- patients per tier:
  - `Tier A = 80`
  - `Tier B = 80`
  - `Tier C = 80`
- total synthetic patients:
  - `240`

This keeps the benchmark comparable to the current medium dataset while materially improving:

- test-set size,
- tier-wise stability,
- and especially the reliability of `Tier C` conclusions.

### Design Decision: First Confirmation Runs

The first runs on the confirmation benchmark should be limited to the highest-value headline comparisons:

1. `LOCF`
2. `UNet-image+mask`
3. `ResUNet-image+mask`
4. best residual model (`k=1`, `prior=4.0`)

Optional later addition:

5. one mechanistic `PDE` baseline

### Design Decision: Run Order on the Confirmation Benchmark

The preferred execution order is:

1. generate the confirmation dataset,
2. run the direct-CNN suite on `h=1`,
3. run the direct-CNN suite on `h=1,2,3`,
4. run the best residual model on the same settings,
5. compute tier-wise and horizon-wise comparison tables,
6. only then decide whether PDE or real-data validation should be inserted next.

This order keeps the confirmation phase focused on the strongest current scientific questions.

---

## 2026-06-23

### Experiment ID

`EXP-009-confirmation-benchmark-core-reruns`

### Objective

Re-run the core headline comparisons on a larger confirmation-stage synthetic benchmark in order to determine:

- whether the main ranking remains stable at a larger cohort size,
- whether the immediate `h=1` story survives scaling,
- whether the tier-ladder interpretation becomes more reliable,
- and whether `Tier C` remains the decisive frontier.

### Setup

Confirmation benchmark:

- config:
  `configs/benchmark_confirm.yaml`
- patients per tier:
  - `A = 80`
  - `B = 80`
  - `C = 80`
- total patients:
  - `240`

Compared models:

- `LOCF`
- `unet_mask`
- `unet_image_mask`
- `resunet_image_mask`
- `plain_cnn_image_mask`
- `residual_unet_image_mask_k1`

Two evaluation settings:

1. immediate next-step only:
   - `h=1`
2. short-term context:
   - `h=1,2,3`

Shared training settings:

- `fit_sessions = 3`
- `epochs = 12`
- `batch_size = 2`
- `seed = 42`

### Key Results

#### Immediate next-step only (`h=1`)

Overall means:

- `resunet_image_mask`: `0.8167`
- `LOCF`: `0.7395`
- `residual_unet_image_mask_k1`: `0.7057`
- `unet_image_mask`: `0.6005`
- `plain_cnn_image_mask`: `0.5279`
- `unet_mask`: `0.5197`

Tier breakdown:

- `Tier A`
  - `residual_unet_image_mask_k1`: `0.7823`
  - `resunet_image_mask`: `0.7734`
  - `unet_image_mask`: `0.7727`
  - `LOCF`: `0.7177`

- `Tier B`
  - `resunet_image_mask`: `0.8742`
  - `LOCF`: `0.7145`
  - `residual_unet_image_mask_k1`: `0.5776`
  - `unet_image_mask`: `0.4240`

- `Tier C`
  - `resunet_image_mask`: `0.8360`
  - `LOCF`: `0.7851`
  - `residual_unet_image_mask_k1`: `0.6890`
  - `unet_image_mask`: `0.4885`

#### Combined short-term setting (`h=1,2,3`)

Overall means:

- `resunet_image_mask`: `0.7481`
- `residual_unet_image_mask_k1`: `0.7120`
- `unet_image_mask`: `0.6575`
- `LOCF`: `0.6228`
- `unet_mask`: `0.6150`
- `plain_cnn_image_mask`: `0.4971`

Horizon breakdown:

- `h=1`
  - `resunet_image_mask`: `0.8310`
  - `residual_unet_image_mask_k1`: `0.7668`
  - `LOCF`: `0.7395`

- `h=2`
  - `resunet_image_mask`: `0.7004`
  - `residual_unet_image_mask_k1`: `0.6799`
  - `unet_image_mask`: `0.6555`
  - `LOCF`: `0.5822`

- `h=3`
  - `resunet_image_mask`: `0.6607`
  - `residual_unet_image_mask_k1`: `0.6552`
  - `unet_image_mask`: `0.6160`
  - `LOCF`: `0.4657`

Tier breakdown:

- `Tier A`
  - `residual_unet_image_mask_k1`: `0.7669`
  - `resunet_image_mask`: `0.7451`
  - `unet_image_mask`: `0.7435`
  - `LOCF`: `0.5729`

- `Tier B`
  - `resunet_image_mask`: `0.6692`
  - `residual_unet_image_mask_k1`: `0.5895`
  - `LOCF`: `0.5834`

- `Tier C`
  - `resunet_image_mask`: `0.8096`
  - `LOCF`: `0.7291`
  - `residual_unet_image_mask_k1`: `0.7146`

### Interpretation

This confirmation benchmark is one of the most important results so far.

1. The ranking is now much more stable.
   - On the larger confirmation benchmark, `resunet_image_mask` is the strongest direct learned baseline both for immediate `h=1` and for the combined `h=1,2,3` setting.

2. The previous residual result does not remain the lead model under confirmation-scale evidence.
   - The residual `k=1, prior=4.0` model still clearly beats several standard baselines and remains competitive.
   - However, it is now consistently below `resunet_image_mask`.

3. The larger benchmark strengthens the claim that the phenomenon is broader than one weak direct baseline.
   - Some direct CNNs still perform poorly.
   - But a stronger direct CNN architecture can clearly beat `LOCF`, including on `Tier C`.

4. `Tier C` is now much more informative than before.
   - Sample counts are larger.
   - The conclusion is no longer based on only a handful of `Tier C` examples.
   - Importantly, `resunet_image_mask` now exceeds `LOCF` on `Tier C` both in the immediate `h=1` setting and in the combined `h=1,2,3` setting.

5. The tier-ladder story is still valid, but the frontier has shifted.
   - `Tier C` remains the hardest and most meaningful regime.
   - The key result is no longer that `Tier C` resists all learned models.
   - The stronger result is that only the better architectures meaningfully cross that frontier.

### Results Note: Practical Conclusion

The confirmation benchmark supports the following updated main message:

- immediate and short-term tumor forecasting are persistence-heavy regimes,
- `LOCF` is a serious baseline,
- several standard direct predictors fail to beat it,
- but a stronger architecture such as `ResUNet` can surpass it even on the hardest synthetic tier,
- while simpler models continue to struggle substantially.

### Design Decision: Updated Lead Baseline

For the current paper state, the strongest learned benchmark model should now be treated as:

1. `resunet_image_mask`

The residual `k=1, prior=4.0` model should still be retained because:

- it remains clearly useful,
- it offers a conceptually different persistence-aware perspective,
- and it provides an informative comparison against standard direct forecasting.

But it should no longer be described as the strongest learned result on the confirmation benchmark.

### Design Decision: Next Step After Confirmation

The next highest-value experiments should now shift away from more synthetic architecture screening and toward:

1. one mechanistic `PDE` baseline comparison,
2. and then a minimal real-data validation on `SAILOR`.

At this point, the synthetic benchmark evidence is strong enough that the main uncertainty is no longer whether the phenomenon exists, but how well it transfers beyond the synthetic setting.

### Design Decision: Data Audit Becomes a First-Class Workstream

At this stage, the project should no longer treat the synthetic dataset as a passive backdrop for model evaluation.

The data itself has become central to the scientific contribution.

This means the next phase must include a dedicated benchmark-audit workstream covering:

1. synthetic dataset characterization,
2. tier-ladder validation,
3. synthetic-to-`SAILOR` comparison,
4. and explicit realism limitations.

This is necessary both for the DMS paper and for the longer-term goal of evolving the project into a stronger benchmark dataset.

The detailed audit checklist and deliverables are captured in:

- `docs/DATA_AUDIT_PLAN.md`

---

## 2026-06-23

### Experiment ID

`EXP-010-synthetic-vs-sailor-audit-v1`

### Objective

Run the first dataset-centered audit comparing the confirmation synthetic benchmark against processed `SAILOR` in order to assess:

- whether the synthetic benchmark has plausible temporal structure,
- whether the tier ladder corresponds to increasingly real-like properties,
- and whether `Tier C` is actually the closest synthetic regime to real data.

### Setup

Synthetic dataset:

- `synthetic_confirm_v1`
- confirmation benchmark with `80` patients per tier

Real dataset:

- processed `SAILOR`

Audit outputs were generated using:

- `scripts/run_data_audit.py`

Compared statistics:

- sessions per patient
- follow-up duration
- treatment prevalence
- mean inter-scan interval
- tumor volume
- elongation ratio
- compactness proxy
- bounding-box dimensions
- delta-volume
- relative growth rate

### Key Results

#### Temporal structure

Synthetic tiers:

- `n_sessions_mean`: about `5.25` to `5.81`
- `followup_days_mean`: about `163` to `202`
- `mean_interval_days_mean`: about `39` to `42`

`SAILOR`:

- `n_sessions_mean`: `9.15`
- `followup_days_mean`: `495.58`
- `mean_interval_days_mean`: `52.83`

Interpretation:

- the synthetic benchmark is currently much shorter than `SAILOR` in both session count and total follow-up,
- and slightly shorter in inter-scan gap.

#### Shape / geometric proxies

Elongation ratio:

- `Tier A`: about `1.63` to `1.68`
- `Tier B`: about `1.38` to `1.52` on train/test, but unstable on small val split
- `Tier C`: about `1.33` to `1.43`
- `SAILOR`: `1.52`

Compactness proxy:

- `Tier A`: about `0.15` to `0.22`
- `Tier B`: about `0.12` on test, lower on val
- `Tier C`: about `0.09` to `0.18`
- `SAILOR`: `0.1199`

Interpretation:

- on scale-free shape proxies, `Tier B` and `Tier C` appear directionally closer to `SAILOR` than `Tier A`,
- while `Tier A` looks more inflated and idealized geometrically.

#### Tumor size scale

Session-level `volume_vox_mean`:

- synthetic: roughly `800` to `7000` depending on tier/split
- `SAILOR`: `69259`

Interpretation:

- absolute tumor size scale is far apart,
- but this should be treated cautiously because the synthetic benchmark currently uses voxel counts on an abstract grid rather than a physically calibrated mm-scale volume definition.

#### Growth dynamics

Relative growth rate:

- `Tier A`: about `1.13` to `1.28`
- `Tier B`: about `-0.11` to `0.11`
- `Tier C`: about `0.02` to `0.10`
- `SAILOR`: `0.556`

Interpretation:

- `Tier A` currently grows too aggressively relative to `SAILOR`,
- `Tier B` and `Tier C` currently grow too slowly on average relative to `SAILOR`,
- so no current tier matches the real-data growth-rate regime well.

### Interpretation

This audit is extremely important because it shows both strengths and weaknesses of the current benchmark.

1. The synthetic benchmark is **structured**, not arbitrary.
   - The tiers differ in measurable ways.
   - `Tier A` appears more idealized.
   - `Tier B/C` appear more plausible on some shape proxies.

2. `Tier C` is only **partially** the closest regime to real data.
   - It is more credible than `Tier A` as a realistic synthetic frontier.
   - But the benchmark still does not match `SAILOR` well on temporal depth or growth-rate behavior.

3. The largest realism gap is not anisotropy alone.
   - It is also the mismatch in:
     - session count,
     - total follow-up duration,
     - and effective growth dynamics.

4. This means the benchmark is already useful for controlled method comparison, but it is not yet a well-calibrated real-data surrogate.

### Results Note: Practical Conclusion

The first realism audit supports the following honest benchmark claim:

- the synthetic benchmark provides a meaningful tiered forecasting testbed,
- `Tier B/C` are more plausible than `Tier A` on some shape statistics,
- but the benchmark currently under-matches `SAILOR` in longitudinal depth and does not yet reproduce real-data growth-rate behavior well.

### Design Decision: Benchmark v2 Priorities

The next benchmark-improvement priorities should be:

1. increase synthetic session count and follow-up duration,
2. better calibrate growth-rate behavior against `SAILOR`,
3. preserve the tier ladder while making `Tier C` more convincingly real-like,
4. and be cautious about any claim that current synthetic performance alone implies real-data performance.

## 2026-06-23: Strategic Pivot From "More Runs" To "Better Benchmark Design"

### Context

After the first synthetic-to-real audit, we explicitly decided not to rush into a half-baked submission narrative.

The main conclusion was:

- more model runs alone will not solve the central weakness,
- the benchmark itself now needs to be formalized more carefully,
- and the project should be guided first by the data-mining and benchmark-design problem, not by deadline pressure.

### Key discussion points

1. The benchmark should not be narrowly tuned to `SAILOR`.
   - `SAILOR` is important for validation and calibration,
   - but the benchmark should remain a general synthetic testbed.

2. The current three-tier ladder is useful, but still too coarse.
   - `Tier A/B/C` help us narrate the progression from simpler to more heterogeneous growth,
   - but a stronger benchmark needs explicit challenge axes underneath those tiers.

3. Short-horizon forecasting remains the central task.
   - especially immediate next-step forecasting under high persistence,
   - where `LOCF` is a meaningful baseline rather than a trivial one.

4. The benchmark should help uncover forecasting failure modes.
   - not just report a leaderboard,
   - but show when persistence dominates,
   - when learned models help,
   - and which data regimes cause that shift.

5. We should save useful outputs as tables and figures when they clarify important experiment cycles.
   - not every run needs a polished artifact,
   - but the important transitions in project understanding should be documented.

### Project-level decision

We are moving toward a `Benchmark v2` framing built around:

- temporal depth,
- growth regime diversity,
- spatial complexity,
- tumor burden / size scale,
- treatment effects,
- and observation realism.

This shifts the project from:

- "Can one learned model beat `LOCF` on a synthetic dataset?"

to:

- "How should a synthetic longitudinal tumor benchmark be designed so that short-horizon forecasting claims are meaningful, auditable, and scientifically useful?"

### New document

To support this shift, a new blueprint document was created:

- `docs/BENCHMARK_V2_BLUEPRINT.md`

That document defines the rationale, challenge axes, regime structure, evaluation tasks, and next build plan for the next benchmark iteration.

## 2026-06-23: Benchmark v2 Spec Drafted Against The Current Codebase

### Why this was needed

The blueprint established what a stronger benchmark should look like conceptually, but we also needed a practical answer to:

- which controls already exist,
- which improvements can be made by config alone,
- and which ones require code changes.

Without that bridge, the redesign would stay too abstract.

### Outcome

A concrete implementation-facing spec was created:

- `docs/BENCHMARK_V2_SPEC.md`

### Main conclusions from the spec

1. The current generator already gives us a useful base.
   - We have global controls for schedule, growth ranges, initialization, treatment probability, and image synthesis.

2. A real v2 cannot be achieved with YAML changes alone.
   - The current `A/B/C` distinction is still mostly driven by rollout mode.
   - It does not yet expose regime-specific schedule, growth, or observation controls.

3. The best next milestone is a `v2-lite`.
   - Increase longitudinal depth.
   - Increase follow-up.
   - Modestly expand size.
   - Adjust global growth settings.
   - Re-audit before doing another major model sweep.

4. The minimum deeper code upgrades are now clear.
   - regime-specific overrides,
   - growth subtype sampling,
   - size-targeted initialization,
   - richer treatment profiles,
   - and an observation corruption layer.

### Decision

The redesign path is now:

1. build `v2-lite` first,
2. run the audit again,
3. then decide whether to implement the full regime-aware `v2-core`.

## 2026-06-24: v2-lite Audit And First Regime-Control Lessons

### What was run

A first `v2-lite` benchmark candidate was generated with:

- more sessions,
- longer follow-up,
- slightly broader growth ranges,
- and a modestly larger cohort.

### Main audit outcome

The temporal redesign worked meaningfully well.

Overall:

- `n_sessions_mean = 7.54`
- `followup_days_mean = 310.47`
- `mean_interval_days_mean = 47.56`

Interpretation:

- this is substantially closer to the processed `SAILOR` temporal profile than the earlier benchmark versions,
- so the benchmark improved meaningfully on longitudinal depth.

### Main problem exposed by the audit

The growth ladder remained poorly calibrated.

Test-split relative growth means:

- `Tier A`: `0.867`
- `Tier B`: `0.191`
- `Tier C`: `0.043`

Interpretation:

- `Tier A` was still too aggressive,
- `Tier B` remained relatively quiet,
- and `Tier C`, which should be the main synthetic frontier, was still too weak on average.

### Additional concern

`Tier B` and `Tier C` still showed zero-volume sessions in their summaries.

Interpretation:

- some mechanistic trajectories are collapsing too strongly,
- which is a realism and evaluation concern,
- especially if treatment effects dominate too often.

### Decision

`v2-lite` is useful as an intermediate diagnostic benchmark, but not good enough to freeze as the main benchmark version.

The main lesson was:

- temporal structure improved,
- but growth-regime design still needed direct control.

## 2026-06-24: Tier-Specific Override Support Added

### Why this was needed

The `v2-lite` audit showed that global parameter changes were not enough.

We needed to control `Tier A/B/C` separately for:

- schedule,
- simulation,
- labeling,
- and image synthesis.

### Implementation change

The generator was extended to support tier-specific overrides via:

- `schedule_overrides`
- `simulation_overrides`
- `labeling_overrides`
- `image_synthesis_overrides`

These were threaded through the config loader and generation pipeline.

### Impact

This is a key benchmark-design milestone because it shifts the benchmark from:

- one global simulator with three modes

to:

- one generator with explicit regime-level control.

## 2026-06-24: v2_core_candidate Audit

### Goal

The first regime-aware `v2_core_candidate` was designed to:

- calm `Tier A`,
- strengthen `Tier C`,
- and create a more interpretable tier ladder.

### What improved

Temporal structure remained strong:

- `n_sessions_mean = 7.65`
- `followup_days_mean = 313.32`
- `mean_interval_days_mean = 47.30`

`Tier C` improved meaningfully:

- longer histories than `A/B`
- larger tumors than `Tier B`
- stronger growth than `Tier B`

Interpretation:

- `Tier C` no longer looked like the weakest or quietest regime,
- which was an important improvement over `v2-lite`.

### What failed

`Tier A` became too aggressive.

Test-split summaries:

- `Tier A relative_growth_rate_mean = 2.96`
- `Tier A delta_volume_vox_mean = 3262.81`

Interpretation:

- the attempt to calm `Tier A` through initialization changes alone did not work,
- because the procedural growth rule itself remained too expansive.

### Decision

The override mechanism worked.
The design direction was correct.
But the benchmark still needed one more major fix:

- explicit procedural-growth controls for `Tier A`

## 2026-06-25: Mechanism-Aware Procedural Controls And Biophys1 Candidate

### Motivation

After the `v2_core_candidate` audit, the key insight was:

- `Tier A` is exploding because of the procedural dynamics,
- not simply because of initialization scale.

At the same time, a broader design question was raised:

- can we constrain benchmark growth more realistically using mechanism-aware reasoning rather than blind parameter fiddling?

### Implementation change

The procedural simulator was extended with explicit control knobs for:

- procedural step frequency,
- number of local expansion seeds,
- local expansion radii,
- shift probability and shift ranges,
- offshoot probability and offshoot ranges,
- offshoot radii.

These controls allow `Tier A` to be tuned structurally rather than indirectly.

### New candidate

A new config was created:

- `configs/benchmark_v2_core_candidate_biophys1.yaml`

Design intent:

- strongly calm `Tier A` procedural expansion,
- modestly strengthen `Tier B`,
- keep `Tier C` near the improved direction from the previous candidate,
- and move toward literature-informed / mechanism-constrained parameter choices.

### Important methodological note

The current benchmark is still best described as:

- literature-informed,
- mechanism-constrained,
- and audit-calibrated

rather than:

- fully biophysically calibrated.

That distinction matters because the simulator still operates on an abstract voxel grid rather than a fully physical mm-scale model with patient-fitted parameters.

### Current status

Progress is actively being logged in this file.

Other project-tracking documents now include:

- `docs/BENCHMARK_V2_BLUEPRINT.md`
- `docs/BENCHMARK_V2_SPEC.md`
- `docs/DATA_AUDIT_PLAN.md`
- `docs/References.md`

## 2026-06-26: biophys1 Audit - First Balanced Tier Ladder Candidate

### Context

After `v2_core_candidate`, the main remaining failure was:

- `Tier A` was still exploding because the procedural rule itself was too aggressive.

To address this, explicit procedural controls were added and a new candidate was generated:

- `configs/benchmark_v2_core_candidate_biophys1.yaml`

### Overall temporal profile

The improved temporal structure was preserved:

- `n_sessions_mean = 7.63`
- `followup_days_mean = 315.10`
- `mean_interval_days_mean = 47.68`

Interpretation:

- the benchmark remains much stronger than the earlier versions on longitudinal depth,
- while staying in the same broad temporal regime as the more recent candidates.

### Key audit result

The `Tier A` explosion was finally controlled.

Test-split `Tier A`:

- `volume_vox_mean = 280.03`
- `delta_volume_vox_mean = 42.62`
- `relative_growth_rate_mean = 0.220`

Interpretation:

- this is a major correction relative to the previous `v2_core_candidate`,
- and now behaves much more like a controlled persistence regime rather than an explosive growth regime.

### Tier ladder behavior

Test split summaries now show a much cleaner ordering by size:

- `Tier A volume_vox_mean = 280.03`
- `Tier B volume_vox_mean = 1634.27`
- `Tier C volume_vox_mean = 6297.11`

Relative growth means on test:

- `Tier A = 0.220`
- `Tier B = 0.254`
- `Tier C = 0.279`

Interpretation:

- growth rates are now in a much narrower and more believable band,
- `Tier A` is no longer wildly dominant,
- `Tier C` remains the largest and most active regime,
- and the ladder is now substantially more balanced than any earlier candidate.

### Remaining caveats

1. `Tier A` may now be somewhat too small.
   - It behaves well as a persistence regime,
   - but its scale may be overly conservative relative to the other tiers.

2. `Tier B` and `Tier C` still show zero-volume sessions and negative growth transitions.
   - This may be acceptable in a limited treatment-affected subset,
   - but should still be monitored because too many collapse cases can distort realism.

3. The spatial-complexity separation is not yet as strong as the growth/size separation.
   - The current audit supports a better tier ladder,
   - but not yet a fully convincing shape-complexity hierarchy on its own.

### Decision

`biophys1` is the strongest synthetic benchmark candidate so far.

It is the first candidate where:

- the temporal structure is good,
- `Tier A` is not pathological,
- `Tier C` is meaningfully larger and more active,
- and the overall ladder is interpretable enough to justify moving forward.

### Next recommended step

Move to model evaluation on `biophys1`, while keeping two follow-up checks in mind:

1. monitor whether the benchmark remains balanced under forecasting baselines,
2. and later revisit shape-complexity separation and collapse behavior if needed.

## 2026-06-26: biophys1 Baseline Results Support A Regime-Dependent Forecasting Story

### Setup

Baseline pack run on:

- `fixed_dataset_v2_core_candidate_biophys1`

Methods:

- `LOCF`
- `unet_mask`
- `unet_image_mask`
- `plain_cnn_image_mask`
- `resunet_image_mask`

Forecast setting:

- `fit_sessions = 3`
- `horizons = 1,2,3`

### Overall summary

- `LOCF`: `0.7504`
- `unet_mask`: `0.7721`
- `unet_image_mask`: `0.7781`
- `plain_cnn_image_mask`: `0.7473`
- `resunet_image_mask`: `0.8710`

### Key results

1. `LOCF` remains strong, but is no longer dominant overall.

2. `resunet_image_mask` clearly outperforms all other methods.

3. Not every learned model wins.
   - `plain_cnn_image_mask` is roughly tied with or slightly below `LOCF`.
   - This makes the result more believable than a benchmark where every learned model trivially wins.

4. The strongest model beats `LOCF` across all horizons.

Horizon means:

- `h=1`
  - `resunet_image_mask`: `0.9057`
  - `LOCF`: `0.8322`

- `h=2`
  - `resunet_image_mask`: `0.8731`
  - `LOCF`: `0.7506`

- `h=3`
  - `resunet_image_mask`: `0.8303`
  - `LOCF`: `0.6596`

5. The tier behavior is especially informative.

Tier means:

- `Tier A`
  - `resunet_image_mask`: `0.8539`
  - `LOCF`: `0.8518`

- `Tier B`
  - `resunet_image_mask`: `0.8633`
  - `LOCF`: `0.6823`

- `Tier C`
  - `resunet_image_mask`: `0.9009`
  - `LOCF`: `0.6909`

### Interpretation

This is the first benchmark version where the data story and the model story align well enough to support a stronger scientific claim.

The benchmark no longer behaves like a single monotone "easy/medium/hard" ladder.
Instead, it behaves more like a set of longitudinal tumor-growth regimes:

- `Tier A`: controlled persistence regime where `LOCF` remains extremely competitive
- `Tier B`: moderate mechanistic regime where stronger learned models help more clearly
- `Tier C`: larger and more active heterogeneous regime where the strongest learned model shows the clearest advantage

### Hypothesis Update

The project framing should shift from:

- "Can deep models beat `LOCF` on a tiered synthetic benchmark?"

to:

- "How do tumor-growth regime characteristics determine when persistence is sufficient and when stronger learned forecasting models are warranted?"

### Design Decision

Proceed into a deeper analysis phase rather than immediately over-polishing the paper narrative.

The next goal is to mine the sample-level structure behind these regime-dependent wins and losses:

- by size,
- by future growth,
- by recent growth,
- by treatment status,
- by horizon,
- and by tier.

### New analysis artifact

To support that next phase, a reusable script was added:

- `scripts/analyze_regime_drivers.py`

Its purpose is to merge:

- per-sample forecast outputs,
- audit-derived tumor/trajectory features,
- and pairwise model comparisons such as `LOCF` vs `resunet_image_mask`

so that the benchmark can be analyzed as a data-mining problem rather than only a benchmark leaderboard.

## 2026-06-26: Research Standard And Analysis Priorities Reaffirmed

### Discussion Note

The project should now be evaluated primarily as a research task, not as a deadline-driven submission exercise.

The working standard is:

- the work should be solid,
- defensible,
- and able to withstand hard questioning.

This means that:

- weak spots should be surfaced rather than hidden,
- synthetic claims should be scoped carefully,
- and progress should continue to be logged and preserved as concrete artifacts.

### Design Decision

From this point onward:

1. important experiment cycles should be logged promptly,
2. important tables and figures should be saved as persistent artifacts,
3. the analysis should increasingly focus on what tumor / trajectory characteristics inform forecasting strategy,
4. and any novelty gaps should be treated as opportunities for additional analysis or modeling rather than as something to paper over.

### Current novelty opportunities

At the current stage, the most promising novelty lies less in proposing yet another forecasting architecture and more in showing:

- how longitudinal tumor-growth characteristics affect whether persistence is sufficient,
- how regime structure should inform model choice,
- and how a forecasting benchmark can be designed to expose that dependence cleanly.

Possible future novelty extensions remain open:

- an interpretable predictor of when `ResUNet` will beat `LOCF`,
- a treatment-aware regime analysis,
- a model-selection policy based on input-side trajectory features,
- or a more explicit probability-field / uncertainty-aware forecasting view.

### New artifact

To support persistent figure-saving from the regime analysis phase, a plotting/export utility was added:

- `scripts/export_regime_figures.py`

## 2026-06-26: Robustness Confirmation And Architecture-Generalization Planning

### Results Note: ResUNet robustness

An additional seed check was run for `resunet_image_mask` on `biophys1`.

Comparison:

- seed `42`: `0.871031`
- seed `123`: `0.870167`

Interpretation:

- the current lead result is not behaving like a seed fluke,
- and the main `ResUNet` advantage on `biophys1` appears very stable on the overall metric.

### Artifact note

Important figures and tables from the regime-analysis phase were exported and saved, including:

- Dice by horizon across methods
- win rate by tier
- win rate by input volume bin
- win rate by recent growth bin
- win rate by future growth bin

### Design Decision

The next phase should broaden the study carefully rather than immediately scaling into many more models.

The key question is now:

- is the current regime-dependent conclusion specific to `ResUNet`,
- or does it reflect broader model-family behavior across tumor-growth regimes?

### New planning document

To support that next phase, a dedicated roadmap was created:

- `docs/MODEL_GENERALIZATION_ROADMAP.md`

This roadmap defines:

- the recommended compact model set,
- the order of architecture-family comparison,
- the cross-regime transfer plan,
- the broader tumor-data / forecastability analysis phase,
- and the longer-term real-data bridge.

### Implementation note

As the first practical step toward that roadmap, optional transformer-family baseline support was added for:

- `UNETR` via MONAI

This keeps the next experiment small and interpretable:

- one established transformer-style medical 3D model,
- compared fairly against the current CNN family and `LOCF`,
- on the stabilized `biophys1` benchmark.
