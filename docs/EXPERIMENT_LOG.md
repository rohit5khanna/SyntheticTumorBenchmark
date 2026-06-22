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
