# Updated Log

## 2026-07-11: Workspace Consolidation And Research Restart

After a one-week pause, the project was restarted with two priorities:

1. preserve the broader research goal rather than drifting back into a paper-only mindset;
2. reduce the personal and practical disconnect caused by project files being spread across multiple folders and repositories.

A new local hub folder was created at:

`/Users/rohitkhanna/Desktop/ORIE Spring 2026/Tumor_Growth_Project/TaDiff/Current_Work`

This folder links the active project materials into one place without moving the underlying repositories. This preserves existing git paths, script assumptions, and Colab references while making the project easier to re-enter.

The hub contains:

- `repos/`: links to the active and reference repositories;
- `active_docs/`: links to logs, research questions, references, dataset notes, and draft artifacts;
- `literature/`: links to local PDFs relevant to tumor growth modeling and forecasting;
- `notes/`: links to roadmaps and older project notes;
- `utilities/`: links to backup and Google Drive pull scripts;
- `outputs_index/`: links to generated figure/table folders inside the benchmark repo.

A project map was also added at:

`docs/PROJECT_MAP.md`

Current working interpretation:

- `SyntheticTumorBenchmark` remains the canonical code and documentation repo.
- `Current_Work` is the human-facing entry point for the overall research workspace.
- The next research phase should resume from the regime-aware analysis foundation, with special attention to probabilistic growth fields, ranking-style evaluation, and descriptor-conditioned forecasting ideas inspired by the recent literature scan.

## 2026-07-11: Growth-Aware Evaluation Layer Added

After reading the first two suggested papers, two ideas became immediately useful for the next phase:

1. the observation from probabilistic glioma-growth modeling that larger growth cases may be easier for learned models than moderate/subtle cases;
2. the forward-ranking framing, where future tumor growth is evaluated as a problem of ranking likely future-growth regions rather than only maximizing segmentation overlap.

To test these ideas against the existing SRD evidence, a new analysis script was added:

`scripts/analyze_growth_ranking.py`

The script adds a growth-aware evaluation layer over existing baseline outputs. It computes:

- per-sample future growth volume;
- new-growth, absolute-change, and net-growth bins;
- Dice by future new-growth bin;
- model gain over LOCF by future new-growth bin;
- optional forward-growth ranking metrics when model checkpoints are available.

The ranking mode loads accessible learned-model checkpoints and evaluates whether predicted probability maps rank truly new tumor-growth voxels highly outside the input tumor mask. This is intended as a first bridge from standard Dice evaluation toward probabilistic growth-field thinking.

Interpretation standard:

- if learned models only improve in high-growth bins, the project should explicitly treat short-horizon forecasting difficulty as change-size dependent;
- if ranking metrics reveal useful signal even when Dice is modest, future work should move toward probabilistic growth-likelihood or residual-growth prediction;
- if both Dice gains and ranking metrics are weak in moderate-growth cases, that becomes evidence that persistence dominance is not merely a metric artifact but a real difficulty mode.

## 2026-07-11: First Growth-Aware SRD Results

The first growth-aware evaluation pass was run on `fixed_dataset_v3_lite_generalized` using the compact baseline output directory `v3lite_compact_h123_s42`.

The analysis produced:

- `growth_sample_features.csv`
- `dice_by_new_growth_bin.csv`
- `model_gain_vs_locf_by_new_growth_bin.csv`
- `growth_ranking_metrics.csv`
- `growth_aware_evaluation_report.md`

### Main result

The learned-model advantage is strongly dependent on the amount of true new tumor growth.

For ResUNet image+mask versus LOCF:

- low new-growth bin: mean gap approximately `-0.003`, win rate `0.244`;
- medium new-growth bin: mean gap approximately `+0.011`, win rate `0.634`;
- high new-growth bin: mean gap approximately `+0.196`, win rate `0.927`.

For U-Net image+mask versus LOCF:

- low new-growth bin: mean gap approximately `-0.056`, win rate `0.171`;
- medium new-growth bin: mean gap approximately `-0.071`, win rate `0.317`;
- high new-growth bin: mean gap approximately `+0.168`, win rate `0.878`.

This is an important strengthening of the regime-aware story. It suggests that learned models are not uniformly better than persistence; they become useful primarily when the next scan contains meaningful new growth.

### Ranking-style result

Forward-growth ranking metrics were successfully computed from learned-model checkpoints.

Across horizons, both U-Net and ResUNet produced nontrivial average precision and high recall in the top-ranked candidate regions. This supports the idea that the learned models may contain useful spatial information about where new tumor growth is likely to occur, even when standard Dice does not tell the full story.

However, these ranking metrics are not yet conclusive because they still need simple baselines:

- random ranking / prevalence baseline;
- distance-from-current-mask expansion baseline;
- possibly simple dilation or morphology-based growth-front baseline.

Until those baselines are added, the ranking result should be interpreted as promising but incomplete.

### Immediate interpretation

The current evidence now supports a sharper hypothesis:

Short-horizon tumor forecasting is partly a persistence-residual problem. When future new growth is small, LOCF is hard to beat and learned models may add little or even hurt. When future new growth is substantial, learned models can improve strongly, suggesting that the task should be evaluated and modeled around departures from persistence rather than only full-mask overlap.

## 2026-07-12: Forward-Growth Ranking Baselines Added

The growth-ranking analysis was extended with simple reference baselines:

1. `random_prevalence`: analytical random-ranking baseline;
2. `distance_to_input_mask`: a growth-front baseline that ranks candidate voxels by closeness to the current tumor mask.

This addition was necessary because high recall in a future-growth ranking task may be achievable by a simple boundary-expansion heuristic.

### Result summary

The learned models strongly beat random prevalence, as expected.

More importantly, the learned models also outperform the distance-to-input-mask baseline on average precision and recall at the growth-volume budget.

Approximate mean average precision:

- Horizon 1:
  - distance baseline: `0.471`
  - ResUNet: `0.639`
  - U-Net: `0.671`
- Horizon 2:
  - distance baseline: `0.523`
  - ResUNet: `0.648`
  - U-Net: `0.677`
- Horizon 3:
  - distance baseline: `0.474`
  - ResUNet: `0.604`
  - U-Net: `0.625`

Approximate recall at growth-volume budget:

- Horizon 1:
  - distance baseline: `0.470`
  - ResUNet: `0.621`
  - U-Net: `0.639`
- Horizon 2:
  - distance baseline: `0.500`
  - ResUNet: `0.653`
  - U-Net: `0.659`
- Horizon 3:
  - distance baseline: `0.455`
  - ResUNet: `0.615`
  - U-Net: `0.610`

### Important nuance

The distance baseline remains very strong at broad top-region recall, especially recall at 5% of candidate voxels. This means a large amount of future growth is spatially near the current tumor boundary, and any future probabilistic growth-field method should treat boundary expansion as a serious baseline or prior.

### Interpretation

These results strengthen the case for a ranking/probabilistic-growth-field view:

- the learned models are not only improving full-mask Dice in high-growth cases;
- they also provide a better localized ranking of true new-growth voxels than a simple distance-based growth-front heuristic;
- however, much of the broad recall can still be explained by spatial proximity to the existing tumor.

The next methodological implication is that a useful forecasting model may combine:

1. a persistence or growth-front prior;
2. learned correction/ranking over that prior;
3. regime or descriptor conditioning to determine when the learned correction should be trusted.

## 2026-07-12: Stratified Forward-Growth Ranking Findings

The forward-growth ranking analysis was stratified by new-growth bin, tier, and horizon.

### By new-growth bin

The learned models outperform the distance-to-input-mask baseline across low, medium, and high new-growth bins when measured by average precision and recall at the growth-volume budget.

Approximate average precision:

- low growth:
  - distance baseline: `0.115`
  - ResUNet: `0.229`
  - U-Net: `0.221`
- medium growth:
  - distance baseline: `0.394`
  - ResUNet: `0.569`
  - U-Net: `0.620`
- high growth:
  - distance baseline: `0.720`
  - ResUNet: `0.842`
  - U-Net: `0.858`

This suggests the learned models add spatial ranking value beyond simple boundary proximity, especially as true future growth becomes more substantial.

### By tier

The tier-wise ranking results show a useful split:

- Tier A has small future-growth volume and lower average precision overall.
- Tier B and Tier C show very high learned-model ranking performance.
- The learned models outperform the distance baseline in all tiers on average precision and recall at the growth-volume budget.

Approximate average precision:

- Tier A:
  - distance baseline: `0.204`
  - ResUNet: `0.328`
  - U-Net: `0.380`
- Tier B:
  - distance baseline: `0.798`
  - ResUNet: `0.959`
  - U-Net: `0.962`
- Tier C:
  - distance baseline: `0.747`
  - ResUNet: `0.909`
  - U-Net: `0.912`

### Nuance

The distance baseline remains extremely strong for broad capture at 5% of candidate voxels. This means the current tumor boundary is already a powerful growth-location prior. The learned models are most clearly adding value in tighter prioritization metrics, especially average precision and recall at the growth-volume budget.

### Interpretation

The evidence now supports a more precise framing:

Short-horizon forecasting has two separable components:

1. a broad growth-front component that can be captured by proximity to the current tumor;
2. a finer spatial prioritization component where learned models add value, especially in medium/high growth and Tier B/C cases.

This further motivates a future model that explicitly combines a simple growth-front prior with a learned, descriptor-conditioned residual or ranking correction.

## 2026-07-12: Hybrid Growth-Front Plus Learned Ranking Test

A first non-training hybrid test was run to ask whether a simple growth-front prior can be combined with learned-model probability maps to improve forward-growth ranking.

The hybrid score was constructed by rank-normalizing:

1. the distance-to-current-mask score;
2. the learned model probability score;

and taking a weighted combination:

`hybrid = (1 - alpha) * distance_rank + alpha * model_rank`

with `alpha` values `0.25`, `0.50`, and `0.75`.

### Main result

The hybrid does not produce a dramatic new method win. Instead, it behaves as a controlled tradeoff between the distance prior and the learned model.

The best hybrid settings generally use a high learned-model weight (`alpha = 0.75`). This confirms that the learned model carries most of the fine-ranking signal.

### Where hybrid helps

The hybrid can slightly improve some ResUNet ranking summaries and can retain broader top-region recall better than the pure learned model in some settings.

Examples:

- high-growth ResUNet AP:
  - pure ResUNet: `0.842`
  - hybrid alpha 0.75: `0.854`
- Tier C ResUNet AP:
  - pure ResUNet: `0.909`
  - hybrid alpha 0.75: `0.918`

### Where hybrid does not help

The pure U-Net ranking remains very strong and is often equal to or better than the hybrid, especially on average precision. Naive rank fusion is therefore not enough to claim a new forecasting method.

### Interpretation

This is still useful evidence.

It suggests:

1. the distance prior is valuable as a broad growth-front prior;
2. the learned model is valuable for fine spatial prioritization;
3. naive fixed-weight fusion only partly combines these benefits;
4. a stronger future method should probably learn when to weight the growth-front prior versus the learned correction, instead of using a fixed alpha.

This points toward descriptor-conditioned gating or adaptive residual/ranking correction rather than simple static fusion.

## 2026-07-12: Adaptive Ranking Gate Upper-Bound Check

An adaptive ranking-gate analysis was run on the hybrid growth-ranking outputs. The goal was to estimate how much headroom exists if one could choose, per case or per coarse group, between:

- distance-to-input-mask growth-front ranking;
- the pure learned model ranking;
- fixed hybrid distance/model rank-fusion variants.

### Oracle result

For U-Net:

- distance mean AP: `0.489`
- pure U-Net mean AP: `0.659`
- best static method: pure U-Net, `0.659`
- oracle mean AP: `0.674`
- oracle gain over best static: `+0.015`

For ResUNet:

- distance mean AP: `0.489`
- pure ResUNet mean AP: `0.632`
- best static method: hybrid alpha `0.75`, `0.640`
- oracle mean AP: `0.654`
- oracle gain over best static: `+0.014`

### Group-policy result

The best group-level choices are structured but modest:

- U-Net is usually the best static choice.
- Hybrid U-Net with high learned weight is slightly preferred in high-growth, Tier B/C, and Horizon 3 groups.
- ResUNet benefits more from the hybrid, especially with alpha `0.75`.
- Distance alone is never the best group policy on average, although it remains useful as a broad prior.

### Interpretation

The oracle headroom is real but small. This means a complex adaptive gate is not automatically justified by these results alone.

The current evidence suggests:

1. learned ranking is already close to the best static choice;
2. simple distance/model fusion mainly helps weaker learned rankers or selected subgroups;
3. a future gate would need to be very lightweight and descriptor-driven to be worth adding;
4. the stronger methodological direction may be learned residual/ranking correction over a growth-front prior, rather than a post-hoc static selector.

This is a useful narrowing result. It prevents over-investing in naive gating while preserving the idea that descriptor-conditioned weighting could still be useful if implemented inside the model or residual field rather than as a coarse post-hoc rule.

## 2026-06-30: Project Pivot From Competition Draft To Substantial Research Program

This document records the current pivot in the project direction after the first DMS-oriented draft and the subsequent review/assessment phase.

## Why this pivot happened

After reviewing the current manuscript, the conclusion is that the present paper draft does **not** yet stand strongly as a mature research contribution. In its current form, it is still too close to:

- running several forecasting methods,
- comparing them on a synthetic setup,
- and reporting differences without a sufficiently deep contribution layer.

That does **not** mean the project direction is weak. It means the current draft is functioning more like a research scaffold than a final paper.

The useful part is that the last 7-10 days of work uncovered a real and promising structure:

1. short-horizon tumor forecasting is not a uniform problem;
2. forecasting difficulty appears to depend on tumor regime and case characteristics;
3. persistence can dominate in some settings;
4. learned models help in others;
5. mechanism-guided approaches also struggle in the real-data short-horizon setting;
6. therefore, a single uniform forecasting strategy is likely suboptimal.

This structure is the actual foundation of the project.

## Updated objective

The goal is no longer merely to polish the current draft for a competition deadline.

The updated objective is:

> Build this into a substantial research contribution on short-horizon tumor forecasting by using regime-aware analysis to motivate and develop a regime-conditioned forecasting methodology.

This shifts the project from:

- a benchmark-style comparison effort

to:

- an analysis-driven forecasting research program.

## Current interpretation of the project

The project now has two tightly connected components.

### Component 1: Regime-aware evaluation and analysis

This component asks:

- when is short-horizon tumor forecasting difficult?
- which cases are persistence-dominant?
- when do learned models provide value?
- how do synthetic regimes and real-data cases differ?
- how do tumor growth, morphology, treatment state, and temporal context affect forecast difficulty?

This part remains important, but it is no longer viewed as the endpoint of the project. It is the analytical foundation.

### Component 2: Regime-conditioned forecasting

This component asks:

- can regime information be encoded as a prior or conditioning signal?
- can current forecasting models be adapted to use this information instead of learning uniformly over all cases?
- can short-horizon forecasting improve when the model is explicitly informed about the regime it is operating in?

This is now viewed as the likely methodological contribution that would make the overall work stand more strongly.

## New research direction

The strongest current direction is:

1. continue strengthening the regime-aware evaluation framework;
2. identify reliable regime descriptors available at forecast time;
3. inject that regime information into existing tumor forecasting networks as a prior;
4. compare plain forecasting models against regime-conditioned versions;
5. study whether this helps especially in the difficult short-horizon settings.

This is preferred over inventing an entirely new forecasting architecture from scratch, because:

- it is better motivated by the analysis we already have;
- it is more feasible in the near term;
- it creates a clean narrative from problem analysis to method design;
- it can still become a real forecasting-method contribution.

## Candidate methodological forms

The regime-conditioned method could take one or more of the following forms:

1. regime feature channels appended to the model input;
2. regime embeddings injected into the encoder/bottleneck/decoder;
3. a regime-conditioned residual forecast that learns deviations from persistence;
4. a regime-aware selector or hybrid that decides how much to trust persistence versus a learned forecast.

At the current stage, the most promising direction appears to be:

- regime-conditioned residual forecasting built on top of the current learned models.

This is especially aligned with the current short-horizon evidence, where persistence is often strong and the forecasting problem can be interpreted as learning when and how to depart from it.

## What has changed conceptually

The benchmark or synthetic dataset is no longer being treated as the main contribution by itself.

Instead, it is now treated as:

- a controlled synthetic environment,
- a data-mining instrument,
- and a tool for understanding how tumor/data characteristics shape forecasting behavior.

The contribution we want is not:

- "we built a synthetic dataset and ran models on it."

The contribution we want is closer to:

- "we identified regime-dependent structure in short-horizon tumor forecasting and used it to design a more informed forecasting approach."

## Implications for paper strategy

At this point, the project can evolve in one of two ways:

### One-paper path

A single integrated paper that includes:

1. regime-aware evaluation and analysis;
2. the synthetic regime environment;
3. the real-data bridge;
4. a regime-conditioned forecasting method.

This currently appears to be the better near-term direction because it creates a complete story:

- identify the problem structure;
- analyze it;
- use it to motivate a method;
- test the method.

### Two-paper path

A longer-term split into:

1. a regime-aware evaluation paper;
2. a regime-conditioned forecasting-method paper.

This remains possible later, but at the current stage the work is likely better served by one coherent integrated paper.

## Updated standard for success

The project should no longer be judged by:

- whether the current competition draft is polished enough.

It should be judged by whether it achieves the following:

1. establishes short-horizon tumor forecasting as a regime-dependent problem;
2. provides careful evidence across synthetic and real settings;
3. identifies tumor/data descriptors that matter for forecasting difficulty;
4. turns those descriptors into a forecasting prior or conditioning mechanism;
5. demonstrates that this added structure improves or stabilizes forecasting behavior.

If these conditions are met, the project becomes a much stronger and more substantial contribution than the current draft.

## Immediate next-step priorities

The next phase of work should prioritize:

1. consolidating the regime-analysis results into a cleaner and more comprehensive framework;
2. deciding the exact set of regime features available at forecast time;
3. implementing the first regime-conditioned forecasting variant on top of the current learned model pipeline;
4. testing whether the regime-conditioned version helps overall, by horizon, and by regime/tier;
5. preserving the real-data connection so the work does not become purely synthetic.

## Summary

This project is no longer being treated as a short-term competition submission exercise.

It is now being treated as the start of a broader research program on:

- short-horizon tumor forecasting,
- regime-aware evaluation,
- and regime-conditioned forecasting methods.

The current draft does not yet stand strongly on its own, but the project direction still does.

The next meaningful contribution layer is clear:

- move from regime-aware analysis
- to regime-conditioned forecasting.

## 2026-06-30: Two-Track Sprint Through 2026-07-05

For the immediate sprint window from **June 30, 2026 to July 5, 2026**, the project will proceed on two parallel tracks.

### Track 1: Regime-aware evaluation sprint

Primary objective:

- make the current regime-aware evaluation framework rigorous, insightful, and robust

This means the short-term paper-facing goal is **not** to overexpand model scope, but to improve the quality of the current analytical contribution.

Core tasks in this track:

1. tighten the synthetic-regime evaluation story;
2. finalize the most defensible synthetic dataset setup to use as the main experimental environment;
3. strengthen tier-wise, horizon-wise, and model-family-wise analysis;
4. make the real-data bridge clearer and more careful;
5. identify which current findings are truly robust enough to keep in the paper and which should be dropped;
6. ensure the current draft becomes a serious evaluation-and-analysis paper scaffold rather than a loose experiment bundle.

Expected outputs from this track by the end of the sprint:

- a cleaner and more rigorous regime-aware evaluation section;
- a better-defined set of trustworthy claims;
- a clearer list of what evidence is still missing;
- a stabilized foundation for the later regime-conditioned method work.

### Track 2: Literature immersion and idea development

Primary objective:

- study tumor forecasting, longitudinal analysis, and related deep-learning literature in much greater depth

This track is necessary because stronger methodological ideas should be informed by the literature rather than improvised only from current experiments.

Core tasks in this track:

1. read more deeply on tumor forecasting papers;
2. study longitudinal medical-imaging forecasting methods more carefully;
3. examine how prior information, conditioning, or mechanism-guided information has been encoded in related work;
4. extract ideas that can inform a regime-conditioned forecasting method;
5. refine the conceptual bridge from regime-aware evaluation to regime-conditioned forecasting.

Expected outputs from this track by the end of the sprint:

- a stronger understanding of the tumor-forecasting literature;
- a clearer sense of what is original versus already standard;
- a better-informed shortlist of regime-conditioned method ideas;
- stronger conceptual justification for the next methodological phase.

### Combined purpose of the sprint

These two tracks serve complementary roles.

- Track 1 strengthens the current work so it becomes more rigorous and defensible.
- Track 2 prepares the next contribution layer so that the project can evolve beyond evaluation into method development.

The intention is that after this sprint, the project should be in a stronger position to move into aggressive **5-day blocks** of:

1. focused experimentation,
2. progress assessment,
3. methodological refinement,
4. and paper restructuring.

### Strategic note

This sprint is not just about adding more experiments.

## 2026-07-03: Regime-Analysis Robustness Deepening

We are now explicitly stress-testing the descriptor-driven regime story from multiple angles so that the analysis is not dependent on a single modeling choice or a single threshold definition.

### What the latest robustness checks established

The current evidence suggests a split between what is strongly stable and what is only moderately stable.

Strongly stable:

1. the `target_wins` anchor persists across threshold sweeps;
2. the main descriptor signals remain visible even after removing the composite `activity_score` and `structure_score`;
3. morphology and burden descriptors continue to matter when predicting cross-regime pull and transition populations;
4. part of the signal survives patient-grouped validation, meaning the story is not entirely driven by repeated within-patient structure.

Moderately stable:

1. the `both_easy` anchor depends on a reasonable threshold band and is not threshold-invariant;
2. the soft-profile composition is useful but sensitive to stricter stability definitions;
3. transition prediction is structurally informative but remains the noisiest and least stable component.

### New robustness layer now being added

To make the data-analysis side more defensible, we are extending the pipeline with:

1. descriptor ablations that remove `tier`, remove `horizon`, and remove both;
2. permutation checks so that observed predictive signal can be compared against a null label baseline;
3. bootstrap confidence intervals for the anchor-separation feature gaps;
4. PCA-based structure analysis to study whether the descriptor geometry itself reflects meaningful separation between anchor and ambiguous populations.

### Why this matters

This phase keeps the focus on the data and descriptor structure rather than turning the work into a model-comparison exercise.

The goal is to answer a sharper question:

> Which forecast-time tumor descriptors consistently explain when learned forecasting departs from persistence, and which parts of that explanation remain stable under harder scrutiny?

That question is now the center of the current sprint.

## 2026-07-03: Descriptor Robustness Findings Consolidated

The latest descriptor-focused checks strengthened the analysis in a useful way.

### What improved

1. The cross-regime pull signal did **not** disappear when `tier` and `horizon` were removed from the prediction setup.
2. In fact, the pooled `raw_only` cross-regime task performed slightly better than the versions that included `tier` and/or `horizon`, suggesting that the descriptor structure is not merely a reflection of the evaluation scaffold.
3. Bootstrap confidence intervals confirmed that the largest anchor-separation effects are concentrated in:
   - input tumor volume,
   - treatment-at-input state,
   - recent relative growth,
   - and connected-component count.
4. PCA recovered a similar geometry:
   - the first principal component aligned strongly with recent growth, burden, treatment status, and structural fragmentation;
   - the persistence-core and learned-advantage-core populations were separated mainly along this axis;
   - ambiguous populations occupied intermediate regions rather than appearing unstructured.

### What remains limited

1. Patient-grouped validation kept part of the signal but reduced its strength, so the descriptor story survives but is not yet strong enough to be treated as a near-clinical predictor.
2. Transition prediction remains the weakest and noisiest layer.
3. Permutation tests were suggestive rather than decisive, especially once grouped validation was introduced.

### Immediate next refinement layer

The next useful checks are now more localized:

1. test whether the anchor split survives **within tiers** rather than only in the pooled dataset;
2. test whether it survives **within horizons** rather than only across all horizons together;
3. derive simple descriptor rules to see whether part of the regime pattern is expressible in a transparent threshold-style form.

This keeps the focus on whether the descriptor structure is genuinely local and stable, not just visible after aggregation.

## 2026-07-03: Localized Descriptor Checks Added

The descriptor story was then pushed one step further by asking whether the pooled effects remain visible once the data are broken down by tier and by horizon, and whether the same patterns can be expressed with simple transparent rules.

### Within-tier anchor separation

The anchor split remained visible within tiers, but not uniformly:

1. Tier B showed strong and multi-feature separation between the persistence-core and learned-advantage-core cases.
   The largest effects were associated with:
   - input volume,
   - treatment status,
   - compactness,
   - elongation,
   - connected-component count,
   - and recent growth.
2. Tier C showed even larger apparent effects for several descriptors, especially:
   - elongation,
   - recent growth,
   - compactness,
   - connected-component count,
   - and input volume.
3. However, Tier C also had a very small `both_easy_core` comparison set, so those effect sizes must be treated cautiously.
4. Tier A did not provide a comparable within-tier anchor split in this check, which is itself informative: the strongest local descriptor contrast is not being driven by the easiest regime alone.

### Within-horizon anchor separation

The anchor split also remained visible across horizons:

1. Horizon 1 showed a very strong volume effect and clear recent-growth and connected-component differences.
2. Horizon 2 continued to show strong effects in volume, treatment status, recent growth, and connected-component count.
3. Horizon 3 retained similar signals, especially for treatment status, volume, and recent growth.
4. The sign pattern remained broadly consistent across horizons, even though some lower-magnitude features such as elongation and delta-days became unstable.

This suggests that the descriptor structure is not confined to a single forecast horizon.

### Simple rule extraction

Simple depth-limited decision rules recovered a compressed version of the same story.

For cross-regime pull:

1. the first split was on `delta_days`;
2. the next key split was on `recent_relative_growth`;
3. smaller follow-up splits used compactness, elongation, and volume.

For transition:

1. the tree relied mainly on compactness and recent growth;
2. other descriptors played little or no role in the final shallow rule set.

### Current interpretation

These localized checks reinforce three points:

1. the descriptor story is not purely a pooled-data artifact;
2. volume, recent growth, treatment state, and structural descriptors remain central even after localization;
3. the transition population is still harder to stabilize than the core anchor contrast.

At the same time, the results also show where caution is required:

1. some tiers contain small comparison groups;
2. some very large effect sizes are partly driven by those small subsets;
3. therefore the right interpretation is that the local signal is promising and structurally consistent, but not yet fully stabilized in every subgroup.

## 2026-07-03: Compact Core-Descriptor Phase Started

After the evidence-map step, the descriptor story now appears concentrated in a small core set rather than spread evenly across all available variables.

The current working core set is:

1. `input_volume_vox`
2. `recent_relative_growth`
3. `treated_at_input`
4. `input_connected_component_count`

with `input_compactness_proxy` as the next descriptor just outside the core.

### Why this phase matters

The next question is no longer only whether descriptor signal exists, but whether that signal can be compressed into a small, reusable signature that:

1. still explains the anchor split;
2. still helps with cross-regime ambiguity;
3. survives grouped validation reasonably well;
4. and can later serve as a practical conditioning prior for a forecasting model.

### Immediate checks

The compact-descriptor pass will therefore test:

1. pooled predictive signal using only the core descriptors;
2. grouped predictive signal using only the core descriptors;
3. PCA structure using only the core descriptors;
4. shallow decision rules using only the core descriptors.

This is intended to answer whether the regime story can be made simpler, sharper, and more operational.

## 2026-07-03: Objective Critique Of Current Status

At this stage, the project has clearly improved in rigor, but it is still better described as a strong analytical foundation than as a finished research contribution.

### What is genuinely working

1. The core research question is good:
   - short-horizon forecasting is not uniform,
   - persistence dominates in some cases,
   - learned models help in others,
   - and the difference appears to depend on observable forecast-time descriptors.
2. The project has moved well beyond a simple model-comparison exercise.
3. The descriptor story is now supported by multiple checks:
   - threshold sensitivity,
   - raw-only ablations,
   - grouped validation,
   - bootstrap confidence intervals,
   - PCA structure,
   - within-tier checks,
   - within-horizon checks,
   - and shallow rule extraction.
4. The compact-core analysis suggests a useful two-layer view:
   - a small core descriptor set explains the main persistence-vs-learned axis,
   - while additional descriptors appear necessary for ambiguity resolution.

### What is still weak or incomplete

1. The project does not yet contain a strong method contribution.
   The best part of the work is currently the analysis, not a new forecasting mechanism.
2. The regime language can become stronger than the evidence if used carelessly.
   What is firmly supported is structured forecast-difficulty variation in descriptor space, not yet a universal or clinically grounded regime taxonomy.
3. Some of the strongest subgroup effects, especially in Tier C, are driven by small comparison sets and therefore need caution.
4. Grouped validation preserves only moderate signal, so descriptor-based generalization is promising but not yet strong.
5. The transition population remains the noisiest and least stable part of the analysis.

### What can be defended now

The following claims are currently on solid ground:

1. short-horizon forecasting difficulty is not uniform;
2. performance differences between persistence and learned models are structured rather than random;
3. a small set of forecast-time descriptors consistently separates persistence-core and learned-advantage-core populations:
   - input volume,
   - recent relative growth,
   - treatment-at-input state,
   - and connected-component complexity;
4. additional descriptors seem to refine ambiguity rather than define the core axis;
5. the descriptor structure survives several robustness checks, though not perfectly.

### What is not yet established

The following should **not** be claimed yet:

1. a clinically stable regime taxonomy;
2. robust patient-level classification of ambiguous cases;
3. strong real-world conclusions from the synthetic tier structure alone;
4. a better forecasting method;
5. a causal explanation for the descriptor-performance relationship.

### Current strategic interpretation

The present direction is still good, but only if the project now moves from:

- descriptor analysis

to

- an operational forecasting mechanism or conditioning prior built from that analysis.

If that next step is taken carefully, the current analysis can serve as a strong foundation rather than an endpoint.

## 2026-07-03: Exception-Case Audit Added To SRD Analysis Stack

The next refinement layer in the SRD analysis was to stop looking only at average behavior and explicitly audit the cases that break the dominant regime story.

This led to the addition of a dedicated **exception-case audit**.

### Why this was needed

By this stage, the analysis had already established several stable aggregate patterns:

1. tier A behaves much more like a persistence-friendly region;
2. tiers B and C are more favorable to learned corrections;
3. recent growth, volume, treatment-at-input, and connected-component count repeatedly emerge as important descriptors;
4. the two-axis activity/structure map provides a compact summary of descriptor-level regime position.

However, aggregate summaries alone are not enough for a serious data analysis layer. The harder and more useful question is:

- which cases break the story?

That is where a more careful audit becomes valuable.

### What the exception-case audit does

The new audit constructs a case-type profile in the activity/structure regime plane and then flags cases that appear atypical in one or more ways:

1. **rare-case exceptions**:
   cases belonging to a very small case-type group;
2. **quadrant mismatches**:
   cases whose regime quadrant differs from the dominant quadrant of their case type, when that case type has a stable dominant quadrant;
3. **centroid-overlap exceptions**:
   cases that are closer in activity/structure space to another case-type centroid than to the centroid of their own assigned case type.

This is useful because it converts “interesting anecdotes” into a reproducible and descriptor-aware exception analysis.

### What this adds conceptually

This exception-case layer sharpens the project in two ways:

1. it tells us where the current descriptor story is strong;
2. it also tells us where the current descriptor story is incomplete.

That is important for the broader research goal, because a regime-conditioned forecasting method should not be built only around the average or dominant behavior. It should also be informed by the ambiguous and boundary cases where the regime structure is less clean.

### Implementation note

The SRD workflow now includes:

- `scripts/analyze_exception_cases.py`

and the consolidated runner:

- `scripts/run_srd_regime_analysis.py`

was extended so that this exception-case audit becomes part of the same reproducible analysis bundle.

## 2026-07-03: Soft Regime Membership Layer Added

The first exception-case audit was useful, but it also exposed a limitation in the current hard-label analysis.

### Why this additional layer was needed

The exception results showed that:

1. `target_wins` still has a strong descriptor core;
2. `both_easy` also has a meaningful persistence-friendly core;
3. `both_hard` and `close_mixed` do not behave like equally stable regimes;
4. a large number of cases were being flagged simply because they were slightly closer to another centroid.

That is not necessarily evidence of true anomalies. It is better interpreted as evidence that the middle of the descriptor map is fuzzy and transitional.

So instead of treating every ambiguity as an exception, the analysis now introduces a softer membership view.

### What the soft regime layer does

This new layer identifies only the **stable descriptor cores** and then measures how strongly each sample aligns with them.

At the current thresholds, the stable profiles are expected to be the regimes with:

1. enough sample count;
2. a sufficiently dominant descriptor quadrant.

The key idea is:

- `both_easy` and `target_wins` behave more like anchor regimes,
- while `both_hard` and `close_mixed` often behave more like transition populations.

The soft-membership analysis therefore:

1. computes distances from each case to the stable regime centroids;
2. converts those distances into soft membership probabilities;
3. labels cases as:
   - `core_aligned`,
   - `transition`,
   - or `cross_regime_pull`.

### Why this matters

This is a more faithful descriptor-level interpretation than forcing every sample into a rigid category.

It lets the analysis distinguish between:

1. clean support for a stable regime story;
2. ambiguous boundary cases;
3. cases genuinely pulled toward another regime.

That is especially valuable for the broader forecasting question, because a future regime-conditioned model should likely treat:

- core persistence-like cases,
- core learned-advantage cases,
- and fuzzy transition cases

as meaningfully different operating conditions.

### Implementation note

The SRD workflow now also includes:

- `scripts/analyze_soft_regime_membership.py`

and the consolidated runner:

- `scripts/run_srd_regime_analysis.py`

was extended again so this softer regime-membership layer can be reproduced in the same bundle.

## 2026-07-03: Soft Regime Profile Characterization Added

Once the soft regime-membership layer was in place, the next natural question became:

- what do the ambiguous populations actually look like?

Counting `core_aligned`, `cross_regime_pull`, and `transition` cases is useful, but it is still one step short of describing their descriptor-level identities.

### What this new layer adds

The new profile characterization layer explicitly compares four populations:

1. `both_easy_core`
2. `target_wins_core`
3. `cross_regime_pull`
4. `transition`

This turns the soft regime analysis into a more interpretable descriptor study by showing:

1. how the persistence core differs from the learned-advantage core;
2. whether cross-regime-pull cases are closer to one anchor or sit in their own mixed profile;
3. whether transition cases are truly intermediate.

### Why this matters

This is useful for two reasons.

First, it strengthens the data-analysis side of the project by making the ambiguous populations more concrete.

Second, it helps the later methodological direction, because a future regime-conditioned forecasting method could use these populations differently:

- high-confidence persistence-like cases;
- high-confidence learned-correction cases;
- and uncertain boundary cases.

### Implementation note

The SRD workflow now also includes:

- `scripts/analyze_soft_regime_profiles.py`

which reads the soft-membership output and produces:

1. profile-level summaries,
2. tier- and horizon-wise composition tables,
3. and descriptor distribution figures for the core and ambiguous populations.

## 2026-07-03: Anchor Separation And Pull Predictors Added

The soft regime profiles clarified the four main populations, but one more question remained:

- which descriptors most strongly separate the two anchor populations?
- and which descriptors are most associated with entering the ambiguous pool?

That question is now addressed explicitly.

### What this layer does

The new analysis has two parts.

1. **Anchor separation**

It directly compares:

- `both_easy_core`
- `target_wins_core`

and ranks descriptors by standardized mean gap.

This tells us which variables most cleanly separate the persistence-support anchor from the learned-advantage anchor.

2. **Ambiguity predictors**

It then uses simple interpretable classifiers to predict:

- `cross_regime_pull` vs anchor-core cases
- `transition` vs anchor-core cases

This moves the analysis from descriptive grouping to explanatory probing:

- not only what the groups look like,
- but which descriptors are most associated with ambiguous or unstable regime support.

### Why this matters

This is important because it helps identify:

1. which descriptors define the main anchor regimes;
2. which descriptors are responsible for regime ambiguity;
3. which variables are most promising for any later regime-conditioned forecasting design.

### Implementation note

The SRD workflow now also includes:

- `scripts/analyze_anchor_separation.py`

and this produces:

1. anchor-separation tables,
2. cross-regime-pull predictor coefficients,
3. transition predictor coefficients,
4. and compact figures for the most informative descriptors.

## 2026-07-03: Robustness Phase Started

After consolidating the SRD descriptor story, the next step was to ask whether that story survives skeptical scrutiny.

Three concrete risks were identified:

1. sensitivity to hard and soft threshold choices;
2. partial circularity from using composite activity/structure scores in later predictor analyses;
3. optimistic predictive estimates caused by row-level rather than patient-level validation.

The robustness phase has therefore started with two explicit additions.

### 1. Threshold sensitivity analysis

The project now includes:

- `scripts/analyze_threshold_sensitivity.py`

This script sweeps:

- hard case-label thresholds
- and soft regime-membership thresholds

and checks whether the main qualitative story survives, including:

1. whether `both_easy` and `target_wins` remain the two stable anchor populations;
2. whether the persistence and learned-core fractions remain reasonably high;
3. whether horizon `3` still shows more cross-regime pull than horizon `1`;
4. whether Tier `B` still behaves more transition-like than Tier `A`.

This is intended to test whether the current findings are threshold artifacts or stable qualitative structure.

### 2. Independence-aware and raw-feature-aware predictor analysis

The anchor-separation script:

- `scripts/analyze_anchor_separation.py`

has been extended with:

1. `feature_mode = full | raw_only`
2. optional grouped validation via `group_col`

This allows the same ambiguity-predictor analysis to be rerun:

- without composite activity/structure scores,
- and with group-aware splits such as `patient_id`.

The purpose is to reduce circularity and to obtain more conservative estimates of how predictive the descriptors really are.

### Why this matters

This robustness phase is the point where the SRD analysis moves from:

- “interesting exploratory story”

toward:

- “defensible descriptor-level evidence.”

The goal is not to eliminate all subjectivity or all threshold dependence, which is unrealistic in a synthetic exploratory setting.

The goal is to show that the main regime picture survives reasonable perturbations and stricter validation choices.

## 2026-07-03: Consolidated SRD Descriptor Evidence Summary

At this point, the SRD data-analysis sprint has moved beyond isolated tables and can now be summarized as a coherent descriptor-level picture of the forecasting problem.

This section freezes the main evidence so that later method work can build on a stable analytical foundation.

### 1. The original five case types are not five equally stable regimes

The analysis now supports a more structured interpretation.

#### Stable anchor populations

Two populations behave like genuine descriptor-level anchors:

1. `both_easy`
2. `target_wins`

These are the two groups whose cases most consistently align with stable descriptor regions.

#### Ambiguity populations

The remaining populations are better interpreted as ambiguity or failure populations rather than equally stable regimes:

1. `both_hard`
2. `close_mixed`
3. `baseline_wins` as a rare edge case

This is an important conceptual correction. It means that the descriptor map is not best viewed as a clean five-class partition.

Instead, it is better viewed as:

- two anchor regimes,
- plus boundary/failure populations around them.

### 2. The descriptor map contains two anchor regimes

The regime-map and soft-membership analyses now support the following anchor interpretation.

#### Persistence-support anchor

`both_easy_core` corresponds to a persistence-friendly operating region.

Typical descriptor profile:

- negative activity score
- negative structure score
- smaller tumor burden
- lower recent growth
- simpler connected-component structure
- treatment often present

#### Learned-advantage anchor

`target_wins_core` corresponds to a learned-model-benefit region.

Typical descriptor profile:

- positive activity score
- positive structure score
- much larger tumor burden
- higher recent growth
- more connected components
- treatment typically absent

### 3. The cross-regime population is large and meaningful

The soft-regime profile results show that:

- `cross_regime_pull` is not a tiny residual group;
- it accounts for roughly `30%` of the analyzed cases.

This matters because it means ambiguity is not a corner case. It is a major part of the short-horizon forecasting landscape.

The key profile-level finding is that `cross_regime_pull` is **not** strongly persistence-like.

Compared with `both_easy_core`, it shows:

- much higher activity,
- much higher structure complexity,
- much larger volume,
- more connected components,
- and more irregular morphology.

But compared with `target_wins_core`, it is still less extreme in the main activity/burden descriptors.

Interpretation:

- `cross_regime_pull` looks like a partially activated, structurally messy population that leans toward the learned side without cleanly settling into the learned anchor.

### 4. Transition is a smaller but distinct intermediate population

`transition` is much smaller than `cross_regime_pull`, but it appears to reflect a different ambiguity mechanism.

The evidence suggests:

- it is not simply the same as cross-regime pull;
- it is more strongly associated with mixed descriptor support rather than confident pull toward one anchor;
- it is especially associated with Tier `B`.

This suggests that the ambiguous middle of the forecasting map may itself contain at least two submodes:

1. a strong cross-regime pull toward one anchor;
2. a more genuinely intermediate transition zone.

### 5. Tier structure is meaningful and not cosmetic

The soft-regime profile analysis shows a clear tier interpretation.

#### Tier A

Tier `A` is strongly associated with the persistence-support anchor.

`both_easy_core` is dominated by Tier `A`.

#### Tier C

Tier `C` is strongly associated with the learned-advantage anchor.

`target_wins_core` is dominated by Tier `C`.

#### Tier B

Tier `B` contributes disproportionately to the transition population.

Interpretation:

- Tier `A` behaves like the clearest persistence regime;
- Tier `C` behaves like the clearest learned-advantage regime;
- Tier `B` appears to be the strongest mixed or transitional regime.

This is one of the clearest validations so far that the tier construction is analytically meaningful and not arbitrary.

### 6. Horizon changes regime composition, not just difficulty

One of the strongest findings in the data sprint is that increasing horizon changes the descriptor-regime composition of the cases.

The key pattern is:

- at horizon `1`, anchor-aligned cases dominate;
- by horizon `3`, cross-regime-pull becomes as common as or more common than anchor alignment.

This means that longer short-horizon forecasting is not just “the same cases becoming harder.”

It is also:

- a compositional shift away from clean anchor support,
- and toward descriptor ambiguity.

This is a stronger statement than a generic “accuracy decreases with horizon” observation.

### 7. What most strongly separates the two anchor populations

The anchor-separation analysis gives a very clear ranking of anchor descriptors.

Top standardized separators:

1. `activity_score`
2. `input_volume_vox`
3. `treated_at_input`
4. `recent_relative_growth`
5. `structure_score`
6. `input_connected_component_count`

Interpretation:

The persistence-vs-learned split is governed primarily by:

- activity,
- burden,
- treatment state,
- recent growth,
- and structural complexity.

By contrast:

- `delta_days` is only a weak anchor separator;
- `input_elongation_ratio` is almost negligible for distinguishing the anchors.

This is important because it narrows the descriptor story substantially.

### 8. What predicts cross-regime pull

Cross-regime pull is moderately predictable from forecast-time descriptors, with meaningful but not overwhelming signal.

The strongest positive predictors are:

1. `input_connected_component_count`
2. `input_compactness_proxy`
3. `input_elongation_ratio`
4. `structure_score`
5. `horizon_3`

Important negative/offsetting signals include:

- very large `input_volume_vox`
- treatment-at-input
- horizon `1`

Interpretation:

Cross-regime pull is associated less with the most extreme learned-anchor cases and more with:

- structural messiness,
- morphological irregularity,
- and horizon-driven ambiguity.

This is one of the clearest indications that ambiguity is driven by structural complexity rather than only by raw tumor burden.

### 9. What predicts transition

Transition cases are also moderately predictable, but their signature differs from cross-regime pull.

The strongest signals include:

1. lack of treatment-at-input
2. lower recent relative growth
3. higher compactness proxy
4. more connected components
5. lower `delta_days`
6. Tier `B`

Interpretation:

Transition cases appear to be:

- structurally irregular,
- not strongly persistence-like,
- but also not fully activated learned-core cases.

They look more like a middle-zone population with mixed descriptor support.

### 10. Refined descriptor hierarchy

The data sprint now supports a refined descriptor hierarchy.

#### Primary descriptors

These now appear central to the SRD forecasting-regime story:

1. `activity_score` or its underlying activity components
2. `input_volume_vox`
3. `treated_at_input`
4. `recent_relative_growth`
5. `structure_score`
6. `input_connected_component_count`

#### Secondary descriptors

These matter more for ambiguity structure than for anchor separation:

1. `input_compactness_proxy`
2. `input_elongation_ratio`
3. `delta_days`

### 11. Current best research interpretation

The strongest current interpretation is:

1. short-horizon tumor forecasting is not a uniform problem;
2. it contains at least two stable operating regimes:
   - persistence-support
   - learned-advantage
3. it also contains ambiguity populations that are:
   - descriptor-driven,
   - horizon-sensitive,
   - and not well represented by a rigid hard-label taxonomy.

This is now a substantially stronger result than the original framing of simply comparing models by tier or horizon.

### 12. Immediate implication for later method work

When the project transitions from the data-analysis sprint to method development, the first regime-conditioned forecasting design should be based on this descriptor hierarchy.

The most defensible first conditioning variables are now:

1. activity-related descriptors
2. tumor burden
3. treatment-at-input
4. recent growth
5. structure complexity

The method question should not be:

- can we invent a completely new forecasting model from scratch?

It should first be:

- can these regime descriptors help an existing forecasting model decide whether a case is persistence-like, learned-core-like, or ambiguity-prone?

That is the current bridge from SRD analysis to a later regime-conditioned forecasting model.

It is about:

- deciding what the present evaluation framework can genuinely claim,
- learning enough from the literature to avoid shallow or derivative next steps,
- and preparing the transition from a regime-aware analysis project to a regime-conditioned forecasting project.

## 2026-06-30: Research-First Standard Reaffirmed

The project standard is being explicitly reset here.

The goal is **not**:

- to rush a paper for a workshop deadline;
- to force a submission within one month;
- to interpret weak or incomplete evidence generously just because a deadline exists;
- or to artificially constrain the project to what seems easiest in a short window.

The goal **is**:

- to build a body of work that is analytically meaningful;
- to make the work defensible under close questioning;
- to keep the methodology and evidence reproducible;
- and to pursue a contribution that genuinely stands up, whether that ends up being sooner or later.

This means all future judgments should be made from a **research-first** perspective rather than a deadline-first perspective.

## Updated decision rule

From this point onward, choices in the project should not be justified by:

- "we only have one month,"
- "this is enough for a workshop,"
- or "this is probably all we can do before submission."

Instead, choices should be justified by questions like:

1. Does this improve the rigor of the current evidence?
2. Does this make the final claims more defensible?
3. Does this help us understand tumor forecasting more deeply?
4. Does this move the work toward an actual analytical or methodological contribution?
5. Would this still be worth doing if there were no immediate submission deadline?

If the answer is no, then the task should be deprioritized even if it seems convenient for a short-term paper push.

## Expanded scope of the research problem

The tumor-forecasting problem should not be treated narrowly.

There is substantial room to study:

- how different forecasting methods encode tumor information;
- how much information is lost when using 2D slices instead of 3D volumes;
- which tumor properties matter most at short horizons;
- how longitudinal structure is represented across model families;
- how persistence, morphology, treatment, temporal spacing, and growth heterogeneity interact;
- whether probabilistic or field-based representations can better capture the forecasting problem;
- and how regime-aware or prior-aware modeling can use this information more intelligently.

This means the regime-aware framework is not viewed as a finished endpoint. It is a foundation from which broader analytical and methodological directions can emerge.

## Role of literature going forward

The literature-review track is now a central part of the research program rather than a side activity.

The purpose is not only to collect citations, but to understand in depth:

- how tumor forecasting has been formulated historically;
- how tumor properties are represented across methods;
- how priors, conditioning, physics, uncertainty, and temporal information are handled;
- which modeling assumptions are standard and which remain open;
- and where a regime-aware perspective can make a genuinely original contribution.

The `References` document should therefore function not just as a bibliography staging file, but as an evolving guide to the conceptual backbone of the project.

## Practical implication

This project is now being treated as a serious long-horizon research effort.

Missing a near-term workshop or competition is acceptable.

Producing work that is shallow, weakly justified, or not robust is not acceptable.

The standard is therefore:

- aim high,
- go deeper where needed,
- keep the work reproducible and testable,
- and prefer a slower, stronger contribution over a faster, thinner one.

## 2026-06-30: Regress Audit of the Final Synthetic Stack

This entry audits the synthetic-data setup we actually ended up with after the DMS-oriented sprint, the current state of the analysis built on top of it, and the most important directions for strengthening the work.

The purpose of this audit is not to defend the current state automatically. It is to state clearly:

- what is already solid,
- what is only partially convincing,
- what is currently stitched together,
- and what we should improve before treating the work as a mature research contribution.

## What synthetic data we actually ended up with

At this point, the project is not built around one single synthetic artifact in the strictest possible sense. It is built around two closely related synthetic environments that ended up serving different roles.

### 1. Final generalized synthetic benchmark: SRD

The main final benchmark used in the current draft is:

- `fixed_dataset_v3_lite_generalized`

This should be treated as the current **Synthetic Regime Dataset (SRD)** for the project.

Its purpose is:

- to provide the main compact benchmark,
- to support the headline model comparisons,
- and to represent the current best generalized synthetic environment for short-horizon forecasting.

Its defining features are:

- longer histories than the early sprint benchmark;
- higher treatment prevalence;
- broader temporal spacing;
- larger lesion scale;
- stronger representation of stable, mixed, and aggressive cases;
- and a three-regime interpretation that is no longer tied to a simple "easy-medium-hard" ladder.

The three regimes in SRD are best interpreted as:

- `A`: stability / persistence regime
- `B`: mixed compromise regime
- `C`: aggressive heterogeneous growth stress regime

### 2. Auxiliary analysis synthetic dataset: `biophys1`

The richer regime-driver and cross-regime analyses were carried out on:

- `fixed_dataset_v2_core_candidate_biophys1`

This dataset currently supports:

- cross-regime transfer analysis,
- case-type analysis,
- morphology/growth/treatment summaries,
- and the clearest evidence that different synthetic regimes behave differently rather than simply lying on one monotone difficulty axis.

In practice, this means our synthetic story currently has a **two-layer structure**:

1. SRD carries the final main benchmark results.
2. `biophys1` carries much of the deeper interpretive analysis.

That is not automatically wrong, but it is a real structural issue that must be handled carefully.

## What the final generator is actually doing

The synthetic generator is stronger than a random toy setup, but it is also more stylized than the paper should ever imply.

### Regime family split

The most important structural fact is:

- `Tier A` is not just a low-parameter PDE case.
- `Tier A` is a **different simulation family**.

Specifically:

- `Tier A` uses procedural geometric mask evolution.
- `Tier B` uses isotropic reaction-diffusion evolution.
- `Tier C` uses anisotropic and heterogeneous reaction-diffusion evolution.

This matters a lot.

It means the tiers encode not only different parameter ranges, but also different generative mechanisms.

That gives the benchmark expressive value, because it creates qualitatively different forecasting regimes.
At the same time, it also means the tier structure is partly hand-designed rather than emerging only from one unified mechanistic model.

### Core controllable variables

Across the final SRD YAML, the main variables we control are:

- number of sessions per patient;
- interval between scans;
- treatment prevalence and onset timing;
- growth rate `rho`;
- diffusion strength `Dw`;
- treatment sink strength;
- number of initial foci;
- initial lesion scale via `sigma`;
- initial lesion amplitude;
- and, for the more mechanistic tiers, anisotropy and heterogeneity through the simulator pathway.

### What treatment currently means

Treatment is currently encoded as:

- a binary per-session on/off flag;
- sampled at the patient level with a treatment-start session;
- and implemented in the PDE regimes as a simple sink term that reduces growth mass once treatment is active.

This is a useful control, but it remains stylized.
It does **not** yet encode:

- response heterogeneity,
- delayed response,
- rebound,
- nonresponse,
- regimen switching,
- or observation-time treatment uncertainty.

### What the benchmark is good at representing

The current generator is good at representing:

- persistence-heavy short-horizon cases;
- moderate-growth longitudinal cases;
- aggressive heterogeneous expansion cases;
- treatment-affected attenuation in a simplified form;
- and a meaningful contrast between geometric and reaction-diffusion-like evolution.

### What the benchmark is still weak at representing

The generator is still weak at representing:

- explicit response phenotypes;
- clinically grounded rebound patterns;
- irregular observation processes;
- registration and acquisition noise seen in real scans;
- calibrated physical units tied to patient-specific geometry;
- and a unified family of realistic growth modes under one mechanistic formulation.

## What is already strong in the current synthetic analysis

Several parts of the current synthetic investigation are genuinely useful and should be kept.

### Strength 1. The regime interpretation is no longer naive

One of the most important improvements we made was moving away from:

- `A = simple`
- `B = medium`
- `C = most realistic`

and toward:

- `A = stability / persistence regime`
- `B = mixed transferable regime`
- `C = aggressive stress regime`

This reinterpretation is analytically stronger and much more defensible.

### Strength 2. The final SRD produces a nontrivial short-horizon story

On the final SRD:

- `LOCF` is still strong overall;
- `UNet-image+mask` only modestly improves on it;
- `ResUNet-image+mask` improves more clearly;
- and the gains are not uniform across regimes or horizons.

This is exactly the kind of structure we need:

- persistence is not trivial,
- learned gains are not automatic,
- and model behavior depends on condition.

### Strength 3. Tier-wise behavior is meaningful

The tier-wise summaries already show a useful pattern:

- Tier `A` is nearly tied between `LOCF` and `ResUNet`.
- Tiers `B` and `C` produce clearer learned-model gains.

This is one of the strongest analytical results in the project so far, because it supports the idea that short-horizon forecasting difficulty is regime-dependent.

### Strength 4. Horizon-wise behavior is meaningful

The horizon breakdown on SRD is also useful:

- learned models are not only helping at the longest horizon;
- `ResUNet` is better across horizons `1`, `2`, and `3`;
- but the magnitude and interpretation of those gains still depend on the regime mix.

This makes the synthetic environment more interesting than an average-only benchmark.

### Strength 5. Cross-regime transfer and case analysis were the right move

The `biophys1` analyses were scientifically valuable because they moved the project beyond:

- average Dice tables,
- and simple model ranking.

Those analyses started to answer:

- which regimes transfer,
- which ones are specialized,
- and which case properties are associated with learned-model wins.

That is the right analytical direction for the project.

## What is still weak or incomplete

This is the part we should be most honest about.

### Weakness 1. The synthetic story is currently split across two datasets

Right now:

- SRD carries the headline benchmark;
- `biophys1` carries much of the deeper regime evidence.

This creates a narrative risk:

- the final benchmark and the explanatory analysis are not yet fully unified.

That does not invalidate the work, but it means we have not yet completed the cleanest possible synthetic evidence chain.

If we want a stronger contribution, we should reduce this split.

### Weakness 2. The final claims currently depend more on saved summaries than on a consolidated artifact bundle

Locally, the repo preserves:

- generated tables,
- generated figures,
- and paper-facing summary documents.

But the raw run outputs for every claimed experiment are not fully consolidated inside the repo workspace.

That is a reproducibility concern.

It means the current state is still somewhat dependent on:

- experiment logs,
- copied tables,
- and prior Colab/Drive outputs,

rather than one fully frozen, easily rerunnable local artifact tree.

### Weakness 3. Tier semantics are meaningful, but still partly hand-authored

The tiers are not simply discovered from data.
They are designed through:

- YAML ranges,
- generator choices,
- and distinct simulation families.

That is acceptable for a controlled synthetic environment, but it means we should never overstate the regimes as if they were natural clinical subtype discoveries.

### Weakness 4. Treatment remains too simplified

The treatment variable is useful, but still very coarse.

At the moment, it mainly acts as:

- an on/off switch,
- with an immediate sink-like effect in PDE evolution.

This is enough for a controlled synthetic probe, but not enough for stronger biological claims.

### Weakness 5. We still do not have a final "difficulty-driver map" on the exact main benchmark

The most interpretive analyses currently live on `biophys1`.
What we still need is the same level of driver analysis on the final SRD itself:

- case types,
- win/loss structure,
- growth-conditioned behavior,
- treatment-conditioned behavior,
- morphology-conditioned behavior,
- and within-tier heterogeneity.

Until that is done, the final benchmark is still better at **showing** regime dependence than at **explaining** it.

### Weakness 6. The current synthetic analysis still leans more descriptive than causal

We now know that performance differs by tier, horizon, and case properties.
But we do not yet have the strongest version of the next question:

- which forecast-origin properties are actually sufficient to predict when persistence will fail?

That is the bridge from evaluation into a real methodological contribution.

## Blunt assessment of where we stand

The current synthetic stack is not weak, but it is not fully closed either.

The honest assessment is:

- the generator is coherent enough to support serious controlled experiments;
- the final SRD is good enough to serve as a main synthetic environment;
- the current analyses already point to a real scientific question;
- but the total synthetic story is still not yet as integrated, deterministic, and explanatory as it should become.

In other words:

- we do have a real synthetic research foundation;
- we do **not** yet have the cleanest final synthetic evidence package.

## What should be enhanced next

The next improvements should be chosen to close the exact weaknesses above, not to expand scope randomly.

### Priority 1. Unify the synthetic story around SRD

We should decide explicitly whether:

1. SRD remains the sole main synthetic environment going forward, with deeper analyses rerun on it;
2. or SRD and `biophys1` are both kept, but with clearly separated roles.

My current judgment is that the stronger path is:

- keep SRD as the main synthetic dataset,
- and rerun the most important regime-driver analyses on SRD itself.

That would remove a major narrative fracture.

### Priority 2. Build a final SRD audit bundle

For the exact main SRD used in the paper/project, we should generate and freeze:

- patient/session/transition summaries;
- tier-wise temporal summaries;
- tier-wise growth summaries;
- morphology summaries;
- treatment summaries;
- and forecastable-sample counts by horizon and tier.

This should exist as one reproducible audit artifact set, not just as scattered tables.

### Priority 3. Quantify within-tier heterogeneity

The tiers are useful, but each tier is still broad.
We should explicitly measure within-tier spread for:

- baseline volume;
- recent growth;
- future growth;
- interval length;
- treatment-at-input;
- elongation;
- compactness;
- and component count.

This will help answer whether the tier labels are too coarse by themselves.

### Priority 4. Recompute pairwise driver analysis on SRD

For the main model pair we care about most, namely:

- `LOCF` vs `ResUNet-image+mask`,

we should run the full driver analysis on SRD and produce:

- by-tier win rates;
- by-horizon win rates;
- by volume bin;
- by recent growth bin;
- by future growth bin;
- by treatment state;
- and case-type summaries.

This is one of the highest-value next steps.

### Priority 5. Define forecast-origin regime descriptors explicitly

We should stop speaking about "regime" only at the tier label level.

Instead, we should define a forecast-origin descriptor set such as:

- input volume;
- recent relative growth;
- treatment-at-input;
- elapsed interval to target;
- compactness proxy;
- elongation ratio;
- connected-component count;
- and maybe one simple uncertainty/stability indicator.

This descriptor set can later become the prior/conditioning vector for a regime-conditioned method.

### Priority 6. Freeze determinism wherever possible

Going forward, we should make the synthetic pipeline as non-accidental as possible.

That means:

- fixed dataset seeds;
- fixed split seeds;
- frozen YAML configs;
- frozen run manifests;
- frozen exported tables;
- and reruns only when deliberately changing one controlled factor.

Randomness should be used only when it is scientifically intentional and then reported clearly.

### Priority 7. Treat PDE synthetic evaluation as optional but valuable

If we include the PDE baseline in the final research story, it should also be evaluated on the same final SRD environment rather than appearing only in the real-data section.

That would help complete the model-family comparison:

- persistence,
- standard learned,
- residual learned,
- and mechanism-guided.

But this should only be done if we can preserve consistency and artifact quality.

## Immediate actionable next block

The best next research block on the synthetic side is:

1. freeze SRD as the main synthetic environment;
2. regenerate a full SRD audit artifact set;
3. rerun the regime-driver and case-type analyses on SRD;
4. summarize which forecast-origin descriptors most strongly track persistence failure;
5. decide from that evidence what the first regime-conditioned prior should be.

That block directly supports both:

- a stronger analytical paper backbone,
- and the transition into a regime-conditioned forecasting method.

## Final audit verdict

The synthetic data we ended up with is **good enough to build on seriously**, but it is **not yet fully exploited analytically**.

The main weakness is no longer that the generator is too toy-like.
The main weakness is that our evidence is still somewhat split, partially consolidated, and not yet pushed to the point where it naturally yields the next method.

That is actually encouraging.

It means the highest-value work now is not throwing away the synthetic setup.
It is:

- tightening it,
- auditing it properly,
- extracting a cleaner descriptor-driven story from it,
- and using that story to motivate the next forecasting method.

## 2026-07-01: SRD Regime-Analysis Workflow Consolidation

To support the next refinement block, the SRD-side analysis pipeline has now been consolidated into a single reproducible workflow script:

- `scripts/run_srd_regime_analysis.py`

### Why this was needed

Before this step, the synthetic-regime investigation existed across several separate utilities:

- dataset audit
- pairwise regime-driver analysis
- case-type analysis
- morphology/treatment analysis
- figure export

All of those pieces were useful, but they were still too easy to run inconsistently or leave partially scattered across folders.

That is exactly the kind of looseness we want to reduce now.

### What the consolidated workflow does

The new workflow runs, in order:

1. dataset audit
2. pairwise regime-driver analysis
3. case-type analysis
4. morphology/treatment analysis
5. figure export

and writes them into one output bundle with:

- a summary JSON manifest
- a short bundle report
- subdirectories for each analysis stage

This is meant to become the standard way of running the final SRD-facing synthetic analysis.

### Additional analytical refinement added

As part of this consolidation step, the audit and downstream analysis now include:

- `connected_component_count`

This gives us one more morphology/fragmentation descriptor at the forecast origin, alongside:

- volume
- elongation
- compactness
- treatment state
- temporal spacing
- and recent growth

This matters because regime descriptors should eventually be defined through measurable forecast-origin properties, not only through the tier label.

### Immediate implication

The next SRD experiment block should use this consolidated workflow rather than manually chaining older scripts.

That will help us:

- keep one frozen artifact tree per experiment pass;
- compare iterations more cleanly;
- and avoid mixing partially regenerated outputs with older results.

## 2026-07-02: First Consolidated SRD Regime-Analysis Pass

The first consolidated SRD regime-analysis bundle was run on:

- dataset: `fixed_dataset_v3_lite_generalized`
- baseline family: `v3lite_compact_h123_s42`
- pairwise comparison: `LOCF` vs `ResUNet-image+mask`

### Key results

#### By tier

- `A`
  - count: `48`
  - win rate: `0.458`
  - mean Dice gap: `-0.00036`

- `B`
  - count: `39`
  - win rate: `0.718`
  - mean Dice gap: `0.0924`

- `C`
  - count: `36`
  - win rate: `0.667`
  - mean Dice gap: `0.1323`

#### By horizon

- `h=1`
  - count: `41`
  - win rate: `0.585`
  - mean Dice gap: `0.0449`

- `h=2`
  - count: `41`
  - win rate: `0.634`
  - mean Dice gap: `0.0909`

- `h=3`
  - count: `41`
  - win rate: `0.585`
  - mean Dice gap: `0.0678`

#### Case types

- `both_easy`: `59 / 123` (`47.97%`)
- `target_wins`: `35 / 123` (`28.46%`)
- `both_hard`: `15 / 123` (`12.20%`)
- `close_mixed`: `13 / 123` (`10.57%`)
- `baseline_wins`: `1 / 123` (`0.81%`)

### Interpretation

These results strengthen several parts of the current synthetic story.

1. Tier `A` really is a persistence regime.
   - `LOCF` and `ResUNet` are effectively tied there.
   - This supports the `A = stability / persistence` interpretation rather than an "easy tier" reading.

2. Learned gains are concentrated in the non-stable regimes.
   - Tier `B` shows the highest win rate for `ResUNet`.
   - Tier `C` shows the largest mean Dice gap.
   - Together, this suggests that `B` and `C` capture different but both useful learned-model opportunity regimes.

3. The final SRD is not trivialized into "deep model always wins."
   - Nearly half the cases are `both_easy`.
   - This means persistence still handles a large share of the benchmark well.
   - That is scientifically useful because it preserves a meaningful role for the baseline.

4. The horizon story is real but not monotone.
   - `ResUNet` beats `LOCF` on average at all three horizons.
   - The largest average gain occurs at horizon `2`, not simply at the farthest horizon.
   - This suggests that forecast difficulty is shaped by regime composition and case structure, not only by temporal distance.

5. Pure baseline-only wins are now extremely rare.
   - Only `1` case out of `123` is a clear `baseline_wins` case under the current thresholds.
   - This means SRD has shifted well away from the earlier synthetic settings where persistence dominated almost automatically.

### What this clarifies

This first consolidated pass makes the main SRD story cleaner:

- Tier `A` behaves like a persistence regime.
- Tiers `B/C` are where learned corrections matter.
- The main question is no longer whether regime dependence exists.
- The main question is which forecast-origin descriptors best separate:
  - `both_easy`
  - `target_wins`
  - `both_hard`
  - the rare `baseline_wins`

### Immediate next analytical need

The next high-value probe should focus on the case-type breakdown in more detail:

1. case type by tier
2. case type by horizon
3. case type by recent growth
4. case type by future growth
5. case type by treatment-at-input
6. case type by morphology descriptors, including connected-component count

That is the most direct route toward defining a robust forecast-origin regime descriptor set for the later regime-conditioned forecasting method.

## 2026-07-02: SRD Descriptor-Level Findings From Case-Type Analysis

The next SRD inspection step moved from aggregate win rates into descriptor-conditioned case-type structure.

The main objective here was to understand whether the synthetic regime story can be expressed through forecast-origin properties rather than only through tier labels.

### Key findings

#### 1. `target_wins` cases are concentrated in tiers `B` and `C`

Case-type by tier showed:

- `target_wins`
  - `A`: `1` case (`2.86%`)
  - `B`: `14` cases (`40.0%`)
  - `C`: `20` cases (`57.1%`)

- `both_easy`
  - `A`: `36` cases (`61.0%`)
  - `B`: `13` cases (`22.0%`)
  - `C`: `10` cases (`16.9%`)

- `close_mixed`
  - overwhelmingly concentrated in `A`

Interpretation:

- Tier `A` really is the main persistence-sufficient regime.
- Tier `C` is the richest source of clear learned-model gains.
- Tier `B` remains an important mixed regime where learned gains occur often, but not as exclusively as in `C`.

#### 2. The largest concentration of `target_wins` occurs at horizon `2`, not only at the farthest horizon

Case-type by horizon showed:

- `target_wins`
  - `h=1`: `10`
  - `h=2`: `15`
  - `h=3`: `10`

- `both_hard`
  - increases toward `h=3`

Interpretation:

- learned gains are not simply a monotone function of forecasting farther out;
- horizon `2` appears to be a particularly informative compromise point where persistence begins to weaken but cases are not yet dominated by both-model difficulty.

#### 3. Input volume is one of the clearest forecast-origin separators

Case-type by input-volume bin showed:

- `target_wins`
  - `large`: `23 / 35` (`65.7%`)
  - `medium`: `11 / 35` (`31.4%`)
  - `small`: `1 / 35` (`2.9%`)

- `both_easy`
  - heavily concentrated in `small` and `medium`

Interpretation:

- larger tumors are much more likely to benefit from the learned residual model;
- small tumors are much more likely to remain persistence-sufficient or ambiguous;
- input tumor burden should almost certainly be one of the first regime descriptors kept in the later forecasting prior.

#### 4. Recent growth is also a strong forecast-origin separator

Case-type by recent-growth bin showed:

- `target_wins`
  - `high`: `21 / 35` (`60.0%`)
  - `medium`: `13 / 35` (`37.1%`)
  - `low`: `1 / 35` (`2.9%`)

- `both_easy`
  - `low`: `30 / 59` (`50.8%`)

Interpretation:

- when recent growth is already high at the forecast origin, the learned model is much more likely to provide value;
- low recent growth strongly aligns with persistence sufficiency;
- recent growth should be treated as a high-priority descriptor for any regime-conditioned method.

#### 5. Treatment at input is strongly associated with persistence-sufficient cases

Case-type by treatment-at-input showed:

- `target_wins`
  - untreated at input: `35 / 35`
  - treated at input: `0 / 35`

- `both_easy`
  - untreated: `36`
  - treated: `23`

- `both_hard`
  - mostly untreated, but not exclusively

Interpretation:

- in the current SRD, clear learned-model wins occur entirely in untreated-at-input cases;
- treatment appears to stabilize or simplify many trajectories enough that persistence remains competitive;
- treatment-at-input is therefore not just a metadata field but a meaningful regime descriptor.

This finding should still be interpreted carefully because treatment is simplified in the generator.

#### 6. Connected-component count looks promising as a morphology descriptor

From the morphology summary and component-bin breakdown:

- `target_wins` are concentrated in the `high` component bin (`48.6%`)
- `both_hard` are even more concentrated there (`60.0%`)
- `both_easy` are more concentrated in `low` and `medium` bins

Interpretation:

- lesion fragmentation or multi-component structure appears to matter;
- higher component complexity does not guarantee success for the learned model, but it clearly marks a departure from the easiest persistence-dominated cases;
- connected-component count should be retained in the forecast-origin descriptor set.

#### 7. Compactness and morphology appear useful, but less clean than volume and recent growth

The morphology numeric summary suggests that:

- `target_wins` tend to have higher connected-component counts than `both_easy`;
- `both_hard` tend to show even more extreme complexity;
- compactness and elongation vary across case types, but not yet as cleanly as volume and recent growth.

Interpretation:

- morphology matters, but the first descriptor set should probably prioritize:
  - input volume,
  - recent growth,
  - treatment-at-input,
  - and connected-component count

before relying too heavily on noisier secondary morphology features.

### Important methodological note

Some analyses elsewhere in the workflow use future growth for explanation.
That is valid for scientific interpretation, but future growth is **not available at forecast time** and therefore cannot be used directly as a regime-conditioning prior.

So the distinction should now be explicit:

- explanatory descriptors:
  - future growth
  - future delta volume

- forecast-origin descriptors:
  - input volume
  - recent growth
  - treatment-at-input
  - time interval to target
  - compactness
  - elongation
  - connected-component count

Only the second group is eligible for the later regime-conditioned model.

### Updated working hypothesis

The current SRD evidence suggests that short-horizon forecasting behavior is strongly shaped by a small forecast-origin descriptor set:

1. tumor burden at input
2. recent growth at input
3. treatment state at input
4. morphology/fragmentation complexity

This is the first concrete descriptor-level candidate for a regime-conditioned forecasting prior.

### Immediate next step

The next analysis should move from grouped tables to a compact predictive probe:

- fit a simple interpretable model using only forecast-origin descriptors to predict:
  - `target_wins` vs `both_easy`
  - or more generally whether `ResUNet` is likely to outperform `LOCF`

The purpose is not to build a final classifier for publication, but to test whether the descriptor set already carries enough signal to support a later regime-conditioned forecasting method.

## 2026-07-02: Descriptor-Signal Probe Added

The next refinement step has now been implemented in code.

A new script:

- `scripts/analyze_descriptor_signal.py`

has been added and wired into the consolidated SRD workflow.

### Purpose

This probe studies whether forecast-origin descriptors alone carry enough signal to explain or predict regime behavior.

It explicitly avoids using future-only explanatory variables and focuses on descriptors available at the forecast origin.

### Forecast-origin descriptor set used

The current first-pass descriptor set is:

- `input_volume_vox`
- `recent_relative_growth`
- `treated_at_input`
- `delta_days`
- `input_elongation_ratio`
- `input_compactness_proxy`
- `input_connected_component_count`
- `n_sessions`
- `followup_days`
- `mean_interval_days`

### Tasks included

The probe currently evaluates three binary tasks:

1. `resunet_beats_locf`
   - whether `ResUNet` outperforms `LOCF`

2. `target_wins_vs_both_easy`
   - separates cases where learned correction clearly helps from cases where both methods are already easy

3. `both_hard_vs_both_easy`
   - separates structurally difficult cases from persistence-sufficient easy cases

### Models used

The probe intentionally uses simple interpretable models:

- standardized logistic regression
- shallow decision tree (`max_depth=3`)

### Why this matters

This is the first step that directly tests whether the regime story can be recovered from measurable tumor state rather than from hand-authored tier labels.

If the descriptor signal is strong, then we have a more rigorous basis for:

- defining regime descriptors;
- deciding which features should become a forecasting prior;
- and later comparing plain forecasting against regime-conditioned forecasting.

### Workflow integration

The consolidated SRD runner now calls this descriptor-signal probe automatically unless explicitly skipped.

This means the SRD artifact bundle can now contain:

- audit summaries
- pairwise regime summaries
- case types
- morphology/treatment summaries
- figure exports
- descriptor-signal outputs

in one reproducible run.

## 2026-07-03: Descriptor-Signal Probe Findings

The first descriptor-signal probe has now produced interpretable feature rankings from forecast-origin descriptors alone.

The attached output filenames suggest the three logistic-importance tables correspond to:

1. `resunet_beats_locf`
2. `target_wins_vs_both_easy`
3. `both_hard_vs_both_easy`

That mapping should be preserved explicitly in later saved output directories, but even at this stage the qualitative pattern is already informative.

### Overall descriptor-only predictability

The simple descriptor-only models already carry meaningful signal:

- shallow decision tree:
  - accuracy: `0.759`
  - balanced accuracy: `0.753`
  - ROC-AUC: `0.808`

- logistic regression:
  - accuracy: `0.687`
  - balanced accuracy: `0.680`
  - ROC-AUC: `0.750`

Interpretation:

- forecast-origin descriptors alone are not trivial or weak;
- they already support useful discrimination of forecasting regimes;
- a simple nonlinear model performs notably better than linear logistic regression, which suggests that regime structure is at least partly interaction-based rather than purely linear.

### Task 1: Predicting whether `ResUNet` beats `LOCF`

Top logistic features:

- `recent_relative_growth` (`+1.19`)
- `n_sessions` (`-0.61`)
- `input_connected_component_count` (`+0.59`)
- `input_compactness_proxy` (`+0.42`)
- `input_elongation_ratio` (`-0.40`)

Interpretation:

- recent growth is the strongest single descriptor for learned-model advantage;
- structural complexity also matters, especially fragmentation/component count;
- persistence becomes relatively less favorable when recent growth and structural complexity are high;
- the negative sign on `n_sessions` suggests that longer accumulated histories may align with more persistence-sufficient settings in the current SRD;
- input volume is less dominant here than expected once recent growth and morphology are already included.

### Task 2: Separating `target_wins` from `both_easy`

Top logistic features:

- `treated_at_input` (`-1.10`)
- `input_volume_vox` (`+1.03`)
- `delta_days` (`+0.66`)
- `recent_relative_growth` (`+0.66`)
- `mean_interval_days` (`-0.46`)
- `input_compactness_proxy` (`+0.45`)

Interpretation:

- this is one of the clearest results in the current project;
- untreated-at-input status strongly distinguishes cases where learned correction helps from cases where both methods are simply easy;
- input volume is a major separator here, much more strongly than in the broader win/loss task;
- larger tumors, stronger recent growth, and longer immediate forecast gaps all push a case toward the learned-win side;
- this strongly supports keeping:
  - treatment-at-input,
  - input volume,
  - recent growth,
  - and local time-to-target
  as core regime descriptors.

### Task 3: Separating `both_hard` from `both_easy`

Top logistic features:

- `input_connected_component_count` (`+1.56`)
- `input_compactness_proxy` (`+1.06`)
- `delta_days` (`+0.81`)
- `treated_at_input` (`-0.63`)

Interpretation:

- this is extremely useful because it separates "learned model helps" from "case is hard for everyone";
- connected-component count is the strongest feature here by a wide margin;
- fragmentation and morphology appear to be major markers of intrinsic case difficulty;
- compactness is also important;
- longer forecast gaps contribute to overall hardness;
- treatment again pushes toward easier/persistence-sufficient behavior.

### Main research takeaway

The descriptor signal is not just "some features matter."
It is more structured:

1. `recent_relative_growth` is the strongest general signal for learned advantage.
2. `input_volume_vox` is especially important for separating learned-win cases from simple persistence-sufficient cases.
3. `treated_at_input` is a major stabilizing descriptor in the current SRD.
4. `input_connected_component_count` is the clearest marker of intrinsic structural difficulty.
5. `input_compactness_proxy` also contributes meaningfully, especially on the harder-case side.

### Updated descriptor grouping

The current evidence supports a natural grouping of forecast-origin descriptors:

#### Group A: Learned-advantage descriptors

- `recent_relative_growth`
- `input_volume_vox`
- `delta_days`
- `treated_at_input`

#### Group B: Structural-hardness descriptors

- `input_connected_component_count`
- `input_compactness_proxy`
- `input_elongation_ratio`

These groups overlap, but they are not identical.

That is important because it suggests that "when learning helps" and "when the case is intrinsically hard" are related but distinct questions.

### Immediate implication

The next regime-conditioned method should probably not use only one scalar regime score.

A stronger design would likely use at least two kinds of regime information:

1. activity / growth state
2. morphology / structural complexity state

This could later support:

- a compact descriptor embedding,
- a two-branch conditioning signal,
- or a light gating mechanism that modulates trust in persistence versus learned correction.

## 2026-07-03: Two-Axis Regime Map Added

The next analysis layer has now been implemented as an explicit two-axis regime map:

- `scripts/analyze_regime_map.py`

### Purpose

The descriptor-signal probe showed that the forecast-origin story is not one-dimensional.

At least two different descriptor families emerged:

1. activity / change state
2. structural complexity state

The regime-map step makes that explicit rather than leaving it only as an interpretation of coefficient tables.

### Current hand-built axes

#### Activity score

Built from signed standardized versions of:

- `input_volume_vox` (`+`)
- `recent_relative_growth` (`+`)
- `delta_days` (`+`)
- `treated_at_input` (`-`)

#### Structure score

Built from signed standardized versions of:

- `input_connected_component_count` (`+`)
- `input_compactness_proxy` (`+`)
- `input_elongation_ratio` (`-`)

### Why this is useful

This creates a more concrete object for the project:

- not just tier labels,
- not just feature-importance tables,
- but an explicit regime state space in which different case types can be located.

This should help answer:

- where `both_easy` cases sit;
- where `target_wins` sit;
- where `both_hard` cases sit;
- and whether the descriptor space suggests natural conditioning or gating structure for a later forecasting method.

### Outputs

The regime-map script produces:

- scored case table
- score summary by case type
- quadrant summary by case type
- quadrant summary by tier
- scatter plot in activity/structure space
- boxplots for activity and structure scores by case type

### Workflow integration

The consolidated SRD runner now supports this regime-map step by default unless explicitly skipped.

This keeps the SRD artifact bundle aligned with the current research direction:

- tier-level summaries
- case-type summaries
- descriptor signal
- and now an explicit two-axis regime representation

## 2026-07-03: First Two-Axis Regime-Map Findings

The first regime-map outputs now give a much clearer structure to the SRD case space.

### Score summaries by case type

Mean scores:

- `target_wins`
  - activity: `0.484`
  - structure: `0.169`

- `both_hard`
  - activity: `0.073`
  - structure: `0.162`

- `both_easy`
  - activity: `-0.321`
  - structure: `-0.158`

- `close_mixed`
  - activity: `-0.002`
  - structure: `0.018`

Interpretation:

- `target_wins` are primarily distinguished by **high activity**, not by extreme structure alone.
- `both_easy` are low on both axes, which is exactly what we would want from a persistence-sufficient regime.
- `both_hard` are not as active on average as `target_wins`, but they remain structurally elevated.
- `close_mixed` sits near the origin, which is consistent with cases that are ambiguous or only weakly separated.

### Quadrants by case type

The most important result here is:

- `target_wins`
  - `highA_highS`: `28 / 35` (`80.0%`)

while:

- `both_easy`
  - `lowA_lowS`: `30 / 59` (`50.8%`)

Interpretation:

- clear learned-model wins are concentrated in the quadrant with both high activity and high structure;
- easy persistence-sufficient cases are concentrated in the low-activity / low-structure quadrant;
- this is the cleanest regime-map separation obtained so far.

`both_hard` are split much more evenly across:

- `highA_highS`
- `highA_lowS`
- `lowA_highS`

Interpretation:

- being hard for both models does not require the full high-activity/high-structure combination;
- it can arise from either activity-driven difficulty or structure-driven difficulty;
- this reinforces the idea that "learned advantage" and "intrinsic hardness" are related but not identical phenomena.

### Quadrants by tier

Tier-to-quadrant structure is also informative:

- `A`
  - mostly `lowA_lowS` (`45.8%`)
  - plus a substantial `lowA_highS` share (`27.1%`)

- `B`
  - much more mixed across all four quadrants

- `C`
  - overwhelmingly `highA_highS` (`72.2%`)

Interpretation:

- Tier `A` remains the main persistence regime, but not because it is uniformly trivial; it includes some structurally elevated cases with low activity.
- Tier `B` behaves like a true mixed regime in descriptor space, which supports its role as a compromise/intermediate regime rather than simply a middle point on a ladder.
- Tier `C` is strongly concentrated in the high-activity/high-structure region, which explains why it is the richest source of learned-model gains.

### Main research takeaway

The two-axis regime map supports a stronger descriptor-based interpretation of short-horizon forecasting:

1. **Low activity + low structure**
   - persistence is usually sufficient
   - many `both_easy` cases live here

2. **High activity + high structure**
   - learned correction is most likely to help
   - most `target_wins` live here

3. **Intermediate or one-axis-high regions**
   - cases are more mixed or hard
   - this is where ambiguity and model failure are more likely

This is useful because it suggests the next methodological step should not be a single hard tier label, but a descriptor-driven conditioning signal built from:

- activity state
- structure state

### Immediate next step

The next research step should be to formalize this regime state in one of two ways:

1. a compact two-dimensional conditioning vector
2. a four-quadrant discrete regime indicator derived from the two scores

Then we can test whether conditioning a forecast model on this state improves:

- stability in mixed cases
- gains in high-activity/high-structure cases
- or calibration of when to trust persistence versus learned correction

## 2026-07-03: Stratified Descriptor Audit Added

To stay aligned with the current July 5 analysis sprint, the next step has been kept on the data-analysis side rather than moving into model design.

A new script has been added:

- `scripts/analyze_descriptor_signal_stratified.py`

### Purpose

The first descriptor-signal probe was useful, but still pooled the full SRD together.

That is not enough if the project is really about regime-dependent forecasting behavior.

The stratified audit is designed to answer:

1. which forecast-origin descriptors matter by tier?
2. which descriptors matter by horizon?
3. which features are genuinely necessary, rather than only correlated with the others?

### What the new script does

For each binary task:

- `resunet_beats_locf`
- `target_wins_vs_both_easy`
- `both_hard_vs_both_easy`

the script evaluates:

- overall
- by tier
- by horizon

using:

- logistic-regression importance
- leave-one-feature-out ablation importance

### Why this matters

This is a better fit for the current sprint than immediately implementing a new conditioned model, because it directly strengthens the data/regime analysis.

In particular, it helps answer whether:

- the same descriptors matter in all tiers;
- immediate and slightly longer short horizons are governed by the same signals;
- and which forecast-origin variables survive ablation as indispensable descriptors.

### Intended research value

If the descriptor story remains stable under:

- stratification by tier,
- stratification by horizon,
- and ablation,

then the regime analysis becomes much more robust.

At that point, we will be in a much stronger position to decide whether a later regime-conditioned forecasting method should use:

- one compact descriptor set for all cases,
- or different conditioning logic in different regions of the regime space.

## 2026-07-03: Stratified Descriptor Findings

The stratified descriptor audit now provides a much more rigorous view of which forecast-origin descriptors are stable and which are regime-specific.

## Important caution before interpretation

Some strata are:

- small,
- class-imbalanced,
- or skipped entirely because one class is nearly absent.

So the strongest conclusions should be drawn from:

- overall strata,
- the tier-wise `resunet_beats_locf` task,
- and the horizon-wise `target_wins_vs_both_easy` task

rather than from the sparsest `both_hard` splits.

### 1. The descriptor story is real, but not uniform across all strata

Overall summaries remain strong:

- `resunet_beats_locf`
  - ROC-AUC: `0.756`
  - top feature: `recent_relative_growth`

- `target_wins_vs_both_easy`
  - ROC-AUC: `0.884`
  - top feature: `treated_at_input`
  - top ablation feature: `input_volume_vox`

- `both_hard_vs_both_easy`
  - ROC-AUC: `0.776`
  - top feature: `input_connected_component_count`
  - top ablation feature: `input_connected_component_count`

Interpretation:

- the pooled results were not misleading in a trivial way;
- the three core descriptor families remain visible:
  - activity / change,
  - treatment/stabilization,
  - structural complexity.

### 2. Within tier `A`, treatment dominates the rare learned-vs-persistence differences

For `resunet_beats_locf` in tier `A`:

- top feature: `treated_at_input`
- top ablation feature: `treated_at_input`
- ROC-AUC: `0.878`

Interpretation:

- once we condition on being inside the persistence regime, the remaining distinction is strongly tied to treatment state;
- this is consistent with tier `A` functioning as a low-activity stabilization regime rather than a generic low-difficulty group.

### 3. Within tiers `B` and `C`, time-to-target becomes much more important

For `resunet_beats_locf`:

- tier `B`
  - top feature: `delta_days`
  - top ablation feature: `delta_days`
  - ROC-AUC: `0.881`

- tier `C`
  - top feature: `delta_days`
  - top ablation feature: `delta_days`
  - ROC-AUC: `0.957`

Interpretation:

- after moving into the non-stable regimes, the immediate temporal gap to the target becomes a major differentiator;
- this is a stronger and more precise conclusion than the earlier pooled horizon story;
- it suggests that in active or aggressive cases, "how far ahead the next step is" matters more than in the stability regime.

### 4. The drivers of learned advantage are different across tiers

Tier-wise logistic coefficients for `resunet_beats_locf` indicate:

- tier `A`
  - treatment and history-related features matter most

- tier `B`
  - `delta_days`, `input_volume_vox`, and `input_elongation_ratio` are strongest

- tier `C`
  - `delta_days`, `treated_at_input`, and `recent_relative_growth` are strongest

Interpretation:

- there is no single universal descriptor ordering that explains all tiers equally well;
- the descriptor set seems stable, but the **relative importance** of descriptors changes by regime;
- this is a strong argument against treating all tumor cases as one uniform forecasting problem.

### 5. Horizon-specific ablations show different short-horizon mechanisms

For `target_wins_vs_both_easy`:

#### Horizon 1

- strongest ablation feature: `delta_days`
- ROC-AUC drop from removing it: `0.123`

Interpretation:

- the immediate next-step distinction between learned-win and both-easy cases is driven most strongly by how far away the next scan is;
- this is an important refinement of the earlier "short horizon" story.

#### Horizon 2

- strongest ablation feature: `treated_at_input`
- ROC-AUC drop: `0.013`

Interpretation:

- horizon `2` is the most easily separable overall (`ROC-AUC ~ 0.986`);
- the problem may already be so separable in this stratum that no single feature removal causes a very large drop;
- treatment still appears as the most fragile feature here, but the margin is much smaller.

#### Horizon 3

- strongest ablation feature: `treated_at_input`
- ROC-AUC drop: `0.104`

Interpretation:

- at the farthest tested short horizon, treatment state again becomes the most decisive separator between learned-win and both-easy cases;
- this suggests that treatment may increasingly dominate regime separation once the forecast extends beyond the immediate step.

### 6. A refined descriptor picture is emerging

The stratified analysis suggests the descriptor space should not be summarized too crudely.

The most stable and reusable descriptors now appear to be:

- `treated_at_input`
- `delta_days`
- `recent_relative_growth`
- `input_volume_vox`
- `input_connected_component_count`

But their role changes by setting:

- `treated_at_input`
  - strongest in tier `A`
  - strongest at horizons `2` and `3` for `target_wins_vs_both_easy`

- `delta_days`
  - strongest in tiers `B` and `C`
  - strongest at horizon `1`

- `recent_relative_growth`
  - strongest pooled activity descriptor
  - especially relevant in aggressive settings

- `input_volume_vox`
  - strong global separator between learned-win and easy cases
  - especially important in tier `B`

- `input_connected_component_count`
  - most reliable marker of structural hardness overall

### 7. What this changes in our understanding

The earlier pooled descriptor story can now be refined as follows:

1. There is a stable core descriptor set.
2. The same descriptors do not have the same importance in every tier or horizon.
3. The forecasting problem is therefore not only regime-dependent, but **descriptor-hierarchy dependent** across regimes.

That is a stronger and more precise statement than saying merely that different tiers behave differently.

### Immediate implication for the data-analysis sprint

The current sprint should continue emphasizing:

- descriptor stability,
- descriptor hierarchy by regime,
- and which forecast-origin variables survive ablation.

The strongest current data-analysis contribution is moving toward:

- a regime characterization framework based on forecast-origin descriptors,

not merely a tiered synthetic benchmark.

## 2026-07-03: Descriptor Redundancy Audit Added

To refine the descriptor set further, a redundancy/dependency audit has now been added:

- `scripts/analyze_descriptor_redundancy.py`

### Purpose

The project now has a small set of promising forecast-origin descriptors, but before freezing them as the core regime characterization set, we need to know:

1. which descriptors are genuinely distinct,
2. which are mostly proxies for each other,
3. and whether redundancy structure changes by tier or horizon.

### What the audit computes

Using the forecast-origin descriptor set, the script exports:

- feature summary table
- Pearson correlation matrix
- Spearman correlation matrix
- ranked pairwise correlation tables
- tier-level correlation summary
- horizon-level correlation summary
- high-correlation pair lists by tier and horizon

### Why this matters

A cleaner descriptor set will help the project in two ways:

1. it makes the regime-analysis section more rigorous by showing that the chosen descriptors are not arbitrary duplicates;
2. it reduces future ambiguity if we later encode the regime state into a forecasting method.

### Current role in the sprint

This remains fully aligned with the July 5 analysis sprint.

It is a data-focused refinement step meant to:

- tighten the descriptor story,
- identify the minimal useful descriptor core,
- and prevent the analysis from turning into a diffuse feature collection.

## 2026-07-03: Descriptor Redundancy Findings

The redundancy audit now clarifies which forecast-origin descriptors are overlapping strongly and which remain meaningfully distinct.

### 1. One temporal pair is clearly redundant

Across both Pearson and Spearman summaries:

- `n_sessions` and `followup_days`
  - Pearson: `0.894`
  - Spearman: `0.892`

This is the only globally high-correlation pair that clearly exceeds the current redundancy threshold in both measures.

Interpretation:

- these two variables are largely encoding the same history-length signal;
- we do not need both in the final core descriptor set.

Preferred choice:

- keep `followup_days` if we want a continuous historical-span variable;
- or keep `n_sessions` if we want a simpler discrete count variable.

At the current stage, `followup_days` is likely the more interpretable continuous descriptor, while `n_sessions` can be treated as secondary or optional.

### 2. Volume, recent growth, and component count are related, but not redundant

Important correlations:

- `input_volume_vox` vs `recent_relative_growth`
  - Pearson: `0.624`
  - Spearman: `0.790`

- `input_volume_vox` vs `input_connected_component_count`
  - Pearson: `0.532`
  - Spearman: `0.705`

- `recent_relative_growth` vs `input_connected_component_count`
  - Pearson: `0.416`
  - Spearman: `0.511`

Interpretation:

- larger tumors tend to grow more and also tend to have more components;
- but these relationships are not so extreme that the descriptors collapse into one variable overall;
- this supports keeping:
  - `input_volume_vox`,
  - `recent_relative_growth`,
  - `input_connected_component_count`
  as distinct descriptors for now.

This is especially important because the earlier feature-importance analyses suggested that these three variables play different roles:

- activity,
- burden,
- structural difficulty.

The redundancy audit supports that separation rather than undermining it.

### 3. Treatment is related to activity and burden, but still distinct

Notable correlations:

- `recent_relative_growth` vs `treated_at_input`
  - Pearson: `-0.553`
  - Spearman: `-0.480`

- `input_volume_vox` vs `treated_at_input`
  - Pearson: `-0.402`
  - Spearman: `-0.538`

Interpretation:

- treated-at-input is associated with smaller and less active tumors;
- but the correlation is moderate rather than overwhelming;
- this means `treated_at_input` is not just a noisy proxy for size or recent growth.

That is consistent with the earlier stratified and ablation analyses, where treatment repeatedly acted as a stabilizing descriptor in its own right.

### 4. `delta_days` is not redundant with the broader temporal-history descriptors

Important comparison:

- `delta_days` vs `mean_interval_days`
  - Pearson: `0.266`
  - Spearman: `0.234`

Interpretation:

- the immediate forecast gap (`delta_days`) is not simply the same as the patient’s average sampling interval;
- this is useful, because `delta_days` showed strong regime- and horizon-specific importance in the stratified analyses;
- we should therefore keep `delta_days` as a core descriptor and not replace it with broader temporal-history variables.

### 5. Morphology redundancy is regime-dependent

Overall, morphology descriptors are not strongly redundant.

But tier-level summaries show something important:

- tier `A` top pair:
  - `input_compactness_proxy` vs `input_connected_component_count`
  - correlation: `-0.949`

Interpretation:

- in tier `A`, compactness and component count are almost mirror descriptors of the same simple structural phenomenon;
- in tiers `B` and `C`, the strongest pair shifts away from this and the overall morphology space appears more mixed.

This means:

- morphology redundancy is not uniform across regimes;
- in the persistence regime, we may not need both compactness and component count;
- in the more heterogeneous regimes, keeping both may still be defensible.

### 6. Horizon-level redundancy is relatively stable

Across horizons:

- the dominant redundant pair remains `n_sessions` vs `followup_days`
- mean absolute correlation stays fairly stable (`~0.268-0.273`)

Interpretation:

- the descriptor dependency structure does not change dramatically by horizon at the broad level;
- the major redundancy problem is historical-span duplication, not horizon-specific collapse.

### 7. Refined core descriptor set

Based on:

- pooled importance,
- stratified importance,
- ablation,
- regime-map structure,
- and redundancy,

the current best **core forecast-origin descriptor set** is:

1. `treated_at_input`
2. `delta_days`
3. `recent_relative_growth`
4. `input_volume_vox`
5. `input_connected_component_count`

### 8. Secondary descriptors

The following are still useful, but currently look more secondary or context-dependent:

- `input_compactness_proxy`
- `input_elongation_ratio`
- `followup_days` or `n_sessions` (choose one, not both)
- `mean_interval_days`

### Main takeaway

The descriptor story has now been tightened significantly.

We do **not** have one giant redundant feature set.
We have:

- one clearly redundant temporal-history pair,
- one regime-dependent morphology overlap,
- and a small core set of descriptors that remain meaningfully distinct.

This is a strong result for the analysis sprint because it means the regime characterization framework can now be based on a compact, defensible descriptor set rather than on an overly broad list of candidate variables.

## July 12, 2026 - Growth-Aware Evaluation Stress Test

### Context

After resuming the project, we shifted from only asking whether learned models beat LOCF to asking a more diagnostic question:

> When and where do learned models actually help short-horizon tumor forecasting?

The new growth-aware ranking analyses suggested that the forecasting problem separates into at least three pieces:

- persistence-dominant cases, where LOCF remains hard to beat;
- growth-front localization, where distance to the current tumor boundary is already a strong prior;
- fine spatial ranking of new growth, where learned models appear to add information beyond the distance prior.

### Current Evidence

The strongest current pattern is that learned models help most when there is meaningful future new growth. In low-growth cases, LOCF remains competitive and sometimes preferable. In high-growth cases, learned models show much stronger gains over LOCF.

The ranking analyses also showed that a simple distance-to-input-mask reference is a surprisingly strong broad prior, especially for coarse recall, but learned models generally improve average precision and growth-volume recall. Static hybrid fusion between distance and learned scores produced only small gains, while oracle/adaptive selection showed limited additional headroom.

### New Stress-Test Direction

To avoid over-interpreting the clean summaries, we started an explicit exception audit. The goal is to identify cases that challenge the current interpretation:

- high-growth cases where learned models still rank new-growth regions worse than distance;
- high-growth cases where learned models still lose to LOCF in Dice;
- low-growth cases where learned models unexpectedly improve over distance or LOCF;
- cases where ranking is good but Dice is bad;
- cases where Dice is good but ranking is bad.

This is important because a defensible regime-aware analysis should not only summarize the dominant trend. It should also know where the trend breaks.

### Next Step

Run the exception audit on the SRD growth-aware evaluation outputs and inspect whether the failure modes are rare, structured, or random. If the exceptions are structured, they may point directly toward the next methodological idea: a growth-front-aware residual/ranking model rather than a broad post-hoc gating method.

## July 12, 2026 - Ranking-Dice Tradeoff Follow-Up

The exception audit showed that the dominant failure mode was not simply learned models failing to localize growth. Instead, many cases had improved new-growth ranking but worse full-mask Dice compared with LOCF.

This suggests a deeper split in the short-horizon forecasting task:

- preserving the existing tumor mask, where LOCF is naturally strong;
- ranking likely new-growth voxels, where learned models can add signal;
- converting a probabilistic growth field into a hard segmentation mask, where learned models may lose Dice despite useful spatial information.

A follow-up tradeoff analysis was added to separate:

- zero-growth cases;
- small nonzero absolute growth;
- medium nonzero absolute growth;
- large nonzero absolute growth.

It also maps each sample into ranking/Dice quadrants:

- good ranking and good Dice;
- good ranking but bad Dice;
- bad ranking but good Dice;
- bad ranking and bad Dice;
- neutral cases.

This should help determine whether the current evidence supports a persistence-preserving residual growth model rather than a plain full-mask predictor.

## July 12, 2026 - Persistence Plus Ranked-Growth Budget Test

The ranking-Dice tradeoff analysis produced a strong split:

- large absolute growth: learned models improve both ranking and Dice;
- small absolute growth: learned models, especially U-Net, can improve growth ranking while hurting full-mask Dice;
- zero-growth cases need to be separated because ranking metrics are undefined.

This supports the idea that the learned model may contain useful growth-location information even when its full hard segmentation is not preferable to LOCF.

The next operational test is therefore:

> Keep the current input mask as the persistent core and add only a limited budget of top-ranked candidate growth voxels.

This tests whether a persistence-preserving residual-growth formulation can translate the ranking signal into better masks without letting the learned model overwrite the stable tumor core.

Budget policies to test:

- oracle true future growth volume;
- previous observed growth volume;
- one percent of candidate voxels;
- five percent of candidate voxels.

Score sources to compare:

- distance-to-input-mask;
- U-Net probability score;
- ResUNet probability score;
- distance/model hybrid score with high learned-model weight.

## July 12, 2026 - Deployable Growth-Budget Sweep Added

The first persistence-plus-growth-budget test showed that the ranked-growth residual idea is promising, but only when the growth budget is sensible.

Key interpretation:

- oracle true-growth budget gives a clear upper bound and confirms that the spatial ranking signal is useful;
- previous-growth budget is deployable and mildly positive;
- fixed candidate-percentage budgets fail because they add far too many voxels in small/medium-growth cases.

The next refinement is a deployable budget-policy sweep:

- scale previous observed growth by factors such as `0.25`, `0.5`, `0.75`, `1.0`, `1.25`, and `1.5`;
- cap scaled budgets by fractions of current input tumor volume;
- optionally force the budget to zero when recent previous growth is zero or near-zero.

This should test whether a simple history-aware budget estimate can preserve the positive ranking signal without over-expanding the tumor mask.

## July 12, 2026 - Anti-Overfitting Guardrail For Budget Policies

A concern was raised that the current analysis could become over-engineered around the synthetic dataset.

This is a valid risk. The immediate guardrail is:

- do not select budget policies from the same test table used for reporting;
- run the candidate budget sweep on a validation split;
- select a deployable policy on validation only;
- evaluate the selected policy on held-out test;
- keep oracle and future-growth-bin analyses clearly labeled as diagnostics, not deployable methods.

A validation-to-test selector was added to enforce this workflow.

## July 12, 2026 - Validation-Selected Budget Policy Held-Out Test Result

The validation-to-test budget selection workflow was run for the persistence-preserving ranked-growth method.

Selected validation policy:

- score source: `hybrid_distance_resunet_image_mask_a0.75`
- budget policy: `prev_growth_x1p5`
- policy family: scaled previous growth
- validation mean Dice: `0.821196`
- validation LOCF mean Dice: `0.750335`
- validation mean gap vs LOCF: `+0.070861`
- validation win rate vs LOCF: `0.657895`

Held-out test result for the selected policy:

- test mean Dice: `0.759344`
- test LOCF mean Dice: `0.726978`
- test mean gap vs LOCF: `+0.032367`
- bootstrap CI for mean gap: `[+0.016370, +0.048781]`
- test win rate vs LOCF: `0.561404`

Interpretation:

- the validation-selected policy does transfer positively to held-out test;
- the test gain is much smaller than the validation gain, which is expected and should be treated as a correction against over-optimism;
- the confidence interval remains above zero in this bootstrap check, which makes this stronger than a post-hoc test-set sweep;
- the selected policy still appears to over-budget relative to true growth on test, so growth-budget estimation remains the main unresolved bottleneck.

This result supports the current research direction, but the claim should remain conservative: a simple persistence-preserving, ranked-growth residual rule can improve LOCF on SRD when selected on validation, but broader real-data validation is required before treating it as a general forecasting method.

## July 12, 2026 - Selected Budget Policy Robustness Audit Added

After the validation-selected persistence-growth budget policy transferred positively to held-out test, the next scrutiny step was identified as a breakdown audit rather than another model run.

A script was added to audit the selected policy by:

- validation versus test distribution shift;
- test performance by tier;
- test performance by horizon;
- test performance by absolute and relative growth bins;
- tier-by-horizon and tier-by-growth interactions;
- missing samples for the selected hybrid score source.

This is intended to test whether the selected policy is broadly useful or only supported by a few favorable subgroups, and to explain why the hybrid policy test count was smaller than the full test sample count.

## July 12, 2026 - Selected Budget Policy Audit Interpretation

The selected validation policy `hybrid_distance_resunet_image_mask_a0.75 + prev_growth_x1p5` was audited on held-out test.

Overall held-out test result on available hybrid-score rows:

- count: `114`
- mean Dice: `0.759344`
- LOCF mean Dice: `0.726978`
- mean gap vs LOCF: `+0.032367`
- median gap vs LOCF: `+0.006226`
- win rate vs LOCF: `0.561404`
- bootstrap CI for mean gap: `[+0.016370, +0.048781]`

The audit showed strong regime dependence:

- Tier A: mean gap `-0.014891`; mostly small-growth cases where LOCF is already strong.
- Tier B: mean gap `+0.046742`.
- Tier C: mean gap `+0.086731`.
- Large nonzero growth: mean gap `+0.133852`, win rate `0.937500`.
- Medium nonzero growth: mean gap `+0.016296`, win rate `0.606061`.
- Small nonzero growth: mean gap `-0.023611`, win rate `0.437500`.
- Zero growth: mean gap `-0.022101`, win rate `0.000000` on non-empty cases included in the hybrid score table.

Nine test samples were missing from the selected hybrid-policy rows. All were empty-input/empty-target zero-growth cases with LOCF Dice `1.0`. These were absent because the hybrid score requires a distance/model ranking outside an input mask, and empty input masks do not provide a valid distance-based ranking. For fair headline reporting, these samples should be treated explicitly rather than silently excluded; the natural deployable behavior is to output an empty mask when the input mask is empty.

Main interpretation:

- the persistence-preserving ranked-growth rule is promising for B/C and large-growth regimes;
- it is not appropriate as a blanket replacement for LOCF;
- the next methodological step should be a growth-gating or budget-selection rule that chooses LOCF/zero-budget for small or zero-growth-like cases and activates ranked-growth addition only when the case appears growth-active.

## July 12, 2026 - Validation-Selected Growth-Activity Gate Added

The selected ranked-growth residual policy improved held-out test overall but harmed Tier A, small-growth, and zero-growth cases. This suggests that the next method should not always add growth.

A validation-selected gating script was added to test a simple deployable decision rule:

- if the case appears growth-active, use the ranked-growth residual policy;
- otherwise, use LOCF;
- if the input mask is empty or the selected hybrid score is unavailable, fall back to LOCF.

The gate uses prediction-time variables only, such as:

- selected policy growth budget;
- budget-to-input-volume ratio;
- input tumor volume;
- delta days.

Future growth bins remain diagnostic only and are used only after evaluation to understand where the gate helps or fails. The goal is to test whether a simple growth-activity gate can preserve the large-growth benefits while removing the small/zero-growth harm.

## July 12, 2026 - Validation-Selected Growth Gate Result

The validation-selected growth-activity gate was run for the ranked-growth residual policy.

Selected gate:

- gate: `growth_budget_vox >= 2085`
- base policy: `hybrid_distance_resunet_image_mask_a0.75 + prev_growth_x1p5`
- validation mean gap vs LOCF: `+0.072384`
- validation gate active rate: `0.500000`

Held-out test result:

- count: `123`
- mean Dice: `0.777026`
- LOCF mean Dice: `0.746955`
- mean gap vs LOCF: `+0.030071`
- bootstrap CI: `[+0.017110, +0.043428]`
- gate active rate: `0.390244`

Important subgroup behavior:

- Tier A: mean gap `0.000000`, gate active rate `0.000000`; the gate fully protected the LOCF-dominant low-growth tier.
- Tier B: mean gap `+0.021451`, gate active rate `0.461538`.
- Tier C: mean gap `+0.079503`, gate active rate `0.833333`.
- Large nonzero growth: mean gap `+0.118728`, gate active rate `0.843750`.
- Medium nonzero growth: mean gap `+0.001871`, gate active rate `0.393939`.
- Small nonzero growth: mean gap `0.000000`, gate active rate `0.000000`; the gate removed the small-growth harm.
- Zero growth: mean gap `-0.006243`, gate active rate `0.307692`; this is the remaining failure mode.

Interpretation:

The gated policy supports the central decomposition: use LOCF for stable/small-growth cases and activate ranked-growth residuals for growth-active cases. It improves held-out test while protecting Tier A and small-growth cases. The remaining bottleneck is false activation in zero-growth cases, especially where previous observed growth is high but future growth stops.

This points to the next research question: can we identify growth cessation or stability from prediction-time descriptors rather than only extrapolating previous growth magnitude?

## July 12, 2026 - Gate False-Activation Audit Added

The gated ranked-growth method still harms some zero-growth cases because it can activate when previous growth was high but future growth stops.

A false-activation audit was added to:

- label gate outcomes as active true growth, false activation on zero growth, protected zero growth, and inactive missed growth;
- summarize these classes overall and by tier;
- compare prediction-time feature profiles across these classes;
- test a validation-selected suppression guard that can turn off the growth gate under simple feature-threshold conditions.

The suppression guard uses only prediction-time features:

- input volume;
- delta days;
- selected growth budget;
- selected budget-to-input-volume ratio;
- selected budget-to-input-volume percentage.

This tests whether the zero-growth false-activation problem can be reduced with a simple deployable rule, or whether richer growth-cessation modeling is required.

## July 12, 2026 - False-Activation Audit Result

The false-activation audit was run on the validation-selected gated ranked-growth method.

Gate case classes on held-out test:

- active true growth: `40` cases, mean gap `+0.096526`.
- false activation on zero growth: `8` cases, mean gap `-0.020289`.
- inactive missed growth: `57` cases, mean gap `0.000000` because LOCF was used.
- protected zero growth: `18` cases, mean gap `0.000000` because LOCF was used.

Feature profiles showed that false zero-growth activations resemble true active-growth cases under simple prediction-time scalar features:

- active true growth median budget-to-input ratio: `0.456546`.
- false activation median budget-to-input ratio: `0.458212`.
- active true growth median growth budget: `6167` voxels.
- false activation median growth budget: `3261` voxels.
- false activations had longer median delta days (`149.5`) than active true growth (`92.0`), but this did not provide a useful suppression rule.

A validation-selected suppression guard was selected:

- guard: `suppress_if_delta_days <= 46`.
- validation mean gap: `+0.073891` vs `+0.072384` without the guard.
- test mean gap: `+0.030570` vs `+0.030071` without the guard.
- test bootstrap CI: `[+0.018071, +0.043430]`.

However, the selected guard did not suppress the zero-growth false activations. The false-activation class still had guarded gate active rate `1.000000` and mean gap `-0.020289`. The small test improvement came from suppressing a few short-interval active-growth cases, not from solving the growth-cessation problem.

Interpretation:

- the first-stage growth gate is useful and robust enough to keep;
- simple scalar suppression rules are not sufficient to identify future growth cessation;
- the remaining problem likely requires richer longitudinal state descriptors, treatment/timing context, image intensity cues, or a learned probability of future growth rather than thresholding previous-growth-derived budget alone.

## July 12, 2026 - Growth Continuation Analysis Added

The false-activation audit showed that simple scalar suppression guards cannot reliably identify cases where previous growth stops. The next analysis therefore shifts from gate tuning to longitudinal growth-state characterization.

A growth-continuation analysis script was added to label each forecast sample as:

- continued growth: previous growth active and future growth active;
- stopped growth: previous growth active but future growth inactive;
- newly active: previous growth inactive but future growth active;
- stable: previous growth inactive and future growth inactive.

The script compares stopped versus continued growth using prediction-time descriptors:

- previous growth volume and relative previous growth;
- previous interval days and forecast horizon gap;
- input tumor volume;
- treatment at input and previous session;
- input morphology descriptors such as compactness, elongation, and connected components.

This is intended to determine whether growth cessation has an observable signature before building a learned continuation classifier.

## July 12, 2026 - Growth Continuation Analysis Result

The growth-continuation analysis was run on SRD test with `min_growth_vox = 0`.

Overall states:

- continued growth: `97 / 123` (`78.86%`)
- stopped growth: `14 / 123` (`11.38%`)
- stable: `12 / 123` (`9.76%`)

By tier:

- Tier A: `48 / 48` continued growth under a strict `>0 voxel` definition.
- Tier B: `20 / 39` continued, `13 / 39` stopped, `6 / 39` stable.
- Tier C: `29 / 36` continued, `1 / 36` stopped, `6 / 36` stable.

By horizon:

- stopped growth increases with horizon: H1 `1 / 41`, H2 `5 / 41`, H3 `8 / 41`.
- Tier B is the clearest cessation regime: stopped growth rises from `1 / 13` at H1 to `7 / 13` at H3.

Stopped-versus-continued feature contrast:

- stopped cases have longer future delta days than continued cases: median `146.5` vs `109.0` days, Cohen's d `0.646`.
- stopped cases had no treatment at input, while continued cases had input treatment in about `18.6%` of samples; this is likely SRD-regime-specific rather than a biological conclusion.
- stopped cases had higher median previous relative new growth: `0.3338` vs `0.2158`.
- morphology differences were modest.

Interpretation:

The continuation analysis suggests that growth cessation is concentrated in Tier B and at longer horizons. Simple morphology/history features offer some weak signal but not a clean separation. The strict `>0 voxel` activity definition likely over-labels tiny Tier A changes as continued growth, so the next robustness step is to repeat continuation analysis with nonzero minimum growth thresholds before building a learned continuation classifier.

## July 12, 2026 - Growth Continuation Threshold Sensitivity Result

Growth-continuation analysis was repeated for minimum growth thresholds `0`, `50`, `100`, `250`, and `500` voxels.

Main robustness finding:

- Tier B and Tier C continuation structures are stable across thresholds.
- Tier A is highly threshold-sensitive, meaning its apparent `continued_growth` at `>0` voxels is largely tiny mask drift or very small new growth.

Tier-level behavior:

- At threshold `0`, Tier A is `48 / 48` continued growth.
- At threshold `100`, Tier A becomes mixed: continued `9 / 48`, newly active `14 / 48`, stable `19 / 48`, stopped `6 / 48`.
- At threshold `250`, Tier A is mostly stable: stable `41 / 48`, newly active `4 / 48`, stopped `3 / 48`.
- At threshold `500`, Tier A is mostly stable: stable `43 / 48`, newly active `2 / 48`, stopped `3 / 48`.
- Tier B remains mixed with substantial stopped growth: at threshold `250`, continued `20 / 39`, stopped `13 / 39`, stable `6 / 39`; at threshold `500`, continued `20 / 39`, stopped `10 / 39`, stable `9 / 39`.
- Tier C remains growth-continuing: at both `250` and `500`, continued `29 / 36`, stable `6 / 36`, stopped `1 / 36`.

Overall state counts shift strongly with threshold:

- threshold `0`: continued `97`, stable `12`, stopped `14`.
- threshold `250`: continued `49`, stable `53`, stopped `17`, newly active `4`.
- threshold `500`: continued `49`, stable `58`, stopped `14`, newly active `2`.

Stopped-versus-continued contrast strengthens at meaningful thresholds:

- at threshold `250`, stopped cases have smaller input volume than continued cases, Cohen's d about `1.05` in magnitude;
- at threshold `250`, stopped cases have smaller previous new growth than continued cases, Cohen's d about `0.78` in magnitude;
- at threshold `250`, stopped cases have longer future delta days, Cohen's d about `0.66`;
- at threshold `500`, input treatment and treatment started at input show large differences, but this should be treated cautiously as SRD-specific.

Interpretation:

A meaningful-growth threshold is necessary. Thresholds around `250` to `500` voxels reveal the intended regime structure much better than a strict `>0 voxel` definition. Tier A behaves as mostly stable/small-change, Tier B behaves as mixed continuation/cessation, and Tier C behaves as mostly continuing growth. This supports using a nonzero activity threshold before training or evaluating a growth-continuation classifier.

## July 12, 2026 - Growth-Activity Classifier Added

After continuation-threshold sensitivity showed that meaningful growth should use a nonzero threshold, a simple validation-selected growth-activity classifier was added.

Purpose:

- predict whether future new growth exceeds a meaningful threshold, initially `250` voxels;
- use only prediction-time descriptors;
- train on train split, select model and probability threshold on validation, and evaluate once on held-out test.

Candidate models:

- class-balanced logistic regression;
- shallow class-balanced decision trees;
- small class-balanced random forest.

Default features exclude synthetic tier and horizon unless explicitly requested. This keeps the first test closer to deployable descriptors rather than synthetic-regime labels.

This is not yet plugged into the mask forecast. The first question is whether future growth activity is predictable at all from available descriptors.

## July 12, 2026 - Growth-Activity Classifier Result and Transition Failure

The growth-activity classifier was run with a meaningful future-growth threshold of `250` voxels.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_growth_activity_classifier_thr250_v1`

Validation-selected model:

- selected model: small class-balanced random forest;
- validation balanced accuracy: `0.913952`;
- validation F1: `0.924242`;
- validation ROC AUC: `0.888332`.

Held-out test performance:

- accuracy: `0.829268`;
- balanced accuracy: `0.840836`;
- precision: `0.742424`;
- recall: `0.924528`;
- F1: `0.823529`;
- ROC AUC: `0.944879`;
- confusion matrix: TN `53`, FP `17`, FN `4`, TP `49`.

Important subgroup result:

- continued growth: `49 / 49` predicted active correctly;
- stable: `53 / 53` predicted inactive correctly;
- stopped growth: `17 / 17` predicted active incorrectly;
- newly active: `4 / 4` predicted inactive incorrectly.

Interpretation:

The aggregate classifier result is strong but potentially misleading. It mostly learns persistence of the current growth state: previous active growth tends to remain active, and previous inactive growth tends to remain inactive. It does not yet solve the actual transition problem. In particular, it fails exactly where the gated ranked-growth policy needs help: stopped-growth cases, where previous growth exists but future growth is absent or below the meaningful threshold.

Research implication:

This is a useful negative/diagnostic result. The next necessary test is a focused continuation classifier restricted to previous-active cases. That experiment asks whether stopped growth can be separated from continued growth using prediction-time descriptors. If that focused classifier fails, then scalar descriptors are likely insufficient and we need richer image-derived or model-derived features for cessation detection.

## July 12, 2026 - Previous-Active / Previous-Inactive Transition Classifier Results

The growth-activity classifier was rerun with two focused subsets at `min_growth_vox = 250`:

1. `previous_active`: distinguish continued growth from stopped growth;
2. `previous_inactive`: distinguish stable cases from newly active growth.

Output directories:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_growth_activity_classifier_thr250_previous_active_v1`
- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_growth_activity_classifier_thr250_previous_inactive_v1`

### Previous-active result: continuation versus cessation

Selected model:

- small class-balanced random forest;
- selected threshold: `0.433919`.

Held-out test metrics:

- count: `66`;
- positive class = continued growth;
- accuracy: `0.787879`;
- balanced accuracy: `0.588235`;
- precision: `0.777778`;
- recall: `1.000000`;
- F1: `0.875000`;
- ROC AUC: `0.786315`;
- confusion matrix: TN `3`, FP `14`, FN `0`, TP `49`;
- specificity for stopped growth: `0.176471`.

By continuation state:

- continued growth: `49 / 49` predicted active correctly;
- stopped growth: only `3 / 17` predicted inactive correctly, with `14 / 17` false activations.

Interpretation:

This confirms that the classifier mostly learns continuation, not cessation. Even after restricting to previous-active cases, the model strongly favors predicting continued growth. This directly explains why the gated ranked-growth policy still falsely activates in some zero-growth or stopped-growth cases.

### Previous-inactive result: stable versus newly active

Selected model:

- class-balanced logistic regression;
- selected threshold: `0.159688`.

Held-out test metrics:

- count: `57`;
- positive class = newly active growth;
- accuracy: `0.719298`;
- balanced accuracy: `0.733491`;
- precision: `0.166667`;
- recall: `0.750000`;
- F1: `0.272727`;
- ROC AUC: `0.872642`;
- confusion matrix: TN `38`, FP `15`, FN `1`, TP `3`;
- specificity for stable cases: `0.716981`.

By continuation state:

- newly active: `3 / 4` detected;
- stable: `38 / 53` protected, but `15 / 53` falsely activated.

Interpretation:

New activation appears more detectable than cessation, but the positive class is very rare. The classifier can recover most newly active cases, but only with substantial false positives. This is useful diagnostically, but not yet deployable as a clean gate.

Research implication:

The main unresolved problem is not broad future-growth activity. It is transition detection:

- stopping after recent growth is difficult with current scalar descriptors;
- new activation from apparent stability is detectable but noisy;
- richer image-derived or uncertainty-derived features are likely needed before using a learned transition gate inside the forecasting policy.

The next analysis should therefore avoid over-tuning scalar gates and instead inspect what information might identify stopped-growth cases: spatial residual shape, model uncertainty, intensity context, treatment timing/history, or learned probability calibration.

## July 12, 2026 - Pivot From False-Positive Suppression to Delayed-Growth Hits

After reviewing the stopped-growth classifier results, we reconsidered whether false-positive growth predictions should be treated as purely bad. In a tumor forecasting setting, false negatives can be more clinically concerning than false positives, especially if the model is interpreted as a growth-risk or attention-guidance field rather than a final binary segmentation.

Important reframing:

- immediate false positives may represent delayed, sub-threshold, or plausible future growth regions;
- a prediction that misses the next scan may still be useful if the same region becomes tumor at a later scan;
- Dice alone can penalize early warnings even when the predicted region is biologically or spatially plausible.

A new delayed-hit analysis script was added:

- `scripts/analyze_delayed_growth_hits.py`

The script evaluates whether top-ranked candidate voxels outside the input tumor mask are:

1. true growth at the immediate target scan;
2. apparent immediate false positives that become tumor in later sessions;
3. eventual growth from the input scan over the remaining follow-up window.

It reports immediate precision, eventual precision, delayed-hit rate among immediate false positives, and eventual precision gain. It includes learned model scores, distance-to-input reference scores, random scores, and optional distance/model hybrid scores.

Research question introduced:

Are apparent short-horizon false positives actually early warnings of future growth, or are they merely over-expansion artifacts?

This is a stronger and more clinically aligned test than simply suppressing all false activations.

## July 12, 2026 - Delayed-Growth Hit Result

The delayed-growth hit analysis was run on SRD test samples using U-Net, ResUNet, distance-to-input-mask, random scores, and distance/model hybrids.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_delayed_growth_hits_v1`

Main finding:

Apparent immediate false-positive growth predictions often overlap with tumor regions that appear in later sessions. This supports reframing some false positives as possible early-warning or risk-field predictions rather than treating them only as segmentation errors.

Overall diagnostic results:

- ResUNet, top immediate-growth-volume budget:
  - immediate precision: `0.496508`;
  - eventual precision: `0.616293`;
  - eventual precision gain: `0.119785`;
  - delayed hit rate among immediate false positives: `0.371297`.
- U-Net, top immediate-growth-volume budget:
  - immediate precision: `0.502414`;
  - eventual precision: `0.604273`;
  - eventual precision gain: `0.101859`;
  - delayed hit rate among immediate false positives: `0.326317`.
- Distance-to-input-mask, top immediate-growth-volume budget:
  - immediate precision: `0.429999`;
  - eventual precision: `0.529137`;
  - eventual precision gain: `0.099139`;
  - delayed hit rate among immediate false positives: `0.213002`.
- Hybrid distance+ResUNet, top immediate-growth-volume budget:
  - immediate precision: `0.540144`;
  - eventual precision: `0.662738`;
  - eventual precision gain: `0.122594`;
  - delayed hit rate among immediate false positives: `0.373552`.

Important nuance:

The delayed-hit effect is not exclusively learned-model behavior. The distance-to-input-mask baseline also gains eventual precision, meaning part of the signal is explained by simple boundary-proximal growth. However, learned and hybrid scores provide stronger delayed-hit behavior under tighter growth-volume-style budgets, especially compared with the distance baseline.

Tier and horizon structure:

- delayed-hit signal is weak in Tier A because true growth volume is small;
- signal is much stronger in Tier B and Tier C;
- horizon 1 has the largest eventual precision gain because there are more later sessions available for delayed validation;
- horizon 3 has less delayed-hit opportunity, so delayed gains are naturally smaller.

Interpretation:

This result supports evaluating tumor forecasting as a probabilistic growth-risk field in addition to hard Dice segmentation. A prediction that is false-positive at the immediate target may still be informative if it marks a region that becomes tumor later. The next step should distinguish ordinary over-expansion from genuine delayed-hit predictions, preferably through per-sample delayed-hit case inspection and a delayed-hit-versus-never-hit stratification.

## July 12, 2026 - Delayed False-Positive Profile Result

The delayed false-positive profile analysis was run to separate immediate false positives into:

1. delayed-hit false positives: selected voxels not tumor at the immediate target scan but tumor later;
2. never-hit false positives: selected voxels not tumor at the immediate target or later observed sessions.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_delayed_fp_profiles_v1`

Important corrective finding:

The delayed-hit story survives, but the voxel-weighted profile shows that boundary proximity explains a large part of the useful signal. The distance-to-input-mask baseline is extremely strong under voxel-weighted growth-volume budgets.

For the `top_immediate_growth_volume` budget:

- distance-to-input-mask:
  - weighted immediate precision: `0.799419`;
  - weighted eventual precision: `0.829165`;
  - weighted eventual precision gain: `0.029745`;
  - weighted delayed-FP fraction among immediate FPs: `0.148296`.
- ResUNet:
  - weighted immediate precision: `0.101388`;
  - weighted eventual precision: `0.142728`;
  - weighted eventual precision gain: `0.041340`;
  - weighted delayed-FP fraction among immediate FPs: `0.046004`.
- U-Net:
  - weighted immediate precision: `0.095726`;
  - weighted eventual precision: `0.137783`;
  - weighted eventual precision gain: `0.042057`;
  - weighted delayed-FP fraction among immediate FPs: `0.046509`.
- Hybrid distance+ResUNet:
  - weighted immediate precision: `0.230393`;
  - weighted eventual precision: `0.379367`;
  - weighted eventual precision gain: `0.148973`;
  - weighted delayed-FP fraction among immediate FPs: `0.193571`.

Distance structure:

Across methods, delayed-hit false positives are much closer to the input tumor boundary than never-hit false positives. For example, for hybrid distance+ResUNet at the `top_immediate_growth_volume` budget, the mean distance of delayed false positives is about `2.43` voxels versus `5.11` voxels for never-hit false positives. For pure model scores, delayed false positives are also closer to the boundary than never-hit false positives.

Interpretation:

This result prevents overclaiming. The delayed-hit phenomenon is real in SRD, but it is not primarily evidence that the neural network is independently discovering distant future tumor. Much of the delayed-hit signal is a boundary-growth phenomenon. Learned/hybrid scores may still be useful for sample-level prioritization and some delayed-risk cases, but the strongest voxel-level prior is simple proximity to the current tumor.

Research implication:

A mature forecasting/risk-field method should explicitly include a growth-front or distance-to-boundary prior. The neural model should be framed as a correction or sharpening layer over that prior, not as a replacement for it. This also strengthens the argument that evaluation should report both sample-level and voxel-weighted summaries, because they answer different questions and can lead to different conclusions.

## July 16, 2026 - SAILOR Distance-Only Delayed-Hit Stress Check

A distance-only delayed-hit analysis was run on SAILOR test samples after model-probability ranking caused a Colab SIGKILL during full-volume inference.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/sailor_real_stress_v1/delayed_hits_distance_only`

Main result:

The distance-to-current-mask growth-front prior shows strong delayed-hit behavior on SAILOR, far above random scoring.

Overall, for the distance-to-input-mask baseline:

- `top_0.01_candidate_fraction`:
  - immediate precision: `0.205737`;
  - eventual precision: `0.566667`;
  - eventual precision gain: `0.360930`;
  - delayed-hit rate among immediate false positives: `0.414153`.
- `top_immediate_growth_volume`:
  - immediate precision: `0.311227`;
  - eventual precision: `0.695690`;
  - eventual precision gain: `0.384463`;
  - delayed-hit rate among immediate false positives: `0.524032`.

For random scoring, eventual precision stayed near `0.027` to `0.028`, confirming that the distance effect is not merely due to high future-growth prevalence in the volume.

Horizon structure:

- top immediate-growth-volume distance prior:
  - H1 eventual precision: `0.780976`, delayed-hit rate among immediate false positives: `0.647444`;
  - H2 eventual precision: `0.722877`, delayed-hit rate: `0.579123`;
  - H3 eventual precision: `0.583216`, delayed-hit rate: `0.345529`.

Interpretation:

This supports the real-data relevance of the growth-front/delayed-risk framing. Boundary-proximal regions that are false positives at the immediate scan often become tumor later in SAILOR. The finding is preliminary because the SAILOR test set has only 12 samples, but it is directionally consistent with the SRD delayed-hit analysis.

Research implication:

A distance-to-boundary growth-front prior should be treated as a central baseline and likely as a component of any modified forecasting method. The next technical bottleneck is obtaining model probability maps on SAILOR without full-volume memory failure, likely through cropping around the current tumor, downsampling, or one-sample inference.

## July 16, 2026 - Cropped SAILOR Model-Ranking Utility Added

The full-volume SAILOR model-ranking cell was killed by Colab (`SIGKILL`), most likely because loading full 3D volumes through the trained ResUNet checkpoints exceeded available memory.

To avoid turning this into a hardware problem, we added a memory-safe cropped evaluation script:

- `scripts/analyze_cropped_model_growth_ranking.py`

Purpose:

- run trained models only inside an input-mask-centered crop;
- compare model, distance-to-input-mask, random, and distance+model hybrid ranking inside the same local candidate region;
- report crop coverage so we know how much true future growth the crop actually contains;
- keep this as a SAILOR stress-test utility rather than modifying the original SRD growth-ranking pipeline.

Interpretation plan:

This should tell us whether the learned SAILOR model probabilities add anything beyond the boundary-distance prior when inference is made feasible. If they do not, that is still useful: it would strengthen the conclusion that the growth-front prior is the dominant real-data signal and that learned models should be treated as correction/sharpening layers rather than standalone growth detectors.

## July 16, 2026 - SAILOR Cropped Image+Mask Hybrid Ranking Check

A memory-safe cropped SAILOR ranking analysis was run for `resunet_image_mask` and compared against the prior cropped `resunet_mask` run and the distance-to-input-mask growth-front prior.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/sailor_real_stress_v1/cropped_model_ranking_resunet_image_mask_m48`

Main overall ranking results on 12 SAILOR test samples:

- distance-to-input-mask:
  - mean AP: `0.279188`;
  - recall at growth volume: `0.303252`;
  - recall@1%: `0.610423`;
  - recall@5%: `0.822788`.
- ResUNet image+mask:
  - mean AP: `0.297440`;
  - recall at growth volume: `0.364521`;
  - recall@1%: `0.567454`;
  - recall@5%: `0.732567`.
- hybrid distance + ResUNet image+mask, alpha `0.75`:
  - mean AP: `0.334626`;
  - recall at growth volume: `0.395149`;
  - recall@1%: `0.591498`;
  - recall@5%: `0.797094`.

Paired sample-level comparison:

- image model beats mask-only model AP in `9/12` samples;
- image hybrid beats distance-only AP in `10/12` samples;
- image hybrid beats mask-only hybrid AP in `9/12` samples;
- image hybrid improves recall-at-growth-volume over distance in `11/12` samples.

Interpretation:

This is the strongest SAILOR evidence so far that image context adds useful ranking signal when combined with a growth-front prior. The distance prior remains extremely competitive for top-candidate recall, especially recall@5%, but the image-informed hybrid improves average precision and recall at the true growth-volume budget across most samples. This supports the current research framing: forecasting should not be treated as pure end-to-end mask prediction alone, but as persistence plus growth-front prior plus learned image/context correction.

Caution:

The SAILOR test set is still small (`n=12` samples), so these results should be treated as a real-data stress check rather than definitive clinical validation.

## July 16, 2026 - SAILOR Cropped Hybrid Robustness Check

A bootstrap, paired sign-permutation, and leave-one-patient-out robustness check was run for the cropped SAILOR image+mask hybrid ranking result.

Bootstrap / paired checks on 12 SAILOR test samples:

- image model vs mask-only model AP:
  - mean gain: `0.063317`;
  - 95% bootstrap CI: `[0.019691, 0.113531]`;
  - positive samples: `9/12`;
  - paired sign-permutation p-value: `0.0159`.
- image hybrid vs distance-only AP:
  - mean gain: `0.055439`;
  - 95% bootstrap CI: `[0.023169, 0.087970]`;
  - positive samples: `10/12`;
  - paired sign-permutation p-value: `0.0081`.
- image hybrid vs mask-only hybrid AP:
  - mean gain: `0.071729`;
  - 95% bootstrap CI: `[0.012185, 0.138970]`;
  - positive samples: `9/12`;
  - paired sign-permutation p-value: `0.0042`.
- image hybrid vs distance-only recall-at-growth-volume:
  - mean gain: `0.091898`;
  - 95% bootstrap CI: `[0.054557, 0.130728]`;
  - positive samples: `11/12`;
  - paired sign-permutation p-value: `0.0011`.

Leave-one-patient-out checks:

Every comparison retained a positive mean gain after excluding any one of the four SAILOR patients. For image hybrid vs distance-only AP, leave-one-patient-out mean gains ranged from `0.045471` to `0.068063`. For image hybrid vs distance-only recall-at-growth-volume, leave-one-patient-out mean gains ranged from `0.079472` to `0.111967`.

Interpretation:

This materially strengthens the real-data evidence. The cropped image+mask hybrid improvement is not driven by a single SAILOR patient, and the paired gains remain positive under bootstrap and leave-one-patient-out perturbations. The result remains preliminary because the test set is small, but within this SAILOR stress check the direction of evidence is consistent: image/context-informed model scores add useful ranking signal on top of a boundary-distance growth-front prior.

## July 16, 2026 - Hybrid Forecast Policy Evaluator Added

Added and pushed:

- `scripts/evaluate_hybrid_forecast_policy.py`

Purpose:

This script moves the project from analysis-only evidence toward a deployable method prototype. For each candidate score source, it selects a growth-budget rule on validation, selects a simple activation gate on validation, and evaluates the selected policy on held-out test.

The intended policy family is:

`future mask = current mask + gated top-ranked growth-front candidates`

Candidate score sources can include:

- `distance_to_input_mask`;
- `resunet_image_mask`;
- `hybrid_distance_resunet_image_mask_a0.75`.

Important design choice:

The script is validation-driven. It does not choose the budget or gate directly on test. This is meant to reduce the risk of overfitting the current SRD test set while still letting us evaluate whether the growth-front + image-context correction idea produces a usable forecast mask.

## July 16, 2026 - SRD Hybrid Forecast Policy Prototype Result

Ran the validation-selected hybrid forecast policy evaluator on SRD budget-sweep validation/test outputs.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_hybrid_forecast_policy_v1`

Selected setup:

- budget policy selected on validation for all score sources: `prev_growth_x1p5`;
- gate selected on validation: `growth_budget_vox >= 2085`;
- candidate score sources tested:
  - `distance_to_input_mask`;
  - `resunet_image_mask`;
  - `hybrid_distance_resunet_image_mask_a0.75`.

Held-out test overall:

- distance policy:
  - mean Dice: `0.770612`;
  - LOCF mean Dice: `0.746955`;
  - mean gap: `+0.023657`;
  - bootstrap CI: `[0.012068, 0.034927]`.
- ResUNet image+mask score policy:
  - mean Dice: `0.776783`;
  - mean gap: `+0.029828`;
  - bootstrap CI: `[0.016403, 0.043160]`.
- hybrid distance + ResUNet image+mask policy:
  - mean Dice: `0.777026`;
  - mean gap: `+0.030071`;
  - bootstrap CI: `[0.016741, 0.043434]`.

Regime behavior:

- Tier A is protected: gap `0.000000`, gate active rate `0.000000`.
- Tier B gains modestly: hybrid gap `+0.021451`.
- Tier C gains more strongly: hybrid gap `+0.079503`.
- Large-growth cases benefit most: hybrid gap `+0.118728`.
- Small-growth cases are protected: gap `0.000000`.
- Zero-growth cases still have a small harm: gap `-0.006243`, caused by false gate activations.

Interpretation:

The prototype confirms that the decomposition can produce a deployable improvement over LOCF without direct test-set policy tuning. The gate protects low-change Tier A cases and concentrates gains in Tier C and large-growth cases. However, the policy is not yet competitive with the direct ResUNet overlap benchmark from the compact SRD run (`~0.815` mean Dice). This means the current decomposition is analytically useful and deployable in a conservative sense, but the method is not yet the final forecasting model.

Research implication:

The next methodological bottleneck is budget/gating quality, not growth-front ranking alone. The hybrid score gives only a small improvement over ResUNet image+mask score under the same budget/gate, suggesting that ranking is useful but budget selection dominates final Dice. A stronger method will likely need either adaptive budget prediction, uncertainty-aware growth allocation, or a policy that allows both growth and shrinkage instead of only adding candidate growth to the persistence mask.

## July 16, 2026 - Hybrid Policy vs Direct ResUNet Bottleneck Analysis Added

Added and pushed:

- `scripts/analyze_hybrid_vs_direct_model.py`

Purpose:

The hybrid forecast policy improves over LOCF but still trails the direct ResUNet Dice benchmark. This script diagnoses that gap sample-by-sample by comparing:

- direct ResUNet Dice;
- validation-selected hybrid policy Dice;
- LOCF Dice;
- growth volume;
- loss/shrinkage volume;
- net growth direction;
- gate activation;
- budget-to-growth ratio.

Research question:

Where does the direct model's advantage come from? Candidate explanations include better shrinkage/loss handling, better implicit growth budgeting, soft boundary correction, or cases where the hybrid gate is inactive/mis-budgeted.

Why this matters:

If direct ResUNet mainly wins on shrinkage/loss or boundary correction, then the next method should not simply add ranked growth to LOCF. It should become a fuller decomposition:

`future mask = persistence core + shrinkage/loss correction + budgeted growth-front expansion`

## July 16, 2026 - SRD Hybrid vs Direct ResUNet Bottleneck Result

Ran the bottleneck analysis comparing the validation-selected hybrid forecast policy against direct ResUNet image+mask forecasts.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_hybrid_vs_direct_resunet_v1`

Overall:

- direct ResUNet mean Dice: `0.814833`;
- hybrid policy mean Dice: `0.777026`;
- LOCF mean Dice: `0.746955`;
- direct minus hybrid mean Dice: `+0.037807`;
- bootstrap CI for direct minus hybrid: `[0.023325, 0.054493]`;
- direct beats hybrid in `57.7%` of samples;
- hybrid beats direct in `35.0%` of samples.

By absolute growth bin:

- large nonzero growth:
  - direct Dice: `0.934564`;
  - hybrid Dice: `0.824713`;
  - direct minus hybrid: `+0.109851`.
- medium nonzero growth:
  - direct minus hybrid: `+0.030002`.
- small nonzero growth:
  - direct minus hybrid: `-0.002425`, meaning the hybrid slightly outperforms direct ResUNet on average.
- zero growth:
  - direct minus hybrid: `+0.008561`, with hybrid still hurt by false gate activation.

By net growth direction:

- net-growth cases:
  - direct minus hybrid: `+0.045781`.
- net-shrinkage cases:
  - direct minus hybrid: `+0.014186`.
- net-stable cases:
  - no difference; all methods perfect.

Gate activity:

- inactive gate cases:
  - direct minus hybrid: `+0.029730` because the hybrid falls back to LOCF while direct ResUNet can still make useful corrections.
- active gate cases:
  - direct minus hybrid: `+0.050428`, indicating that even when the hybrid activates, the direct model has better growth allocation/shape correction.

Correlations with direct-model advantage:

- strongest Spearman correlations:
  - relative new growth: `0.463840`;
  - growth volume: `0.443620`;
  - growth-to-loss ratio: `0.426018`;
  - net delta volume: `0.416230`.
- loss volume and relative loss are weakly correlated with direct advantage.

Interpretation:

This result changes the bottleneck diagnosis. The direct ResUNet advantage is not primarily due to shrinkage/loss correction. It is strongest in large-growth and net-growth cases, meaning the direct model is likely doing a better job with growth magnitude, soft boundary allocation, or shape-consistent expansion than the hard top-k hybrid policy. The hybrid policy protects small-growth cases and improves over LOCF, but its hard budgeted growth addition is too crude for large-growth cases.

Research implication:

The next method should not only add a shrinkage module. The more important next step is a softer growth allocation mechanism: use the growth-front/image-context ranking as a probability or risk field, but let the model infer spatially coherent growth magnitude rather than enforcing a hard previous-growth top-k budget. A future method could still include shrinkage/loss correction, but the current evidence says growth magnitude and shape allocation are the dominant bottlenecks.

## July 16, 2026 - Low-Growth Operating-Regime Analysis Added

Added and pushed:

- `scripts/analyze_low_growth_operating_regime.py`

Motivation:

The hybrid policy trails direct ResUNet overall, especially on large-growth cases. However, for short-term forecasting, many cases may be low-growth or near-persistent. In that setting, a conservative hybrid policy can still be valuable if it protects LOCF-like behavior and avoids unnecessary neural overcorrection.

Purpose:

This script uses the bottleneck output to quantify:

- how often low-growth cases occur by horizon;
- whether the hybrid policy is preferable to direct ResUNet in low-growth cases;
- whether the hybrid preserves LOCF behavior in low-growth cases;
- best-method counts across growth bins and horizon-growth bins.

Important framing:

The defensible claim should not be that the hybrid improves Dice over LOCF in low-growth cases. The more precise claim is that the hybrid can preserve persistence in low-growth regimes and may be preferable to direct neural forecasting when direct models overcorrect.

## July 16, 2026 - Low-Growth Operating-Regime Result

Ran the low-growth operating-regime analysis using the SRD hybrid-vs-direct bottleneck output.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_low_growth_operating_regime_v1`

Overall operating mix:

- total samples: `123`;
- low-growth rate using `zero + small_nonzero`: `0.471545`;
- H1 low-growth rate: `0.439024`;
- H2 low-growth rate: `0.487805`;
- H3 low-growth rate: `0.487805`.

Low-growth subset overall (`zero + small_nonzero`, n=`58`):

- LOCF Dice: `0.694602`;
- hybrid Dice: `0.691804`;
- direct ResUNet Dice: `0.694303`;
- hybrid minus direct: `-0.002500`, bootstrap CI `[-0.011346, 0.003810]`;
- hybrid minus LOCF: `-0.002799`, CI `[-0.008396, ~0]`.

This means the broad low-growth bucket does not support a claim that hybrid is better overall. It is essentially comparable but slightly below LOCF/direct because zero-growth false activation can hurt.

Short-horizon low-growth subset (H1 + zero/small_nonzero, n=`18`):

- LOCF Dice: `0.879135`;
- hybrid Dice: `0.879135`;
- direct ResUNet Dice: `0.873675`;
- hybrid minus direct: `+0.005459`, bootstrap CI `[0.000207, 0.011516]`;
- hybrid minus LOCF: `0.000000`;
- direct minus LOCF: `-0.005459`, CI `[-0.011516, -0.000207]`.

Best-method counts:

- small nonzero cases: hybrid best in `19/32` (`59.4%`), direct best in `13/32` (`40.6%`);
- zero cases: hybrid best in `15/26` (`57.7%`), direct best in `10/26` (`38.5%`), LOCF best in `1/26` (`3.8%`).
- H1 zero cases: hybrid best in `5/5` cases.
- H1 small-nonzero cases: hybrid best in `8/13` (`61.5%`).

Interpretation:

The correct claim is narrower and stronger: in the short-horizon low-growth operating regime, the hybrid preserves LOCF/persistence and avoids the direct ResUNet overcorrection penalty. This supports a selective regime-aware policy, but not a broad claim that hybrid dominates all low-growth cases. For all low-growth cases pooled across horizons, the hybrid is approximately comparable but not superior because false activation in zero-growth cases can still hurt.

Research implication:

The method direction should emphasize selective operating regimes. Direct ResUNet is preferable in large-growth cases. Hybrid/persistence-aware behavior is preferable in short-horizon low-growth cases. A future deployable method should learn or infer this operating regime rather than forcing one forecasting behavior across all samples.

## July 19, 2026 - Additive Growth-Field Control Against Threshold-Tuned Direct ResUNet

A validation-threshold-tuned direct ResUNet control was run to test whether the additive growth-field improvement was merely a threshold-calibration artifact.

Outputs reviewed:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_direct_threshold_control_v1`
- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/srd_calibrated_growth_field_v1/growth_field_vs_tuned_direct_*`

Direct threshold control:

- validation selected direct ResUNet threshold: `0.70`
- tuned direct ResUNet test Dice: `0.814331`
- default direct ResUNet test Dice: `0.814833`
- tuned direct vs default direct gap: `-0.000502`
- bootstrap CI for tuned direct vs default direct: `[-0.002017, 0.001004]`

Thus, validation threshold tuning does not explain the additive growth-field result.

Additive growth field vs tuned direct ResUNet:

- raw model-probability growth field Dice: `0.818525`
- tuned direct ResUNet Dice: `0.814331`
- paired gap vs tuned direct: `+0.004194`
- bootstrap CI: `[+0.001778, +0.006699]`
- win rate vs tuned direct: `0.617886`
- mean selected voxels: growth field `3283.40`, tuned direct `9997.44`

Calibrated growth field vs tuned direct:

- calibrated growth-field Dice: `0.817888`
- paired gap vs tuned direct: `+0.003557`
- bootstrap CI: `[+0.001052, +0.006152]`
- win rate vs tuned direct: `0.544715`

By growth bin for raw model-probability growth field vs tuned direct:

- `large_nonzero`: `+0.006296`, win rate `0.522727`
- `medium_nonzero`: `+0.006943`, win rate `1.000000`
- `small_nonzero`: `+0.006170`, win rate `0.931818`
- `zero`: `-0.003658`, win rate `0.115385`

Interpretation:

The additive growth-field formulation appears to be a real structural improvement, not just threshold tuning. It preserves the current observed tumor and adds learned probable-growth voxels, producing a small but stable gain over both default direct ResUNet and threshold-tuned direct ResUNet. The improvement is clearest in nonzero-growth cases, including small-growth cases, while the remaining weakness is true zero-growth where the growth field can still add unnecessary voxels.

Current methodological statement:

`future tumor = observed current tumor + learned growth field`

This gives a more interpretable persistence-preserving forecasting formulation than full future-mask regeneration. The next critical stage is to test this formulation on a fuller SAILOR setup using patient-level splits and sliding longitudinal windows rather than the earlier tiny 4-patient / 12-sample stress check.

## July 19, 2026 - Pivot From SRD Controls to Fuller SAILOR Window Audit

After the additive growth-field method beat both default direct ResUNet and validation-threshold-tuned direct ResUNet on SRD, the next research step was defined as a fuller SAILOR real-data audit rather than additional SRD refinements.

Rationale:

- The earlier SAILOR experiments used only a tiny 4-patient / 12-sample stress check.
- The local TaDiff code confirms that SAILOR is treated as a longitudinal dataset with approximately 25 valid patients and random contiguous 4-session windows.
- Sliding windows can increase real-data forecast samples, but must preserve patient-level train/validation/test splits to avoid leakage.

Planned audit:

- enumerate sliding longitudinal windows for input lengths `3,4,5` and configurable horizons;
- report patient counts, session counts, follow-up days, interval distributions, treatment changes, LOCF Dice, future growth volume, and growth bins;
- summarize by patient, input-window length, horizon, growth bin, and net growth/shrinkage direction.

A new script was added:

- `scripts/audit_longitudinal_windows.py`

This script is intended to map the fuller SAILOR real-data forecasting opportunity before training or testing modified forecasting methods.

## July 19, 2026 - Fuller SAILOR Longitudinal Window Audit Reviewed

A fuller SAILOR sliding-window audit was run using input lengths `3,4,5`, horizon `1`, and all available patients in the processed SAILOR folder.

Output directory:

- `/content/drive/MyDrive/synthetic_tumor_benchmark/outputs/sailor_longitudinal_window_audit_v1`

Main audit findings:

- patient summary listed `26` processed patients, while `25` patients produced valid forecast windows because `sub-19` has only 3 sessions and cannot produce a 3-input to next-session forecast window.
- total enumerated windows across input lengths `3,4,5`: `406`
- windows are not independent; they are overlapping windows from the same patients.
- overall LOCF Dice across all enumerated windows: mean `0.655169`, median `0.707335`
- mean target delta: `84.60` days, median `89` days
- mean future new-growth volume: `29834.93` voxels, median `12642.5` voxels
- most windows fall into `large_nonzero`: `357/406`
- `medium_nonzero`: `44/406`
- `small_nonzero`: `5/406`
- no `zero` growth windows under the current new-growth definition, because even net-shrinkage cases can contain spatially new voxels.
- net-direction split: `223` net-growth windows and `183` net-shrinkage windows.

Interpretation:

The fuller SAILOR data substantially improves the real-data opportunity compared with the earlier 4-patient / 12-sample stress check. However, the window count should not be treated as independent sample size. Splits and uncertainty must be patient-level. The audit also shows that SAILOR is dominated by spatial new-growth windows under the current growth definition, unlike SRD where zero/small-growth regimes were common and analytically important.

Important cleanup required before model testing:

1. choose a single primary window protocol, likely input length `3` and horizon `1`, to match the current short-horizon framing;
2. decide whether to exclude TaDiff QC-questionable patients `sub-13` and `sub-23`;
3. report patient counts separately from window counts;
4. define growth states more carefully for SAILOR using both spatial new-growth and net volume direction, since `zero_growth_rate=0` under the current spatial-new-growth definition.

Next planned step:

Run a cleaner SAILOR audit with input length `3`, horizon `1`, and a TaDiff-compatible patient subset excluding `sub-13`, `sub-23`, and naturally excluding `sub-19` from windows due to insufficient sessions. Then define patient-level train/validation/test splits for the fuller real-data method test.

## 2026-07-19 - Clean SAILOR Longitudinal Window Audit

- Reviewed the cleaned SAILOR input-3 to next-scan longitudinal audit.
- Usable setup after excluding QC-questionable subjects: 136 forecasting windows across 23 patients.
- Overall LOCF Dice is 0.6394 mean / 0.6901 median, with mean next-scan interval 74.5 days and median interval 84 days.
- The cohort is dominated by spatial new-growth windows: 129/136 windows fall into the large nonzero spatial new-growth bin; zero spatial-growth windows are absent.
- Net direction remains mixed: 75 windows show net growth and 61 show net shrinkage. LOCF is harder in net-growth windows (mean Dice 0.6076) and easier in net-shrinkage windows (mean Dice 0.6786).
- Interpretation: SAILOR is now large enough for a careful real-data stress test using patient-level splits, but it is not a clean zero-growth/persistence dataset. Any additive growth-field method must be evaluated against direct forecasting and must explicitly track net-growth versus net-shrinkage behavior.
- Immediate implication: before training/testing modified methods on SAILOR, define patient-level splits and keep patient-level uncertainty in all summaries to avoid over-counting overlapping windows.

## 2026-07-19 - SAILOR Patient Split First-Pass Critique

- Reviewed the first patient-level train/validation/test split from the clean SAILOR input-3/horizon-1 audit.
- The first split was leakage-safe, but not sufficiently balanced for method testing: train had 14 patients but only 50 windows, while validation/test had 40/46 windows; test also had easier LOCF performance and lower relative new-growth than train/validation.
- Decision: do not use this first greedy split as the main experimental split. It is useful only as a diagnostic.
- Updated the split-manifest tool to search many seeded patient-level assignments and optimize global balance across window count, LOCF difficulty, net-growth burden, relative new-growth, delta days, and treatment-change burden while preserving patient-level separation.

## 2026-07-19 - Accepted SAILOR Patient Split v2

- Reviewed the regenerated SAILOR input-3/horizon-1 patient-level split (`sailor_h1_l3_patient_splits_v2`).
- This split is substantially more balanced than the first greedy split and is acceptable as the first real-data method-test foundation.
- Overall windows/patients: train 82 windows / 14 patients, validation 28 / 5, test 26 / 4.
- LOCF difficulty is now comparable: train mean 0.6458, validation 0.6163, test 0.6443.
- Net-growth burden is comparable: train 0.5488, validation 0.5357, test 0.5769.
- Relative spatial new-growth is comparable: train 1.1033, validation 1.1323, test 1.0107.
- Remaining limitation: validation contains only large-nonzero spatial-growth windows, while test contains medium/small cases; however the rare medium/small cases are too sparse for perfect balancing. Main reporting should emphasize net-growth versus net-shrinkage rather than small/medium/large spatial-growth bins on SAILOR.
- Decision: use v2 for the first SAILOR model stress test, with patient-level uncertainty and stratification by net direction.

## 2026-07-19 - SAILOR Manifest ResUNet vs LOCF First Result

- Completed first SAILOR patient-level manifest baseline comparison using input length 3, horizon 1, split v2.
- LOCF: validation mean Dice 0.6163, test mean Dice 0.6443.
- ResUNet image+mask, 12 epochs, seed 42: validation mean Dice 0.6338, test mean Dice 0.6465.
- Overall paired gain is small: validation +0.0175, test +0.0022.
- Regime split is much clearer:
  - Net-growth windows: validation +0.0340 with 100% win rate; test +0.0310 with 80% win rate.
  - Net-shrinkage windows: validation -0.0016; test -0.0371.
- Interpretation: learned forecasting signal exists on SAILOR, but it is direction-dependent. Direct ResUNet improves growth cases and can damage shrinkage cases. This supports moving from a single direct-mask forecast toward a direction/transition-aware forecasting policy.
- Important caution: do not claim overall SAILOR superiority yet. The honest result is conditional: model useful on net-growth windows, unreliable on shrinkage windows.

## 2026-07-19 - Direction-Gated SAILOR Policy Script

- Added a direction-gated policy evaluator for SAILOR manifest experiments.
- Policy idea: train a simple logistic gate using only pre-forecast input-history descriptors to predict whether the next transition is net-growth. If predicted net-growth, use ResUNet; otherwise use LOCF.
- Features intentionally exclude future growth, future target volume, and future Dice. Current features: input window length/span, delta days, input treatment, input volume, previous growth/loss volume, previous growth ratio, and treatment change within the input window.
- The script trains on train windows, selects probability threshold on validation by policy Dice, and reports validation/test policy performance, classifier metrics, by-net-direction summaries, by-patient summaries, and patient-bootstrap CIs.
- This directly tests whether the SAILOR oracle headroom can be approximated without future leakage.

## 2026-07-19 - SAILOR Direction Gate v1 Result

- Ran the first non-oracle direction-gated policy on SAILOR manifest split v2.
- The validation-selected threshold was 0.05, causing the gate to predict net-growth for every validation and test case.
- Therefore the gated policy collapsed to direct ResUNet: validation 0.6338, test 0.6465.
- This failed to recover oracle headroom on test: oracle test mean was 0.6622, while direct/gated ResUNet was 0.6465 and LOCF was 0.6443.
- Classifier AUC was non-random but weak: validation 0.6513 and test 0.6364. Balanced accuracy at the selected threshold was 0.5 because shrinkage cases were all falsely predicted as growth.
- Interpretation: input-history features contain some direction signal, but selecting the threshold directly by validation Dice is too permissive because validation shrinkage harm was tiny. The next diagnostic should evaluate stricter/safety-first thresholds that trade off growth recall against shrinkage false positives.

## 2026-07-19 - Direction Gate Threshold Sweep Diagnostic

- Reviewed manual threshold sweep for the SAILOR direction gate.
- Validation Dice is maximized by low thresholds that effectively use ResUNet everywhere; safety-constrained thresholds reduce shrinkage false-growth firing but also lose too many true growth cases.
- Test does improve modestly at stricter thresholds: threshold 0.55/0.60 gives test policy mean 0.6512 versus LOCF 0.6443 and ResUNet 0.6465. Threshold 0.80/0.85 gives 0.6507, but with very low growth recall.
- However, these thresholds are not selected by the validation policy objective; they are visible only after inspecting test behavior. Therefore they cannot be claimed as a validated deployable gate.
- Interpretation: input-history direction gating alone is weak/unstable. The better next step is to use inference-time model-derived geometry, e.g., predicted target volume, predicted net volume change, predicted growth/loss volume, and probability mass, to decide when a direct model should be trusted or suppressed.

## 2026-07-19 - SAILOR Model-Geometry Gate Result

- Ran model-derived geometry gating on the SAILOR manifest ResUNet checkpoint.
- Validation selected `pred_relative_net_delta >= -0.5`, which uses ResUNet for every case. Therefore the geometry-gated policy collapsed to direct ResUNet, same as the input-history gate.
- Geometry-gated policy: validation 0.6338, test 0.6465; LOCF test 0.6443; oracle direction policy test 0.6622.
- By net direction, the same pattern remains: ResUNet helps net-growth windows (+0.0310 test) and hurts net-shrinkage windows (-0.0371 test), but the geometry scores do not reliably identify shrinkage cases under validation-selected thresholds.
- Manual test sweep shows no robust geometry threshold that improves meaningfully over direct ResUNet; several stricter thresholds reduce use-model rate but also drop growth recall and often lower test Dice.
- Interpretation: simple post-hoc gating using either input-history features or model-predicted geometry is not enough. The next method branch should target the structural limitation directly: a shrinkage-aware / residual-change formulation, rather than trying to choose between LOCF and a direct full-mask model after the fact.

## 2026-07-19 - SAILOR Transition Error Decomposition Result

- Reviewed transition-level error decomposition for direct ResUNet on SAILOR manifest split v2.
- Direct ResUNet still preserves stable tumor core well: stable-core recall is about 0.979 overall and about 0.986 on test net-shrinkage windows.
- The model is not primarily failing by deleting stable core. Instead, two distinct limitations appear:
  1. It underpredicts true new growth volume: on test, true spatial growth averages 20,789 voxels while predicted growth averages 6,664 voxels; growth recall is only about 0.34 overall.
  2. It severely underpredicts true loss/shrinkage: on test, true loss averages 10,646 voxels while predicted loss averages 2,565 voxels; loss recall is only about 0.12 overall.
- In net-growth windows, ResUNet improves Dice despite missing large amounts of true growth because even partial growth localization improves over LOCF.
- In net-shrinkage windows, ResUNet hurts Dice mostly because it adds false outside growth and still does not remove enough lost tumor; test shrinkage growth precision is low (0.27), and predicted loss volume is far below true loss volume.
- Method implication: the next model should not be merely a post-hoc gate. It should explicitly forecast residual change with separate growth and loss/shrinkage heads, reconstructing future mask as `(input AND NOT loss) OR growth`.

## 2026-07-19 - Residual-Change SAILOR Runtime Diagnostic

- Attempted to train the residual growth/loss SAILOR model on full-resolution SAILOR volumes using an L4 GPU, but the Colab cell produced no progress output after a long wait.
- Local copy diagnostics showed Google Drive I/O was a major contributor but not the only bottleneck: loading `sub-02_image.npy` from Drive took about 39 seconds, while loading the local copy took about 6.5 seconds.
- Full SAILOR volumes are large (`229 x 193 x 193`) and image arrays are stored as flattened session-by-modality blocks, so a direct full-volume image+mask residual run is much heavier than SRD experiments.
- Updated the residual-change runner to support staged, observable experiments: correct session/modality image reshaping, optional spatial downsampling, train/validation/evaluation sample caps, and per-batch progress printing.
- Interpretation: do not treat the silent long run as evidence against the residual-change idea. It is an execution-design problem. We should first validate correctness with a capped/downsampled smoke test, then scale carefully to larger runs.
