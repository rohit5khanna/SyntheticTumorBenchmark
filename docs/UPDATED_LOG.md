# Updated Log

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
