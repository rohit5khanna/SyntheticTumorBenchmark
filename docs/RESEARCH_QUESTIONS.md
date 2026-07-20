# Research Questions

## Purpose

This document defines the scientific backbone of the project.

Its purpose is to prevent the work from drifting into:

- disconnected experiment cycles,
- benchmark-only comparisons,
- or short-term submission-oriented decisions.

Instead, all major experiments, analyses, and methodological ideas should be traceable to a small set of core research questions.

## Central research theme

The broad project theme is:

> short-horizon tumor forecasting should not be treated as a uniform prediction problem

The project investigates whether forecasting difficulty depends on tumor regime and case characteristics, and whether that information can be used to build better forecasting methods.

---

## RQ1. Is short-horizon tumor forecasting fundamentally regime-dependent?

### Core idea

The first question is whether short-horizon tumor forecasting behaves differently across different tumor-growth regimes rather than constituting one homogeneous prediction task.

### Why this matters

If the task is regime-dependent, then:

- uniform model evaluation is incomplete;
- persistence-based forecasts may be appropriate in some cases but not others;
- and any later forecasting method should account for this heterogeneity.

### What this question is asking

1. Do different regimes produce systematically different forecasting difficulty?
2. Do different model families behave differently across those regimes?
3. Are there settings in which persistence dominates and settings in which learned models add value?

### Evidence needed

To support this question, the project should provide:

- tier-wise performance comparisons;
- horizon-wise performance comparisons;
- per-case and per-patient breakdowns;
- cross-model comparisons across persistence, learned, and mechanism-guided families;
- synthetic-to-real comparison where possible.

### What would weaken this question

This question is weakened if:

- all models behave similarly across all tiers/horizons;
- regime definitions do not correspond to meaningful differences in forecasting behavior;
- or observed differences are unstable, noisy, or artifact-driven.

### Current status

Current evidence supports this direction more strongly than the initial draft did. SRD shows clean tier/horizon separation, while SAILOR shows that real one-session transitions are not uniformly easy: mixed growth/loss, non-boundary growth, and high transition burden make some immediate forecasts much harder than others.

The remaining task is not to claim that every regime label is final, but to make the evidence chain clear: which transition properties are consistently associated with LOCF breakdown, learned-model value, or residual-growth difficulty.

---

## RQ2. Which tumor and temporal properties explain short-horizon forecasting difficulty?

### Core idea

If regime dependence exists, then the next question is what measurable properties actually govern that difficulty.

These properties should ideally be available at the forecast origin, not inferred from future information.

### Why this matters

Without identifying such properties, the project risks remaining at the level of descriptive tier comparisons.

With them, the project becomes more analytical and can move toward a more informative or actionable understanding of forecasting.

### Candidate properties

The current project suggests that relevant properties may include:

- input tumor volume;
- recent growth rate;
- future growth magnitude;
- morphology descriptors;
- compactness / elongation proxies;
- treatment state at input;
- temporal spacing between sessions;
- and possibly richer field- or structure-based descriptors later.

### What this question is asking

1. Which forecast-origin features are associated with easy versus hard cases?
2. Which features are associated with persistence dominance?
3. Which features are associated with learned-model gains?
4. Do different model families fail for different reasons?

### Evidence needed

To support this question, the project should provide:

- grouped performance summaries by feature bins;
- pairwise win/loss analysis between models;
- case-type analysis;
- morphology/treatment summaries;
- and careful interpretation of which properties appear robustly linked to forecasting behavior.

### What would weaken this question

This question is weakened if:

- no tumor or temporal descriptors show stable relationships with performance;
- the relationships disappear under small perturbations;
- or the descriptors are too synthetic-specific to inform broader forecasting understanding.

### Current status

There is now enough analysis to treat this as one of the strongest parts of the project. The LOCF operating-range, transition-taxonomy, persistence-breakdown, feature-ablation, and SAILOR/SRD domain-gap analyses all point to measurable transition descriptors as useful explanatory variables.

The strongest current descriptors are change burden, recent growth tendency, input tumor volume, net direction, growth/loss mixture, and boundary versus non-boundary spatial change. Calendar time remains important, but it is not sufficient by itself to define short-term difficulty.

---

## RQ3. Can regime information be encoded as a prior to improve forecasting?

### Core idea

If short-horizon forecasting is regime-dependent and if measurable descriptors explain part of that difficulty, then a natural next question is whether this information can be encoded into forecasting models as a prior or conditioning signal.

### Why this matters

This question turns the project from:

- descriptive evaluation

into:

- analysis-driven method design.

It is the main bridge from the current evaluation framework to a genuine forecasting-method contribution.

### What this question is asking

1. Can regime descriptors help a forecasting model avoid learning uniformly over all cases?
2. Can conditioning on regime information improve forecasting in difficult settings?
3. Can regime-aware residual learning improve upon persistence in a more targeted way?
4. Can regime awareness make learned forecasts more robust, interpretable, or stable?

### Candidate methodological forms

Possible implementations include:

- regime feature channels;
- regime embeddings injected into the network;
- regime-conditioned residual learning over persistence;
- regime-aware gating or hybrid forecasting;
- or later, richer probabilistic/field-informed representations if justified.

### Evidence needed

To support this question, the project should provide:

- controlled comparison between plain and regime-conditioned models;
- performance by regime/tier and horizon;
- evidence of gains in hard cases rather than only average gains;
- and careful ablations showing which encoded information actually matters.

### What would weaken this question

This question is weakened if:

- regime conditioning gives no meaningful improvement;
- gains are too small, unstable, or inconsistent;
- or the conditioning mechanism only adds complexity without clarifying the model’s behavior.

### Current status

This question remains open. The growth-front and growth-only residual experiments are useful probes, but they should not yet be treated as a complete method contribution. They suggest that a persistence-preserving, growth-aware method is plausible; they do not yet establish a strong model-level result.

The next method phase should be driven by the analysis, not by architecture shopping. A credible method should explicitly preserve persistence, add growth only where the evidence is strong, and handle apparent loss/shrinkage as a separate uncertainty-sensitive process rather than forcing symmetric growth/loss prediction into one output.

---

## RQ4. What does the synthetic-to-real comparison reveal about controlled mechanisms versus real transition complexity?

### Core idea

SRD is useful not because it perfectly imitates real longitudinal tumor data, but because it gives us a controlled setting where growth, shrinkage, persistence, horizon, and tier effects can be separated.

The synthetic-to-real question should therefore be framed carefully: not "does SRD reproduce SAILOR?", but "which mechanisms can SRD isolate, and which real-data transition complexities appear in SAILOR but are absent from SRD?"

### Why this matters

Without this question, the project risks becoming isolated inside a synthetic framework or, equally problematically, treating a controlled synthetic dataset as if it were a clinical surrogate.

With it, the synthetic work becomes a controlled tool for probing mechanisms, and SAILOR becomes the place where we audit real transition complexity.

### What this question is asking

1. Which SRD mechanisms appear relevant to real-data forecasting behavior?
2. Which SAILOR transition patterns are not covered by the current SRD design?
3. Does real short-horizon forecasting show similar persistence-dominant and growth-sensitive behavior?
4. What does the synthetic-real mismatch teach us about the real forecasting problem?

### Evidence needed

To support this question, the project should provide:

- a careful real-data bridge using available datasets such as `SAILOR`;
- explicit comparison of synthetic and real performance patterns;
- explicit comparison of transition structure, not only model Dice;
- discussion of which differences arise from unobserved treatment, registration, imaging artifacts, or other real-data complications;
- and strong caution against overclaiming SRD as a real-data surrogate.

### What would weaken this question

This question is weakened if:

- synthetic findings are presented as directly transferable without evidence;
- real-data analysis remains too thin;
- or the synthetic-to-real bridge is handled only rhetorically rather than analytically.

### Current status

The bridge is now more analytical than before. The SAILOR-vs-SRD transition domain-gap analysis shows that SRD is cleaner and more axis-separated, while SAILOR contains mixed growth/loss, larger scale, non-boundary growth, core loss, and high-change tails.

This should be treated as a finding, not as an instruction to immediately generate a more complicated synthetic dataset. For the current phase, SRD should remain the controlled mechanism-isolation environment, and SAILOR should carry the real transition-complexity analysis.

---

## Project-level hypotheses

At the current stage, the project is built around the following working hypotheses:

### H1

Short-horizon tumor forecasting is regime-dependent rather than a uniform prediction problem.

### H2

Forecast-origin tumor and temporal descriptors explain a meaningful part of this regime dependence.

### H3

Encoding regime information as a prior or conditioning signal can improve forecasting beyond uniform learned models.

### H4

Synthetic regimes are useful not because they perfectly mimic real data, but because they expose controlled structures that help explain and stress-test real short-horizon forecasting difficulties.

These remain hypotheses, not final claims.

Current working decision: do not add another synthetic generation cycle just to make SRD look more like SAILOR. Use SRD for isolated mechanism analysis, use SAILOR for real transition complexity, and make the domain gap explicit.

---

## What the project should not become

This document also defines what would count as project drift.

The project should **not** become:

- an endless sequence of model additions with no sharpened question;
- a synthetic benchmark paper with weak analysis;
- a deadline-driven submission exercise that overstates incomplete evidence;
- or a method paper with little analytical grounding.

---

## Standard for future experiments

Every substantial new experiment should answer at least one of the following:

1. Does it strengthen evidence for regime dependence?
2. Does it reveal which tumor/data properties drive forecasting difficulty?
3. Does it help construct or test a regime-conditioned prior?
4. Does it clarify the synthetic-to-real bridge without pretending SRD is a clinical surrogate?

If the answer is no, the experiment should probably be deprioritized.

---

## Summary

The project is now defined by four core research questions:

1. Is short-horizon tumor forecasting regime-dependent?
2. Which tumor and temporal properties explain that difficulty?
3. Can regime information be encoded as a prior to improve forecasting?
4. How much of the synthetic regime structure transfers to real data?
   More precisely, what does the SRD-SAILOR comparison reveal about controlled mechanisms versus real transition complexity?

These questions should guide:

- experiment design,
- literature review,
- model development,
- and interpretation of results.

They also define the threshold for whether the project is becoming a real contribution rather than remaining a collection of experiments.
