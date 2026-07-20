# Short-Horizon Residual Forecaster: Design Blueprint

This document defines a lightweight forecasting direction built around the evidence gathered so far. It is intentionally a design blueprint, not a manuscript section and not an implementation note. The purpose is to prevent another round of experiments from drifting into model-chasing without a clear modeling principle.

## 1. Core Modeling Principle

Short-horizon tumor forecasting should not begin by asking a model to regenerate the entire future tumor mask.

The immediate next scan is often dominated by persistence: much of the tumor present at the current scan remains present at the next scan. A learned model that predicts the entire mask must spend capacity relearning this persistent structure before it can express the clinically and analytically important residual change.

However, "short horizon" should not be defined only by the number of sessions ahead. In real longitudinal imaging, the next scan may occur after very different calendar intervals across patients. A one-session forecast over 14 days and a one-session forecast over 140 days are not the same biological task. Likewise, a short calendar interval can still be difficult if the tumor has a high growth rate.

The method should therefore treat short-horizon forecasting as a combination of:

- session horizon: how many scans ahead are predicted;
- calendar horizon: how many days elapse between input and target;
- biological or change horizon: how much tumor change occurs per unit time.

The proposed modeling principle is therefore:

```text
future tumor = current tumor + small controlled residual change
```

More concretely:

```text
M_hat(t + 1) = M(t) updated by predicted local growth and, later, predicted loss
```

The current mask is treated as a strong prior, not as just another input channel.

This principle is expected to be most appropriate when the calendar horizon and/or expected biological change are small enough that persistence remains a strong baseline. Identifying the limit of that assumption is part of the research problem.

## 2. Why This Direction Fits The Current Evidence

Our existing results point toward a consistent decomposition:

- Last observation carried forward is difficult to beat at short horizon because the persistent tumor core is usually large.
- LOCF strength should depend on time gap and growth/change rate, not only session horizon.
- Direct full-mask ResUNet can help more in net-growth cases than in net-shrinkage cases.
- Residual growth probability maps contain useful ranking signal.
- Symmetric growth/loss residual prediction has not worked cleanly.
- Apparent loss/shrinkage appears less directly learnable and may depend on treatment, surgery, registration, segmentation variability, or nonlocal biological response.

The implication is that the first credible method should preserve persistence and learn only the component that appears most learnable: new spatial growth outside the current tumor. But before claiming this as a method, we need to quantify where the persistence prior is valid and where it breaks down.

## 2.1 LOCF Limit Analysis Required Before Modeling

The project should explicitly estimate the operating range of LOCF.

Minimum analysis:

- LOCF Dice versus `delta_days`;
- LOCF Dice versus new-growth volume;
- LOCF Dice versus new-growth rate;
- LOCF Dice versus net-volume-change rate;
- LOCF Dice versus absolute-change rate;
- interaction table for interval bin by growth-rate bin.

Candidate rate definitions:

```text
new_growth_rate = new_growth_volume_vox / delta_days
net_volume_rate = (target_volume_vox - input_volume_vox) / delta_days
absolute_change_rate = (new_growth_volume_vox + loss_volume_vox) / delta_days
```

This analysis should tell us whether "short-term" means:

```text
next scan only
```

or the more precise:

```text
next scan under low-to-moderate elapsed time and low-to-moderate biological change rate
```

The second definition is more scientifically honest and more useful for regime-aware evaluation.

## 3. First Prototype: Growth-Only Residual Forecaster

The first model should predict future growth outside the current tumor mask.

### Input

For each forecasting window:

```text
MRI history:   I(t-k+1), ..., I(t)
Mask history:  M(t-k+1), ..., M(t)
Time context:  scan intervals or days-to-target
Treatment:     optional treatment/context flag if available
```

The current practical SAILOR setting uses input length 3:

```text
I(t-2), I(t-1), I(t)
M(t-2), M(t-1), M(t)
```

With multimodal MRI, the exact channel count depends on how modalities and masks are stacked. The key design requirement is that the model can identify:

- current tumor boundary;
- recent expansion direction;
- local intensity context;
- elapsed time between scans;
- treatment context when available.

### Target

The target is not the whole future mask.

```text
G(t + 1) = M(t + 1) \ M(t)
```

where `G(t + 1)` is the set of voxels that are absent from the current tumor mask but present in the next tumor mask.

In binary form:

```text
G(x, t + 1) = 1 if M(x, t) = 0 and M(x, t + 1) = 1
G(x, t + 1) = 0 otherwise
```

The model predicts:

```text
P_growth(x) = P(G(x, t + 1) = 1 | history)
```

### Inference Rule

The final forecast preserves the current mask and adds only a controlled number of predicted growth voxels.

```text
M_hat(t + 1) = M(t) union TopK(P_growth outside M(t))
```

`TopK` can be defined by:

- a fixed probability threshold;
- a validation-selected growth budget;
- a time-conditioned growth budget;
- a patient-history-conditioned growth budget.

The first clean prototype should use validation-selected budget selection because the current SAILOR evidence suggests that unconstrained growth prediction can damage Dice and create unrealistic expansion.

## 4. Why Growth-Only First

Growth and loss should not be modeled as symmetric events at this stage.

Growth outside the current tumor has shown meaningful ranking signal in our diagnostics. Apparent loss has not. Loss may reflect true response, surgical removal, treatment effect, segmentation variability, registration artifacts, or intensity ambiguity. A model trained to remove tumor voxels without reliable context can damage the persistent core, which is exactly what short-horizon forecasting should avoid.

Therefore, the first method should be deliberately conservative:

```text
Preserve what is already tumor.
Rank where new tumor is most likely to appear.
Add a small controlled amount of growth.
Do not remove tumor unless loss becomes separately well-characterized.
```

This is not a permanent limitation. It is a principled starting point.

## 5. Candidate Architecture

The first architecture should be a lightweight CNN rather than a transformer.

Rationale:

- short-horizon changes are mostly local around the existing tumor boundary;
- SAILOR sample size is small;
- transformer-style architectures add parameter burden and may overfit;
- a compact CNN is easier to audit and deploy repeatedly for iterative forecasting.

### Suggested Backbone

```text
Input channels
  -> shallow residual CNN encoder
  -> local multiscale convolution blocks
  -> lightweight decoder
  -> sigmoid growth probability map
```

Recommended properties:

- small base channels, e.g. 4, 6, or 8;
- residual blocks rather than a large U-Net if memory is a concern;
- optional dilated convolutions for a larger boundary neighborhood;
- no aggressive downsampling that destroys small growth regions;
- output restricted or masked to outside-current-tumor voxels during loss/evaluation.

### Possible Variant: Boundary-Focused CNN

A more explicitly tumor-aware version can restrict learning to a band around the current tumor:

```text
candidate region = dilation(M(t), r) \ M(t)
```

The model predicts growth probability primarily within this candidate region. This would reduce class imbalance and make the method more interpretable. The danger is that it may miss distant or multifocal growth, so this should be tested rather than assumed.

## 6. Loss Function

The target is extremely imbalanced: true new-growth voxels are sparse compared with background.

A reasonable first loss is:

```text
L = weighted BCE + lambda * soft Dice loss
```

computed outside the current tumor mask, or inside a candidate region around the current tumor.

Important safeguards:

- do not let class weighting produce huge false-positive growth fields;
- monitor growth precision and recall separately;
- monitor full-mask Dice after adding growth to the current mask;
- select threshold/budget on validation data only;
- report patient-level uncertainty, not only window-level averages.

## 7. Evaluation Protocol

This method should not be judged by full Dice alone.

Minimum evaluation:

- full future-mask Dice;
- gap versus LOCF;
- win rate versus LOCF;
- growth average precision;
- growth ROC-AUC;
- growth recall at fixed growth budget;
- growth precision at selected budget;
- net-growth versus net-shrinkage subgroup results;
- patient-level bootstrap confidence intervals;
- seed stability.

Required baselines:

- LOCF;
- distance-to-current-mask growth ranking;
- direct ResUNet full-mask prediction;
- direct ResUNet probability used as a growth ranking map;
- growth-only residual CNN.

The strongest version of the claim would not be:

```text
Our model beats everything by large Dice margin.
```

It would be:

```text
A persistence-preserving residual model improves the growth component while avoiding the instability of full-mask residual prediction.
```

## 8. Iterative Rollout

The user-proposed repeated-use idea is important.

Once a one-step model is trained, we can recursively apply it:

```text
M(t) -> M_hat(t + 1) -> M_hat(t + 2) -> M_hat(t + 3)
```

This tests whether small predicted growth fields accumulate into plausible longer-horizon tumor evolution.

Risks:

- errors compound;
- false-positive growth can accumulate rapidly;
- uncertainty increases with each rollout;
- without new MRI intensity at future steps, the model may lose image context.

Possible rollout modes:

1. Mask-only rollout: update masks but reuse last available MRI.
2. Teacher-forced image rollout: use real MRI at each intermediate step when available, but forecast the next mask.
3. Hybrid assimilation rollout: update with new real scans when they arrive.

The most scientifically clean first test is teacher-forced evaluation across existing longitudinal windows, because it isolates whether the residual growth policy remains useful over repeated one-step transitions.

## 9. What Would Count As Success

Strong success:

- consistent positive gap over LOCF in net-growth cases;
- no major penalty in net-shrinkage cases;
- growth ranking clearly beats distance and random baselines;
- selected budget is stable across validation splits or seeds;
- patient-level bootstrap supports positive growth-case effect;
- behavior transfers at least directionally from SRD to SAILOR.

Moderate success:

- full Dice gains are small, but growth ranking is strong and interpretable;
- the model exposes where short-horizon forecasting is learnable versus not learnable;
- loss/shrinkage remains unresolved but is clearly characterized.

Failure that is still informative:

- growth ranking is not stable across seeds or patients;
- budget selection changes wildly;
- growth gains are only artifacts of a few patients;
- distance-to-mask performs as well as the learned model;
- gains vanish under patient-level bootstrap.

If this happens, the conclusion should shift away from proposing a method and toward showing the limits of short-horizon residual forecasting under the available data.

## 10. What We Must Avoid

We should not:

- chase tiny Dice improvements without understanding where they come from;
- claim the model handles shrinkage if loss is not separately validated;
- tune thresholds on test data;
- report only window-level means when patients are few;
- compare methods trained or evaluated at mismatched resolutions without saying so;
- treat SRD gains as proof of real-data performance;
- call this a transformer method unless a transformer is actually needed and justified.

## 11. Concrete Next Experiment

Before implementing a new architecture, the next clean experiment should be:

1. define the exact candidate region for growth prediction;
2. compare growth targets under full outside-mask versus boundary-band-only settings;
3. evaluate distance-to-mask, direct ResUNet, and existing growth-only model within the same candidate region;
4. decide whether a boundary-focused lightweight CNN is justified.

If boundary-band growth captures most true growth, the lightweight CNN can be built around local expansion.

If substantial true growth occurs away from the boundary, the model needs either a wider search field, anatomical/intensity context, or a probabilistic global component.

## 12. Working Hypothesis

The current best hypothesis is:

```text
Short-horizon tumor forecasting is best treated as persistence-preserving residual growth ranking, with shrinkage/loss handled as a separate uncertainty- and treatment-sensitive process.
```

This hypothesis is narrow enough to test and broad enough to become a meaningful research contribution if supported.
