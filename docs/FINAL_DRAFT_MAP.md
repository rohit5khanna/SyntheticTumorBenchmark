# Final Draft Map: Regime- and Transition-Aware Tumor Forecasting

This document is a working map for turning the current project into a solid data-mining contribution. It intentionally does not limit scope to what is easy or deadline-feasible. We will evaluate feasibility as we go. The goal is to identify everything that may be needed for a credible, robust, non-artifact-driven study.

## 1. Central Research Claim

### Target Claim
Short-horizon tumor forecasting is structurally mis-specified when treated as a single full-mask prediction problem. Before model design, "short horizon" itself must be defined more carefully. A one-session forecast is not automatically biologically short-term if the next scan occurs after a long calendar interval, and a short calendar interval can still be difficult if the tumor is rapidly changing.

The project therefore separates three notions of horizon:

1. Session horizon: how many scans ahead are predicted.
2. Calendar horizon: how many days elapse between input and target scans.
3. Biological or change horizon: how much tumor change occurs per unit time.

Longitudinal tumor transitions combine at least three different components:

1. Persistent tumor structure.
2. Spatial expansion/new growth.
3. Apparent loss/shrinkage.

These components have different statistical behavior, different learnability, and different modeling requirements. A data-mining approach should first characterize temporal gap, growth rate, and transition structure, then design/evaluate forecasting methods around those components.

### Current Status
- Supported directionally by SAILOR analyses: LOCF remains strong, direct ResUNet helps growth cases more than shrinkage cases, growth ranking signal exists, and loss prediction is unreliable.
- Supported directionally by SRD regime analyses, but the SRD-to-SAILOR bridge is not yet fully formalized.

### Missing
- A formal claim-evidence map separating what is proven, suggestive, unsupported, or contradicted.
- A concise conceptual figure showing persistence, growth, and apparent loss as separate transition components.
- A clear falsification criterion: what evidence would make us reject this decomposition as useful?
- A formal operational definition of "short-term" using session horizon, calendar horizon, and observed growth/change rate.
- LOCF-limit analysis showing where persistence breaks down as a function of time gap and growth rate.

## 1.1 Definition Of Short-Term Forecasting

### What Must Be Covered
The project should not define short-term forecasting only as "predicting the next scan." In real clinical sequences, next-scan intervals can vary substantially across patients and visits. The same session horizon can therefore correspond to very different biological forecasting tasks.

We need a working definition based on three axes:

- session horizon: next scan versus two or three scans ahead;
- calendar horizon: elapsed days between input and target scans;
- change horizon: observed tumor change normalized by elapsed time.

Useful derived quantities include:

```text
new_growth_rate = new_growth_volume_vox / delta_days
net_volume_rate = (target_volume_vox - input_volume_vox) / delta_days
relative_growth_rate = relative_new_growth / delta_days
absolute_change_rate = (new_growth_volume_vox + loss_volume_vox) / delta_days
```

The working interpretation is:

```text
LOCF is expected to be strong when both calendar interval and biological change rate are small.
LOCF should break down when either elapsed time or growth/change rate becomes large enough.
```

### Current Status
- SAILOR windows include real `delta_days`.
- SRD includes simulated scan days and can be analyzed similarly.
- Prior analyses already compute growth volume, loss volume, net direction, and LOCF Dice.

### Missing
- LOCF Dice stratified jointly by `delta_days` and growth/change rate.
- Evidence that next-session forecasting difficulty is not uniform across calendar horizons.
- A clinically interpretable short/medium/long interval binning rule.
- A biological-change binning rule that does not leak model predictions.
- A figure or table showing how LOCF limits depend on time and growth rate.

## 2. Data Foundations

### 2.1 Synthetic Regime Dataset / SRD

#### What Must Be Covered
- How SRD is generated.
- What parameters are varied.
- What each tier represents mechanistically, without overclaiming clinical subtype realism.
- How treatment is encoded.
- How randomness enters generation.
- How many random seeds/patients/sessions are used.
- Whether observed conclusions survive multiple synthetic realizations.

#### Current Status
- Existing SRD generation code and parameter ranges exist.
- Prior tier-level and case-type analyses exist.
- SRD has been used heavily for initial regime analysis and model comparisons.
- Transition-taxonomy comparison tooling has been added to quantify where SRD covers or misses SAILOR-like transition structure.

#### Missing
- Multiple SRD realizations beyond current fixed dataset.
- Explicit sensitivity analysis over generation seed.
- Clear distinction between generator-controlled tier labels and discovered transition categories.
- A table decomposing SRD transitions into persistence/growth/loss components.
- Evidence that key SRD findings are not artifacts of one synthetic realization.
- If SRD misses important SAILOR transition regions, a principled decision is needed: either extend SRD or explicitly frame it as a controlled mechanism-isolation environment.

### 2.2 SAILOR Real-Data Processing

#### What Must Be Covered
- Patient/session count after filtering.
- Input length and horizon construction.
- Sliding-window generation.
- Patient-level train/validation/test split.
- Treatment encoding, if used.
- MRI modality handling.
- Resolution/downsampling choices.
- Segmentation/registration limitations.

#### Current Status
- SAILOR input-length-3/horizon-1 manifest exists.
- Patient-level split v2 exists and is more balanced than the first split.
- Local copy improved I/O.
- Corrected image handling now uses 8 channels: 4 MRI modalities + mask + time/treatment channels.

#### Missing
- Formal data card for SAILOR experiments.
- Clear comparison between full-resolution and stride-2 evaluation.
- Justification for stride-2 experiments and whether conclusions survive full resolution or ROI/crop-based resolution.
- Split robustness beyond one patient split.
- Handling of patients with different session counts.

### 2.3 Additional Real Datasets

#### What Must Be Covered
- Whether another open longitudinal glioma/tumor dataset can validate the transition-decomposition claim.
- Whether LUMIERE, BraTS longitudinal variants, or datasets used in prior growth forecasting papers are accessible.
- Whether available datasets include treatment, longitudinal masks, and enough sessions.

#### Current Status
- Possible dataset document exists.
- SAILOR is the only real dataset tested so far.

#### Missing
- Practical access decision for at least one additional dataset.
- Minimal transition taxonomy replication on another real dataset if possible.
- If no second dataset is used, a strong limitation statement and stronger SAILOR robustness are needed.

## 3. Transition Taxonomy

### 3.1 Core Transition Components

#### What Must Be Covered
For each transition, quantify:

- input tumor volume,
- target tumor volume,
- persistent core volume,
- true new-growth volume,
- apparent loss volume,
- net volume change,
- relative growth,
- relative loss,
- persistent-core fraction,
- growth-to-input ratio,
- loss-to-input ratio.

#### Current Status
- Many of these quantities are computed in scripts across SAILOR analyses.
- Growth/loss decomposition exists in residual and transition-error scripts.
- Unified transition-taxonomy tooling has been added to produce per-transition persistence, growth, loss, rate, type, and patient-trajectory tables for SRD and SAILOR.

#### Missing
- Run and review unified transition taxonomy tables for both SRD and SAILOR.
- Figures showing distribution of persistence/growth/loss components.
- Separate summary by horizon, patient, tier/regime, treatment status, and time interval.

### 3.2 Transition Types

#### Candidate Types
- Persistence-dominant transitions.
- Net-growth transitions.
- Net-shrinkage transitions.
- Mixed growth/loss transitions.
- Boundary-only change.
- Distant/offshoot growth.
- High-uncertainty low-volume transitions.

#### Current Status
- Net-growth/net-shrinkage categories are used.
- SRD case types exist from LOCF versus ResUNet behavior.
- Transition-taxonomy tooling now assigns descriptive categories such as persistence-dominant, growth-dominant, loss-dominant, mixed growth/loss, boundary-growth-dominant, and distant-growth-present.
- Radius-sensitivity tooling has been added to test whether boundary/distant labels depend on one arbitrary voxel radius.

#### Missing
- Review boundary versus distant growth analysis after recomputed-mask and radius-sensitivity runs.
- Mixed spatial-change category independent of net volume change.
- Patient-level transition trajectories.
- Whether transition types are stable within patient or change over time.

### 3.3 Randomness / Artifact Risk

#### Required Robustness
- Transition taxonomy should be stable across SRD generation seeds.
- SAILOR conclusions should be stable across patient splits where possible.
- Subgroup claims should report patient counts, not only window counts.

#### Current Status
- Patient bootstrap used for selected SAILOR method results.
- Seed repeat run for growth-only model exists.
- Patient-bootstrap tooling has been added for the conservative persistence-breakdown predictability audit.

#### Missing
- Repeated patient split analysis.
- Multi-seed aggregation tables.
- Synthetic generation repeat analysis.
- A formal rule that no claim is made from fewer than a minimum number of patients unless marked exploratory.

## 4. Predictability Analysis

### 4.1 Growth Predictability

#### What Must Be Covered
- Is new growth spatially predictable from prior scans and MRI?
- Is growth better viewed as ranking than binary segmentation?
- How does learned growth ranking compare with simple baselines?

#### Current Status
- Residual probability diagnostic found strong growth signal: growth AP around 0.63 and AUC around 0.98 on SAILOR stride-2 outputs.
- Growth-only model uses budgeted top-k additions.

#### Missing
- Comparison against distance-to-current-mask baseline on SAILOR under same windows/resolution.
- Growth ranking performance by net-growth/net-shrinkage, patient, treatment, interval, and growth volume.
- Calibration analysis: are probabilities meaningful or only rankings?
- Full-resolution or ROI-resolution confirmation.

### 4.2 Loss/Shrinkage Predictability

#### What Must Be Covered
- Is apparent loss predictable from available inputs?
- Is loss biologically meaningful, annotation noise, registration artifact, treatment response, or boundary uncertainty?
- Should loss be modeled directly, conservatively, or as uncertainty?

#### Current Status
- Residual loss head performed poorly and appeared inverted: loss AUC around 0.32; stable core had higher loss probability than true loss.
- Direct full-mask models harmed shrinkage cases.

#### Missing
- Boundary-versus-core loss analysis.
- Treatment association with loss.
- Time-interval association with loss.
- Patient-specific loss patterns.
- Image-intensity change analysis in loss regions.
- Registration/segmentation uncertainty discussion.
- A principled decision on whether loss should be predicted, suppressed, or represented probabilistically.

### 4.3 Net-Direction Predictability

#### What Must Be Covered
- Can we predict whether next transition is net-growth, net-shrinkage, or mixed using only available input history?
- Is transition-state prediction stable enough to guide model selection?

#### Current Status
- Logistic direction gate had weak/non-random signal but collapsed under validation-selected Dice objective.
- Direction classifier AUC was modest.
- Conservative persistence-breakdown prediction now avoids target-derived treatment variables and can be evaluated with patient-bootstrap uncertainty.

#### Missing
- A scientifically evaluated transition-state classifier independent of model-selection/Dice gating.
- Feature importance and calibration for transition-state prediction.
- Patient-level cross-validation and bootstrap summaries for transition-state prediction.
- Comparison of input-history features, image features, treatment features, and prior volume trend.

## 5. Model Behavior Audit

### 5.1 Baselines To Include

#### Required Baselines
- LOCF.
- Direct U-Net / ResUNet full-mask prediction.
- Residual growth/loss model.
- Growth-only outside-input model.
- Distance-to-mask growth ranking baseline.
- Possibly simple volume extrapolation or previous-growth extrapolation.

#### Current Status
- LOCF and direct ResUNet have SAILOR results.
- Residual growth/loss and growth-only models have SAILOR stride-2 results.
- SRD has broader baseline set.

#### Missing
- Same-resolution, same-split, same-sample comparison table across all SAILOR methods.
- Direct ResUNet stride-2 reevaluation or growth-only full-resolution/ROI evaluation for fair comparison.
- Distance baseline on SAILOR.
- Confidence intervals by patient.

### 5.2 Model Behavior Questions

#### What Must Be Answered
- Does a model improve by adding growth, preserving core, deleting tumor, or changing volume?
- Where does LOCF win?
- Where does direct ResUNet win?
- Where does growth-only help?
- Where do neural methods fail?

#### Current Status
- Transition error decomposition showed direct ResUNet preserves stable core but underpredicts growth/loss and hurts shrinkage through false growth and poor loss handling.
- Growth-only model improves net-growth modestly while mostly preserving persistence.

#### Missing
- Unified model-behavior audit across methods.
- Error decomposition for growth-only model versus direct ResUNet.
- Per-patient case studies and failure examples.
- Visualization of probability maps and selected growth regions.

## 6. Methodological Direction

### 6.1 Current Best Method Hypothesis

#### Hypothesis
A persistence-preserving, growth-only, budgeted ranking model is better aligned with short-horizon tumor forecasting than unconstrained full-mask prediction or symmetric growth/loss residual prediction.

#### Current Status
- Dedicated growth-only model with seeds 42 and 123 selected the same 5% growth budget.
- Validation gains are replicated.
- Test gains are positive but small and uncertain.
- Net-growth benefit is the strongest signal.

#### Missing
- More seed repeats or multi-seed summary.
- Repeated patient splits.
- Fair comparison to direct ResUNet at the same resolution.
- SAILOR full-resolution or ROI-resolution confirmation.
- SRD test of the same method.

### 6.2 Possible Next Method Improvements

#### Candidate Improvements
- Candidate-region restriction near tumor boundary plus allowance for offshoots.
- Distance-aware growth prior.
- Budget predictor instead of fixed validation-selected budget.
- Transition-state-conditioned budget.
- Uncertainty output for shrinkage/loss.
- Mixture-of-experts later: persistence expert, growth expert, loss/uncertainty expert.
- Ranking loss instead of BCE/Dice.
- Pairwise or contrastive growth localization objective.
- Patient-specific adaptation or calibration.

#### Current Status
- Only fixed budget top-k selection has been tested.

#### Missing
- Conceptual selection of which improvements are principled versus over-engineered.
- Ablations showing each addition is necessary.
- Guardrails against overfitting SAILOR.

## 7. Evaluation Metrics

### 7.1 Standard Metrics

#### Required
- Dice overall.
- Dice by transition type.
- Dice by patient.
- Win rate versus LOCF.
- Patient-level bootstrap confidence intervals.

#### Current Status
- Many of these exist for selected experiments.

#### Missing
- Unified reporting template.
- Multi-seed/multi-split aggregation.

### 7.2 Component Metrics

#### Required
- Persistent-core recall.
- Growth precision/recall.
- Loss precision/recall.
- Growth AP/AUC.
- Loss AP/AUC.
- Recall at fixed growth budget.
- Distance-to-mask ranking comparison.

#### Current Status
- Some component metrics exist.
- Growth/loss probability AP/AUC exists for residual model.

#### Missing
- Same metrics for dedicated growth-only model.
- Component metrics for direct ResUNet and distance baseline.
- Visual plots of component-level performance.

### 7.3 Clinical/Operational Metrics

#### Potentially Useful
- Conservative false-growth penalty.
- False-negative growth penalty.
- Boundary versus distant missed growth.
- Patient-level risk ranking.
- Whether method identifies likely future expansion regions even if full Dice changes little.

#### Current Status
- Not fully developed.

#### Missing
- Decide which clinical/operational metric is aligned with the paper.
- Avoid pretending Dice alone captures clinical utility.

## 8. Robustness and Anti-Artifact Requirements

### 8.1 Synthetic Robustness

#### Required
- Multiple SRD random seeds.
- Multiple model seeds.
- Stable transition taxonomy across synthetic generations.
- Stable method ranking or clear variability statement.

#### Current Status
- Limited multi-seed model checks exist for earlier SRD baselines.

#### Missing
- SRD generation repeat experiments.
- Robust synthetic evidence map.

### 8.2 SAILOR Robustness

#### Required
- Multiple model seeds.
- Patient-level bootstrap.
- Repeated patient splits if feasible.
- Same-resolution fair comparisons.
- Sensitivity to input length and horizon.

#### Current Status
- Growth-only seeds 42 and 123 exist.
- Patient bootstrap exists.
- One patient split v2 exists.

#### Missing
- Seed aggregation.
- Additional seed(s), e.g., 777.
- Repeated split evaluation.
- Input length sensitivity, e.g., 2 versus 3 prior scans.
- Horizon sensitivity beyond h=1 if feasible.

### 8.3 Random Sample Dependency

#### Risk
A large portion of the project currently relies on one SRD realization, one SAILOR patient split, and one or two model seeds. This is not sufficient for airtight claims.

#### Required Policy
No central claim should rely on a single random realization. Each claim must be labeled as:

- robust across seeds/splits/datasets,
- replicated but limited,
- exploratory,
- unsupported.

#### Missing
- Claim-status table.
- Reproducibility checklist.
- Explicit random-state inventory.

## 9. Figures and Tables Needed

### High-Priority Figures
1. Conceptual decomposition figure: persistence, growth, apparent loss.
2. Transition taxonomy distributions for SRD and SAILOR.
3. Model behavior by transition type.
4. Growth versus loss ranking performance.
5. Growth-only budget curve showing thresholding unsafe but top-k budgeting safe.
6. Patient-level uncertainty plot for key method comparisons.

### High-Priority Tables
1. Dataset summary and transition statistics.
2. SRD tier/generator parameter table.
3. SAILOR split summary.
4. Baseline/model comparison table by transition type.
5. Component metric table: persistence/growth/loss.
6. Robustness table: seeds, splits, datasets, status.
7. Claim-evidence-status table.

### Current Status
- Many raw CSVs and figures exist from earlier work.
- Not yet organized into a coherent evidence set.

### Missing
- Final curated figure/table manifest.
- Consistent naming and captions.
- Avoid redundant or weak figures.

## 10. Literature Integration

### Areas To Cover
- Longitudinal tumor forecasting.
- Reaction-diffusion / biophysical tumor modeling.
- Deep generative tumor forecasting.
- Growth ranking / forward ranking evaluation.
- Persistence/LOCF baselines in medical longitudinal forecasting.
- Segmentation uncertainty, registration noise, and response assessment.
- Mixture-of-experts or conditional computation only if method evolves there.

### Current Status
- Some references collected.
- User read papers on deep probabilistic glioma growth and forward ranking.

### Missing
- Literature matrix: paper, dataset, task, method, evaluation, limitation, relevance.
- Clear statement of what is new relative to prior work.
- Avoid overclaiming that regime-aware evaluation is absent if literature already touches it.

## 11. Paper Structure Candidate

### Possible Final Structure
1. Introduction: why short-horizon forecasting needs transition-aware analysis.
2. Related work: tumor forecasting, biophysical simulation, deep forecasting, ranking/evaluation.
3. Data and transition decomposition framework.
4. Synthetic regime dataset and SAILOR setup.
5. Transition taxonomy and predictability analysis.
6. Model behavior audit.
7. Growth-only persistence-preserving prototype.
8. Robustness and limitations.
9. Discussion: implications for data mining and future forecasting models.

### Current Status
- Earlier draft exists but should not dictate final paper structure.

### Missing
- Rebuild structure around evidence, not prior DMS deadline draft.

## 12. Claim-Evidence-Status Template

| Claim | Current Evidence | Current Status | Missing Evidence | Risk If Unfixed |
|---|---|---|---|---|
| Short-horizon forecasting mixes persistence, growth, and loss | SAILOR/SRD decomposition scripts partially available | Suggestive | Unified taxonomy across SRD/SAILOR | Claim may sound conceptual but not proven |
| Growth is learnable as ranking | Residual probability diagnostic: growth AUC high | Strong but narrow | Distance baseline, direct model comparison, dedicated model AP/AUC | Could be artifact of one checkpoint/resolution |
| Loss is not symmetric with growth | Residual loss AUC poor/inverted | Suggestive | Boundary/treatment/registration analysis | Could be just bad model/loss function |
| Growth-only model is better aligned than full residual | Seeds 42/123 positive trend | Replicated but limited | More seeds/splits/fair direct comparison | Could be small random effect |
| Direct ResUNet helps growth but hurts shrinkage | SAILOR split v2 direct result | Suggestive | Same-resolution comparison, bootstrap, seed repeat | Could be split/checkpoint artifact |
| SRD regimes connect to real transition behavior | Some bridge analyses exist | Weak | Formal SRD-SAILOR transition comparison | Synthetic story may feel disconnected |

## 13. Immediate Next Audit Tasks

1. Build unified transition taxonomy table for SRD and SAILOR.
2. Build claim-evidence-status table from all existing results.
3. Create random-state inventory: dataset seeds, split seeds, model seeds, stochastic scripts.
4. Build same-resolution SAILOR method comparison plan.
5. Run/prepare distance-to-mask growth ranking baseline on SAILOR.
6. Analyze loss spatially: boundary versus core, treatment, interval.
7. Aggregate growth-only seeds 42 and 123 into one table.
8. Decide whether to run seed 777 or repeated patient split next.
9. Curate figure/table manifest.
10. Decide which claims are strong enough for a workshop submission and which remain future work.

## 14. Quality Bar

The final work should not be submitted merely because it can be made to sound plausible. It should satisfy the following:

- Claims are supported by explicit evidence.
- Randomness and sample-size limitations are visible.
- No main conclusion depends on a single random realization.
- Small Dice gains are not oversold.
- Negative results are incorporated honestly.
- Data-mining insight is central, not decorative.
- The method prototype follows from the analysis rather than appearing as threshold engineering.
- The paper remains valuable even if the proposed method only gives modest performance gains.
