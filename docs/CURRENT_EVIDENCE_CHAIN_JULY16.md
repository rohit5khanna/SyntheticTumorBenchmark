# Current Evidence Chain: Regime-Aware Tumor Forecasting

This document consolidates the current research evidence as of July 20, 2026. The goal is not to declare the project complete, but to make the present reasoning auditable: what we have observed, what it supports, what remains weak, and what should be tested next.

## Working Thesis

Short-horizon tumor forecasting should not be treated only as direct future-mask prediction. The current evidence suggests a more structured decomposition:

1. preserve the current tumor where persistence is reliable;
2. identify the boundary-proximal growth front;
3. estimate how much new growth should be allocated;
4. use learned image/context signals to refine the growth-front ranking;
5. separate controlled mechanism-isolation findings from real-data transition-complexity findings;
6. evaluate delayed-hit behavior, because some immediate false positives may be early growth-risk regions.

In short:

`future tumor = persistence + transition-aware growth/loss analysis + growth-front prior + learned image/context correction + growth-budget/gating logic`

The current role of SRD is not to act as a close synthetic surrogate for SAILOR. SRD is retained as a controlled environment for isolating growth, shrinkage, persistence, horizon, and tier effects. SAILOR is treated as the real-data transition-complexity audit, where mixed growth/loss, non-boundary growth, core loss, larger tumor scale, and high-change tails appear together.

## Compact Evidence Table

| Evidence Block | Dataset | Main Result | Interpretation | Strength | Caveat |
|---|---|---|---|---|---|
| Overall Dice benchmark | SRD synthetic data | LOCF `0.747`, UNet image+mask `0.761`, ResUNet image+mask `0.815` mean Dice | Learned models improve average overlap, especially ResUNet | Useful baseline evidence | Dice alone hides growth-regime differences |
| Tier-level Dice behavior | SRD synthetic data | Tier A: LOCF approximately matches ResUNet; Tier B/C: ResUNet gains are larger | Forecasting difficulty and model value are regime-dependent | Supports regime-aware evaluation | Synthetic tiers are controlled regimes, not clinical subtype labels |
| Growth-bin Dice behavior | SRD synthetic data | ResUNet gains over LOCF are strongest in larger future-growth cases; low-growth cases can favor LOCF | Learned models help when there is meaningful change; persistence dominates when little changes | Strong analytical signal | Depends on generated SRD growth distribution |
| Growth-region ranking | SRD synthetic data | UNet/ResUNet and distance priors rank future-growth regions far above random | Future-growth localization is a useful evaluation axis beyond Dice | Strong conceptual shift | Ranking quality does not automatically imply good final Dice |
| Ranking-Dice tradeoff | SRD synthetic data | Some models rank future growth well but over-expand masks, hurting Dice | Forecasting decomposes into ranking and budget estimation | Strong methodological insight | Requires careful reporting so it is not mistaken for model failure alone |
| Budgeted growth-front policy | SRD synthetic data | Validation-selected hybrid policy improves held-out Dice by about `+0.030` over LOCF | Growth-front ranking plus budget selection can improve deployable forecasting | Promising applied direction | Remaining challenge is false activation in no-growth/low-growth cases |
| Growth activity classifier | SRD synthetic data | High aggregate classification metrics, but weak detection of cessation/new activation transitions | Aggregate classifier metrics can be misleading; transition states are the hard problem | Important negative/control result | Needs richer temporal or imaging features |
| Delayed-hit analysis | SRD synthetic data | Some immediate false positives become tumor later; distance-to-boundary explains much of this | Delayed risk is real, but boundary proximity is a major driver | Strong corrective insight | Must separate sample-level and voxel-weighted conclusions |
| Distance-only delayed-hit check | SAILOR real data | Distance prior top immediate-growth-volume: immediate precision `0.311`, eventual precision `0.696`, delayed FP hit rate `0.524`; random eventual precision near `0.027-0.028` | Boundary-proximal false positives often become future tumor in real data | Strong real-data support for growth-front prior | Only 12 test samples from 4 patients |
| Cropped model ranking | SAILOR real data | Cropped evaluation captured all future growth in crop; image+mask hybrid AP `0.335`, distance AP `0.279`, image+mask model AP `0.297` | Image context adds useful ranking signal on top of boundary prior | Strongest real-data support for hybrid framing | Still small sample size |
| Paired SAILOR robustness | SAILOR real data | Image hybrid beats distance AP in `10/12` samples; recall-at-growth-volume improves in `11/12`; bootstrap CI for AP gain over distance `[0.023, 0.088]`; leave-one-patient-out gains remain positive | Hybrid improvement is not driven by one patient | Good stress-test evidence | Not definitive clinical validation |
| LOCF operating range | SAILOR real data | LOCF Dice is much more strongly tied to relative transition burden than to calendar interval alone | One-session forecasting difficulty is not uniform; biological/change horizon matters | Strong analytical evidence | Relationship to growth/loss is partly mathematical for LOCF Dice and must be interpreted carefully |
| Spatial transition taxonomy | SAILOR real data | Mixed growth/loss is common (`39.0%`) and hardest; non-boundary/distant growth remains more common in net-growth transitions across radius sensitivity | Real short-horizon forecasting involves spatial reorganization, not just smooth boundary expansion | Strong data-mining evidence | Labels are heuristic descriptors, not biological classes |
| SRD transition taxonomy | SRD synthetic data | SRD separates clean axes: Tier A persistence-friendly, Tier B loss-heavy, Tier C growth-heavy | SRD is useful for mechanism isolation | Strong within-SRD evidence | SRD should not be framed as realistic real-data mimic |
| SAILOR-vs-SRD domain gap | SAILOR + SRD | SAILOR has more mixed growth/loss, larger tumor scale, higher relative change, more distant growth, and high-change tails; SRD has `0%` mixed growth/loss and no relative absolute change >= `2.0` | The synthetic-real gap is itself informative: controlled SRD and real SAILOR answer different parts of the research question | Strong reframing evidence | Requires careful writing so the gap is not mistaken for a failed synthetic dataset |

## Claims We Can Defend Right Now

1. Short-horizon tumor forecasting is not a uniform task; model value depends on growth activity and regime structure.
2. Dice overlap alone is insufficient because it mixes persistence, growth localization, and growth-budget errors.
3. Boundary proximity is a strong and biologically plausible growth-front prior on both SRD and SAILOR.
4. Learned image/context features can refine growth-front ranking on SAILOR when combined with the distance prior.
5. The strongest current method direction is not a standalone neural forecaster, but a hybrid pipeline that separates persistence, growth-front ranking, and budget/gating.
6. SRD should be described as a controlled mechanism-isolation environment, not as a biological or SAILOR-like surrogate.
7. SAILOR exposes real transition complexity that SRD intentionally does not fully mimic: mixed growth/loss, non-boundary growth, core loss, large tumor scale, and high-change tails.

## Claims We Should Not Overstate

1. We should not claim clinical validation from SAILOR alone; the current SAILOR test set has only 12 samples.
2. We should not claim the learned model replaces the distance prior; distance remains highly competitive for top-candidate recall.
3. We should not claim SRD is a biological or SAILOR-like surrogate; it is a controlled simulation environment for isolating mechanisms.
4. We should not claim growth cessation is solved; stopped/slow-growth cases remain a major open issue.
5. We should not use aggregate classifier accuracy as evidence of a solved regime classifier; subgroup behavior matters more.
6. We should not treat the SRD-SAILOR domain gap as something to immediately erase with another synthetic generator. For the current work, the gap should be used to clarify what SRD can and cannot answer.

## Current Research Interpretation

The work has moved beyond a simple benchmark of LOCF, UNet, and ResUNet. The stronger contribution is now a problem decomposition:

- persistence is often hard to beat for stable/low-change cases;
- growth-front distance is a robust prior for where tumor may appear next;
- learned image context can sharpen the growth-front ranking;
- growth budgeting and activation gating are the bottlenecks that decide whether ranking improvements become better masks;
- SRD provides a clean controlled setting for mechanism isolation;
- SAILOR shows that real transitions frequently combine growth, loss, non-boundary spatial change, and scale effects;
- delayed-hit evaluation helps distinguish harmful false positives from early risk regions.

This framing gives us a clearer path toward a method rather than only an evaluation paper.

## Highest-Value Next Tests

1. **Claim-evidence-status map**

   Convert the current results into a compact claim-evidence table that separates supported, suggestive, weak, and unsupported claims.

2. **SRD/SAILOR transition evidence package**

   Curate the transition taxonomy, radius sensitivity, and domain-gap results into a small set of tables/figures. The goal is not to show SRD matches SAILOR, but to show exactly how controlled synthetic mechanisms differ from real transition complexity.

3. **Budget estimation stress test**

   Test whether simple observable features can estimate a safe growth budget without using future growth. This is currently the main bottleneck.

4. **Hybrid policy with conservative gating**

   Construct a deployable version of `persistence + growth-front prior + image-context correction` that only activates when evidence for growth is strong.

5. **Delayed-risk reporting**

   Continue reporting both immediate precision and eventual precision. This is important because apparent false positives near the growth front may represent future-risk regions.

6. **External data expansion**

   Keep searching for another longitudinal glioma dataset. A second real dataset would materially improve confidence.

## Suggested Figure/Table Package

| Artifact | Purpose |
|---|---|
| SRD tier/growth-regime summary table | Show that the synthetic environment produces different forecast regimes |
| Dice by growth bin | Show persistence vs learned-model behavior depends on future growth |
| Ranking vs Dice tradeoff plot | Show why ranking and mask overlap must be separated |
| Budgeted growth-front policy table | Show deployable improvement and bottleneck |
| SAILOR distance delayed-hit table | Show real-data boundary-prior relevance |
| SAILOR cropped hybrid robustness table | Show image-context correction adds real-data signal |
| SAILOR transition taxonomy and radius-sensitivity tables | Show real-data mixed growth/loss and non-boundary spatial change |
| SRD-vs-SAILOR domain-gap table | Show that SRD is a mechanism-isolation environment rather than a SAILOR surrogate |

## Bottom Line

The current direction is no longer “we ran a few models on a synthetic dataset.” The emerging contribution is a structured analysis of short-horizon tumor forecasting that identifies persistence, growth/loss decomposition, non-boundary spatial change, growth-front ranking, growth-budget estimation, and delayed-risk behavior as separable components. SRD should be used to isolate mechanisms under controlled conditions; SAILOR should be used to expose real transition complexity. The domain gap between them is not a reason to immediately build another synthetic dataset. It is part of the evidence chain.
