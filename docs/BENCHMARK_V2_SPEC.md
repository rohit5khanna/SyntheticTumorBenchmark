# Benchmark v2 Spec

This document turns the v2 blueprint into a practical specification for the current codebase.

It answers four questions:

1. What should Benchmark v2 contain?
2. Which controls already exist in the generator?
3. Which new controls must be added?
4. What should we build first?

This is the bridge between:

- research framing in [BENCHMARK_V2_BLUEPRINT.md](/Users/rohitkhanna/Desktop/ORIE%20Spring%202026/Tumor_Growth_Project/TaDiff/SyntheticTumorBenchmark/docs/BENCHMARK_V2_BLUEPRINT.md)

and

- actual implementation in:
  - [config.py](/Users/rohitkhanna/Desktop/ORIE%20Spring%202026/Tumor_Growth_Project/TaDiff/SyntheticTumorBenchmark/benchmark/config.py)
  - [simulator.py](/Users/rohitkhanna/Desktop/ORIE%20Spring%202026/Tumor_Growth_Project/TaDiff/SyntheticTumorBenchmark/benchmark/simulator.py)
  - [generator.py](/Users/rohitkhanna/Desktop/ORIE%20Spring%202026/Tumor_Growth_Project/TaDiff/SyntheticTumorBenchmark/benchmark/generator.py)
  - [images.py](/Users/rohitkhanna/Desktop/ORIE%20Spring%202026/Tumor_Growth_Project/TaDiff/SyntheticTumorBenchmark/benchmark/images.py)

## Current Generator Controls

The current dataset generator already supports these top-level families of controls:

### Dataset-level

- `dataset.name`
- `dataset.seed`
- `dataset.output_root`
- `dataset.volume_shape`
- `dataset.modalities`
- `dataset.save_concentration`
- `dataset.overwrite`
- `dataset.patient_id_prefix`

### Schedule-level

- `schedule.n_sessions_min`
- `schedule.n_sessions_max`
- `schedule.days_interval_min`
- `schedule.days_interval_max`
- `schedule.treatment_patient_prob`
- `schedule.treatment_start_session_min`

### Simulation-level

- `simulation.steps_per_day`
- `simulation.dt`
- `simulation.rho_range`
- `simulation.dw_range`
- `simulation.treatment_effect_range`
- `simulation.init_foci_min`
- `simulation.init_foci_max`
- `simulation.init_sigma_vox_range`
- `simulation.init_amp_range`

### Labeling

- `labeling.mask_threshold`

### Tier-level

- `tiers.{A,B,C}.enabled`
- `tiers.{A,B,C}.n_patients`
- `tiers.{A,B,C}.description`

### Image synthesis

- `image_synthesis.noise_std`
- `image_synthesis.bias_amp`
- `image_synthesis.bias_smooth_steps`
- `image_synthesis.core_threshold`
- `image_synthesis.edema_threshold`
- `image_synthesis.t1_core_boost`
- `image_synthesis.t1ce_core_boost`
- `image_synthesis.t2_edema_boost`
- `image_synthesis.flair_edema_boost`

## Important Current Limitations

The current code does **not** yet support:

- different schedule distributions per tier
- different simulation parameter ranges per tier
- explicit slow / medium / aggressive growth subgroups
- explicit size-conditioned regimes
- variable treatment response patterns beyond a simple on/off sink
- irregular missing visits
- observation corruption regimes separate from latent tumor evolution
- physically calibrated voxel geometry
- benchmark regimes beyond the coarse `A/B/C` switch in `simulate_patient()`

So Benchmark v2 cannot be built only by changing YAML files.

Some parts can be done immediately by config changes, but the meaningful v2 upgrade needs simulator and config extensions.

## Benchmark v2 Regime Structure

We should keep the readable ladder idea, but define it more sharply.

### Regime A: Controlled Persistence

Purpose:

- persistence-dominated next-step forecasting
- simplest short-horizon baseline regime

Desired properties:

- smooth low-complexity growth
- modest size change between visits
- shorter histories
- cleaner observation conditions

### Regime B: Mechanistic Structured Growth

Purpose:

- reaction-diffusion style evolution with moderate realism

Desired properties:

- isotropic or mildly heterogeneous growth
- moderate longitudinal depth
- more diverse size and growth changes than Regime A

### Regime C: Heterogeneous Frontier

Purpose:

- hardest core forecasting regime
- most important synthetic bridge toward real data

Desired properties:

- anisotropy
- heterogeneous growth
- stronger treatment perturbation
- harder immediate next-step changes
- more varied temporal structure

### Optional Regime D: Robustness Stress Test

Purpose:

- observation noise and missingness stress test

Desired properties:

- noisier images
- irregular or missing scans
- imperfect visible boundaries

This should be optional for now and not block v2.

## Concrete Design Axes

These are the actual axes we should parameterize.

### Axis 1. Temporal Depth

Target behavior:

- more sessions per patient than v1
- longer follow-up
- more interval variability

Recommended target ranges:

| Parameter | v1 current | v2 target |
|---|---:|---:|
| `n_sessions_min` | 4 | 5 or 6 |
| `n_sessions_max` | 7 | 9 to 12 |
| `days_interval_min` | 20 | 20 |
| `days_interval_max` | 60 | 75 or 90 |

Implementation status:

- supported globally now
- not yet supported per regime

Needed change:

- add tier-specific or regime-specific schedule overrides

### Axis 2. Growth Regime Diversity

Target behavior:

- explicit slow, medium, aggressive subgroups
- not one broad undifferentiated growth family

Recommended latent subgroups:

| Subgroup | `rho` behavior | `Dw` behavior | Expected effect |
|---|---|---|---|
| slow | lower | low to medium | strong persistence |
| medium | medium | medium | balanced growth |
| aggressive | higher | medium to high | visible shape expansion |

Implementation status:

- only broad global `rho_range` and `dw_range` exist now

Needed change:

- add subgroup sampling probabilities and subgroup-specific parameter ranges

### Axis 3. Spatial Complexity

Target behavior:

- clearer separation between smooth and irregular spread
- better control of anisotropy and heterogeneity

Current support:

- `Tier B`: isotropic PDE
- `Tier C`: shuffled axis anisotropy + smoothed heterogeneity field

Needed improvement:

- expose anisotropy strength as a config parameter
- expose heterogeneity strength as a config parameter
- optionally add multi-lobe irregularity or localized barriers

Recommended new parameters:

- `simulation.anisotropy_scale_range`
- `simulation.heterogeneity_strength_range`
- `simulation.barrier_prob`
- `simulation.local_spread_boost_prob`

### Axis 4. Tumor Burden / Size

Target behavior:

- intentional coverage of small, medium, and large baseline tumors
- intentional coverage of tiny-change vs visible-change next-step cases

Current support:

- initialization indirectly controlled through:
  - `init_foci_min/max`
  - `init_sigma_vox_range`
  - `init_amp_range`

Problem:

- these are too indirect for benchmark design

Needed change:

- track and target baseline volume bins explicitly
- optionally reject/resample patients outside target bins

Recommended new parameters:

- `simulation.target_baseline_volume_bins`
- `simulation.target_baseline_bin_probs`
- `simulation.resample_max_attempts`

### Axis 5. Treatment Effects

Target behavior:

- treatment should create trajectory breaks that matter

Current support:

- patient-level treatment probability
- treatment start session minimum
- one scalar treatment sink effect

Problem:

- too simple to represent response diversity

Needed change:

- add treatment response families:
  - weak response
  - moderate response
  - strong response
  - rebound or partial stabilization if feasible

Recommended new parameters:

- `schedule.treatment_start_session_max`
- `simulation.treatment_profile_probs`
- `simulation.treatment_effect_profiles`
- `simulation.rebound_prob`

### Axis 6. Observation Realism

Target behavior:

- separate latent growth difficulty from observed image quality

Current support:

- noise and bias field
- modality-specific edema/core contrast

Needed change:

- allow regime-dependent observation corruption
- optionally include small mask threshold jitter or soft boundary uncertainty

Recommended new parameters:

- `image_synthesis.noise_std_range`
- `image_synthesis.bias_amp_range`
- `image_synthesis.boundary_blur_prob`
- `labeling.mask_threshold_jitter`

## What Can Be Done Without Code Changes

The following v2-like improvements can be prototyped immediately using only new configs:

1. increase total patients
2. increase global session count
3. increase global follow-up duration
4. change global growth rate ranges
5. change global initialization scale
6. change global image noise

This is useful for quick pilots, but it is still not true v2 because the regimes are not explicitly controlled.

## What Requires Code Changes

These are the minimum code changes for a real v2.

### Spec Change 1. Regime-specific overrides

Add per-tier overrides for:

- schedule
- simulation
- image synthesis
- labeling

Why:

- right now `A/B/C` differ mostly by rollout mode
- v2 needs the tiers to differ in controlled ways across several axes

### Spec Change 2. Growth subgroup sampling

Add explicit growth subtypes within each regime.

Why:

- realistic forecasting difficulty comes from a mixture distribution, not one averaged parameter range

### Spec Change 3. Size-targeted initialization

Add initialization acceptance criteria or resampling to hit desired baseline size bins.

Why:

- otherwise "small-change persistence traps" are not intentionally represented

### Spec Change 4. Richer treatment profiles

Replace the one-sink treatment mechanism with a small library of response profiles.

Why:

- treatment transitions are one of the best ways to create meaningful longitudinal complexity

### Spec Change 5. Observation corruption layer

Allow image and label perturbation to be varied independently from the latent simulator.

Why:

- this creates a clean benchmark distinction between growth difficulty and observation difficulty

## Proposed Config Extension

The cleanest way to implement v2 is to add a new `regimes` section rather than overloading the current `tiers` block.

Suggested structure:

```yaml
regimes:
  A:
    enabled: true
    n_patients: 120
    schedule:
      n_sessions_min: 5
      n_sessions_max: 8
      days_interval_min: 20
      days_interval_max: 50
    simulation:
      mode: procedural
      growth_subtype_probs:
        slow: 0.60
        medium: 0.35
        aggressive: 0.05
    image_synthesis:
      noise_std: 0.020
  B:
    enabled: true
    n_patients: 120
    schedule:
      n_sessions_min: 6
      n_sessions_max: 10
    simulation:
      mode: reaction_diffusion_isotropic
  C:
    enabled: true
    n_patients: 120
    schedule:
      n_sessions_min: 6
      n_sessions_max: 12
    simulation:
      mode: reaction_diffusion_anisotropic
```

We do not need to implement this exact schema immediately, but this is the right design direction.

## Recommended v2 Build Order

### Stage 1. v2-lite

Goal:

- use current code with only config changes

Actions:

- increase longitudinal depth
- modestly increase patient count
- expand follow-up
- try broader global growth settings

Output:

- pilot dataset for quick audit

### Stage 2. v2-core

Goal:

- add regime-specific overrides

Actions:

- extend config schema
- thread overrides through `simulate_patient()`
- keep backward compatibility with existing configs

Output:

- first real v2 dataset candidate

### Stage 3. v2-growth-aware

Goal:

- add explicit growth subtypes and size bins

Actions:

- implement subgroup sampling
- implement baseline-size targeting
- regenerate pilot and audit again

Output:

- stronger benchmark for persistence-vs-growth analysis

### Stage 4. v2-treatment-aware

Goal:

- make treatment transitions more informative

Actions:

- enrich treatment response profiles
- inspect whether nontrivial breaks appear in trajectories

Output:

- stronger longitudinal realism and forecasting challenge

## Recommended First Concrete Target

The most practical next milestone is:

`Benchmark v2-lite`

Definition:

- keep current `A/B/C` modes
- increase session depth
- increase follow-up
- modestly expand dataset size
- adjust growth settings toward more realistic diversity
- run a fresh audit before any large benchmark sweep

Why this first:

- fastest path to a better dataset
- low engineering risk
- gives us immediate feedback on whether deeper time structure helps

## Success Criteria For v2-lite

Before moving to heavier modeling, v2-lite should improve on v1 in at least these ways:

1. higher session count per patient
2. longer follow-up
3. broader and more interpretable growth-rate distribution
4. cleaner distinction between controlled and heterogeneous regimes
5. better audit story, even if still not fully real-like

## Bottom Line

The benchmark is at a stage where:

- more models alone are not the best next investment,
- but the generator is strong enough to support a meaningful v2 iteration.

The right immediate move is:

- implement a practical v2-lite,
- audit it carefully,
- then decide which deeper v2-core features are worth building next.
