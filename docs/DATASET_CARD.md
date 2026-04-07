# SyntheticTumorBenchmark Dataset Card (v0.1)

## Purpose

SyntheticTumorBenchmark is a controlled longitudinal benchmark for rapid method iteration in tumor forecasting.
It is meant for algorithm development, debugging, and ablation studies before expensive real-data experiments.

## Data Generation

Each patient has longitudinal sessions with:

- `image`: MRI-like channels (`t1`, `t1ce`, `t2`, `flair`)
- `label`: binary tumor mask
- `days`: absolute time from baseline
- `treatment`: binary treatment-on flag

Generation tiers:

- Tier A: procedural geometric growth
- Tier B: isotropic reaction-diffusion growth
- Tier C: anisotropic + heterogeneous reaction-diffusion growth

## File Format

Per patient files:

- `<pid>_image.npy`: `[S, C, H, W, D]`, `float32`
- `<pid>_label.npy`: `[S, 1, H, W, D]`, `uint8`
- `<pid>_days.npy`: `[S]`, `float32`
- `<pid>_treatment.npy`: `[S]`, `float32`
- `<pid>_brainmask.npy`: `[H, W, D]`, `float32`
- `<pid>_concentration.npy` (optional): `[S, H, W, D]`, `float32`

Global files:

- `manifests/manifest.csv`
- `splits/splits.json`
- `dataset_info.json`

## Intended Use

- Early-stage model selection
- Loss/architecture ablations
- Sanity checks for temporal forecasting logic
- Runtime and memory profiling

## Not Intended Use

- Clinical claims
- Direct replacement of real-world evaluation
- Any safety-critical deployment decisions

## Known Gaps

- Imaging is synthetic and may not capture full scanner/domain variability.
- Tumor biology is stylized; realism is limited by chosen PDE/procedural assumptions.
- Treatment modeling is binary and simplified.

