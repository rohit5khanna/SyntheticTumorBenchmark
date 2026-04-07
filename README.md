# SyntheticTumorBenchmark

Standard synthetic longitudinal brain-tumor benchmark generator for fast, repeatable model testing.

## What This Creates

Each synthetic patient has multiple sessions with:

- MRI-like image tensor: `[S, C, H, W, D]` (`float32`)
- Tumor masks: `[S, 1, H, W, D]` (`uint8`)
- Session days: `[S]` (`float32`)
- Treatment flags: `[S]` (`float32`, 0/1)
- Brain mask: `[H, W, D]` (`float32`)
- Optional concentration field: `[S, H, W, D]` (`float32`)

The generator builds three tiers:

- Tier A: procedural geometric growth
- Tier B: isotropic reaction-diffusion growth
- Tier C: anisotropic + heterogeneous reaction-diffusion growth

## Quick Start

1. Install dependencies:

```bash
cd SyntheticTumorBenchmark
pip install -r requirements.txt
```

2. Generate dataset with default config:

```bash
python scripts/generate_dataset.py --config configs/benchmark_v1.yaml
```

3. Override output path (recommended for Drive or scratch):

```bash
python scripts/generate_dataset.py \
  --config configs/benchmark_v1.yaml \
  --output_root "/path/to/save/synthetic_tumor_benchmark_v1"
```

## Output Structure

```text
<output_root>/
  dataset_info.json
  patients/
    syn-A-00001_image.npy
    syn-A-00001_label.npy
    syn-A-00001_days.npy
    syn-A-00001_treatment.npy
    syn-A-00001_brainmask.npy
    syn-A-00001_concentration.npy   (optional)
    ...
  metadata/
    syn-A-00001.json
    ...
  manifests/
    manifest.csv
    manifest.jsonl
  splits/
    splits.json
```

## Config

Main config is `configs/benchmark_v1.yaml`.  
You can tune:

- number of patients per tier
- volume size and modality channels
- session schedule ranges
- simulation parameters (`rho`, `Dw`, treatment effect)
- label threshold and image synthesis noise/bias

## Colab Note

For large runs, set `--output_root` to Google Drive (or a mounted scratch location) so generation does not fill local runtime disk.
