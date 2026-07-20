# Project Map

This document records where the current tumor forecasting work lives and which
folders should be treated as canonical.

## Local Project Root

`/Users/rohitkhanna/Desktop/ORIE Spring 2026/Tumor_Growth_Project/TaDiff`

## Current Work Hub

`/Users/rohitkhanna/Desktop/ORIE Spring 2026/Tumor_Growth_Project/TaDiff/Current_Work`

This hub links the active files into one place without moving the underlying
repositories. This preserves existing paths used by git, Colab notes, and local
scripts.

## Canonical Repository

`/Users/rohitkhanna/Desktop/ORIE Spring 2026/Tumor_Growth_Project/TaDiff/SyntheticTumorBenchmark`

This is the main repo for:

- SRD synthetic regime generation;
- LOCF, U-Net, ResUNet, and related baseline code;
- regime-driver and descriptor analyses;
- result export scripts;
- logs, draft notes, references, and research scaffolding.

## Important Local Documents

- `docs/UPDATED_LOG.md`: running project log and research-state record.
- `docs/RESEARCH_QUESTIONS.md`: core scientific questions.
- `docs/FINAL_DRAFT_MAP.md`: ambitious map of what a solid data-mining contribution must cover.
- `docs/CURRENT_EVIDENCE_CHAIN_JULY16.md`: current claim/evidence/status chain; despite the filename, this is now updated through July 20.
- `docs/References.md`: literature list and notes.
- `docs/EVIDENCE_MAP.md`: evidence scaffold for descriptor and regime claims.
- `docs/DATASET_CARD.md`: dataset-facing notes.
- `docs/Rohit_DMS_paper_draft.docx`: current draft artifact from the earlier submission phase.

## Related Local Repositories

- `TaDiff-Net`: diffusion-model reference and possible future baseline/method source.
- `tumor-biophysical-parameter-inference`: PDE/mechanistic pilot experiments.
- `TumorGrowthToolkit`: growth-solver toolkit used by the PDE pilot.
- `DeepLearningGliomaGrowthModeling`: deep learning glioma growth reference.
- `GliODIL`: inverse/PDE reference code.
- `Reaction-Diffusion-Model-for-Cancer`: reaction-diffusion reference implementation.
- `Drifting_PDE_Tumor`: additional PDE exploration.
- `GMD`: older exploratory/reference material.

## Colab And Drive Locations

The large data/output side has usually lived on Google Drive:

`/content/drive/MyDrive/synthetic_tumor_benchmark`

Important folders:

- `fixed_dataset_v3_lite_generalized`: main SRD dataset used in the latest compact runs.
- `outputs/v3lite_compact_h123_s42`: compact synthetic benchmark outputs.
- `outputs/srd_regime_bundle_v1`: regime analysis outputs.
- `reports`: generated report artifacts.

## Working Rule

Treat `SyntheticTumorBenchmark` as the canonical code and documentation repo.
Treat `Current_Work` as the human-facing entry point for the whole project.

Current framing rule: SRD should be treated as a controlled mechanism-isolation environment, not as a SAILOR-like clinical surrogate. SAILOR should be treated as the real-data transition-complexity audit. The SRD-SAILOR domain gap is part of the evidence chain and should be documented explicitly rather than hidden or patched over with another synthetic generation cycle.
