# Evaluation Protocol (v0.1)

## Task

Per-patient longitudinal forecasting:

1. Use first `fit_sessions` sessions as observed history.
2. Forecast tumor mask at future horizon(s) `h`.
3. Evaluate against ground-truth future mask.

Default protocol:

- `fit_sessions = 3`
- `horizons = [1, 2]`
- metric = Dice on tumor masks

## Splits

Use fixed split file:

- `splits/splits.json`

Recommended:

- train split: model training
- val split: hyperparameter tuning
- test split: final reporting

## Baselines

### 1) LOCF

Prediction at future time is the latest observed mask (`last-observation-carried-forward`).

### 2) UNet-mask

Input channels:

- last observed mask
- `delta_days` channel
- current treatment channel
- target treatment channel

Output:

- future mask logits

### 3) UNet-image+mask

Input channels:

- last observed MRI channels
- last observed mask
- `delta_days` channel
- current treatment channel
- target treatment channel

Output:

- future mask logits

## Reporting

For each method, report:

- mean Dice on evaluation split
- std Dice on evaluation split
- number of evaluated samples
- per-sample table (`patient_id`, horizon, dice)

Optional:

- per-tier breakdown using patient ID tier tag (`syn-A-*`, `syn-B-*`, `syn-C-*`)

## Reproducibility

- Use fixed random seed(s), report seed.
- Record config and command used.
- Keep dataset version fixed for all model comparisons.

