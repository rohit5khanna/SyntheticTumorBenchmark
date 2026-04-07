# Next Steps: Move From Dataset Setup to Model Experiments

Status: dataset framework, evaluation protocol, and baseline scripts are ready.

## Resume Checklist

1. Freeze current benchmark version as `v0.1`.
2. Generate one fixed dataset instance using a locked config and seed.
3. Run `run_all_baselines.py` once on that fixed dataset.
4. Save and treat the resulting baseline summary as the official reference table.
5. Start model comparisons (drift / flow / diffusion) only against this fixed setup.

## Suggested Commands (when you return)

### 1) Tag current state

```bash
git tag -a v0.1 -m "SyntheticTumorBenchmark v0.1 (dataset+protocol+baselines)"
git push origin v0.1
```

### 2) Generate fixed dataset

```bash
python scripts/generate_dataset.py \
  --config configs/benchmark_v1.yaml \
  --output_root "/path/to/fixed_dataset_v1"
```

### 3) Run baseline pack

```bash
python scripts/run_all_baselines.py \
  --dataset_root "/path/to/fixed_dataset_v1" \
  --fit_sessions 3 \
  --horizons 1,2 \
  --train_split train \
  --eval_split test \
  --epochs 12 \
  --batch_size 2 \
  --output_dir outputs/baselines_v0p1
```

### 4) Preserve the reference baseline

Keep this file as reference:

- `outputs/baselines_v0p1/all_baselines_summary.json`

## Notes

- Do not change dataset config/splits mid-comparison.
- Use the same protocol for all candidate models.
- Promote models only if they improve over the frozen baseline table.

