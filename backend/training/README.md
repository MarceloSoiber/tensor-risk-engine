# Fraud Training Pipeline

This package contains the offline fraud training pipeline focused on:

- Causal feature engineering
- Data normalization and categorical encoding
- Temporal split without leakage
- Baseline tabular training
- Sequence training with GRU/LSTM in PyTorch
- Artifact persistence for reproducible inference

## Inputs

Expected dataset columns include:

- `cc_num`, `trans_date_trans_time`, `is_fraud`
- `amt`, `merchant`, `category`
- `lat`, `long`, `merch_lat`, `merch_long`
- Optional profile columns such as `job`, `state`, `gender`, `city_pop`

## Run Baseline

```bash
python -m training.train_baseline \
  --dataset backend/training/data/fraud.csv \
  --output-dir backend/training/artifacts/baseline
```

## Run Sequence Model

```bash
python -m training.train_sequence \
  --dataset backend/training/data/fraud.csv \
  --output-dir backend/training/artifacts/sequence \
  --backbone gru \
  --seq-len 30
```

## Artifacts

Each training run saves:

- `scaler.joblib`
- `category_mappings.json`
- `preprocessing_metadata.json`
- `feature_spec.json`
- `model.pt` (sequence model) or `baseline_model.joblib`
- `thresholds.json`
- `metrics_val.json`
- `metrics_test.json`
- `training_metadata.json`

