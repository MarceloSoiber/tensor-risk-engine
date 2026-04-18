from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.artifacts import ensure_dir, save_json, save_model_state, save_preprocessing_artifacts
from training.datasets import SequenceFraudDataset, sequence_fraud_collate_fn
from training.metrics import compute_metrics, find_threshold_for_precision
from training.models import SequenceFraudModel
from training.pipeline import run_data_pipeline
from training.sequences import SequenceConfig, build_sequence_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sequence fraud model (GRU/LSTM).")
    parser.add_argument("--dataset", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--output-dir", default="artifacts/sequence", help="Directory for model artifacts.")
    parser.add_argument("--feature-spec", default=None, help="Optional path to feature_spec.json.")
    parser.add_argument("--backbone", default="gru", choices=["gru", "lstm"])
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_pos_weight(y: np.ndarray) -> float:
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives <= 0.0:
        return 1.0
    return max(negatives / positives, 1.0)


def _build_loader(arrays, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SequenceFraudDataset(
        x_num=arrays.x_num,
        x_cat=arrays.x_cat,
        lengths=arrays.lengths,
        y=arrays.y,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=sequence_fraud_collate_fn,
        drop_last=False,
    )


def _predict_scores(model: SequenceFraudModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for x_num, x_cat, lengths, y in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            lengths = lengths.to(device)
            logits = model(x_num, x_cat, lengths)
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            all_scores.append(scores.astype(np.float32))
            all_targets.append(y.detach().cpu().numpy().astype(np.float32))

    if not all_scores:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(all_targets), np.concatenate(all_scores)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    frame, spec, preproc_artifacts = run_data_pipeline(dataset_path=args.dataset, spec_path=args.feature_spec)
    numeric_columns = preproc_artifacts.numeric_columns
    categorical_index_columns = preproc_artifacts.categorical_index_columns

    seq_cfg = SequenceConfig(seq_len=args.seq_len, stride=args.stride)
    train_arrays = build_sequence_arrays(
        frame,
        spec=spec,
        numeric_columns=numeric_columns,
        categorical_index_columns=categorical_index_columns,
        config=seq_cfg,
        split_value="train",
    )
    val_arrays = build_sequence_arrays(
        frame,
        spec=spec,
        numeric_columns=numeric_columns,
        categorical_index_columns=categorical_index_columns,
        config=seq_cfg,
        split_value="val",
    )
    test_arrays = build_sequence_arrays(
        frame,
        spec=spec,
        numeric_columns=numeric_columns,
        categorical_index_columns=categorical_index_columns,
        config=seq_cfg,
        split_value="test",
    )

    train_loader = _build_loader(train_arrays, batch_size=args.batch_size, shuffle=True)
    val_loader = _build_loader(val_arrays, batch_size=args.batch_size, shuffle=False)
    test_loader = _build_loader(test_arrays, batch_size=args.batch_size, shuffle=False)

    categorical_cardinalities = []
    for col in preproc_artifacts.categorical_columns:
        mapping = preproc_artifacts.category_mappings[col]
        cardinality = max(mapping.values(), default=1) + 1
        categorical_cardinalities.append(cardinality)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceFraudModel(
        num_numeric_features=len(numeric_columns),
        categorical_cardinalities=categorical_cardinalities,
        backbone=args.backbone,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        numeric_projection_dim=64,
    ).to(device)

    pos_weight_value = _compute_pos_weight(train_arrays.y)
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_pr_auc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    training_history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for x_num, x_cat, lengths, y in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_num, x_cat, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1

        y_val, val_scores = _predict_scores(model, val_loader, device=device)
        val_metrics = compute_metrics(y_val, val_scores, threshold=0.5) if len(y_val) else None
        val_pr_auc = val_metrics.pr_auc if val_metrics is not None else 0.0
        scheduler.step(val_pr_auc)

        avg_train_loss = running_loss / max(steps, 1)
        training_history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(avg_train_loss),
                "val_pr_auc": float(val_pr_auc),
            }
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    y_val, val_scores = _predict_scores(model, val_loader, device=device)
    tuned_threshold = find_threshold_for_precision(y_val, val_scores, min_precision=0.8) if len(y_val) else 0.5

    val_metrics = compute_metrics(y_val, val_scores, threshold=tuned_threshold) if len(y_val) else None
    y_test, test_scores = _predict_scores(model, test_loader, device=device)
    test_metrics = compute_metrics(y_test, test_scores, threshold=tuned_threshold) if len(y_test) else None

    save_model_state(output_dir, model)
    save_preprocessing_artifacts(output_dir, preproc_artifacts)
    save_json(output_dir / "feature_spec.json", spec.to_dict())
    save_json(
        output_dir / "model_config.json",
        {
            "backbone": args.backbone,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "num_numeric_features": len(numeric_columns),
            "categorical_index_columns": categorical_index_columns,
            "categorical_cardinalities": categorical_cardinalities,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
    )
    save_json(
        output_dir / "training_metadata.json",
        {
            "best_epoch": best_epoch,
            "best_val_pr_auc": best_pr_auc,
            "pos_weight": pos_weight_value,
            "training_history": training_history,
            "split_windows": {
                "train": int(train_arrays.y.shape[0]),
                "val": int(val_arrays.y.shape[0]),
                "test": int(test_arrays.y.shape[0]),
            },
        },
    )
    save_json(output_dir / "thresholds.json", {"decision_threshold": tuned_threshold})
    save_json(output_dir / "metrics_val.json", val_metrics.__dict__ if val_metrics else {})
    save_json(output_dir / "metrics_test.json", test_metrics.__dict__ if test_metrics else {})

    print("Sequence training complete.")
    if val_metrics is not None:
        print(f"Validation PR-AUC: {val_metrics.pr_auc:.4f}")
    if test_metrics is not None:
        print(f"Test PR-AUC: {test_metrics.pr_auc:.4f}")


if __name__ == "__main__":
    main()

