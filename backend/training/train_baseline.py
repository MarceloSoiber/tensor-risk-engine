from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from training.artifacts import ensure_dir, save_json, save_preprocessing_artifacts
from training.metrics import compute_metrics, find_threshold_for_precision
from training.pipeline import run_data_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline logistic regression fraud model.")
    parser.add_argument("--dataset", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--output-dir", default="artifacts/baseline", help="Directory to save baseline artifacts.")
    parser.add_argument(
        "--feature-spec",
        default=None,
        help="Optional path to feature_spec.json. Defaults to training/specs/feature_spec.json.",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split_xy(frame, feature_columns, target_column):
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame[target_column].to_numpy(dtype=np.int64)
    return x, y


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    frame, spec, preproc_artifacts = run_data_pipeline(dataset_path=args.dataset, spec_path=args.feature_spec)
    feature_columns = preproc_artifacts.numeric_columns + preproc_artifacts.categorical_index_columns
    target_column = spec.target_column

    train_df = frame[frame["split"] == "train"]
    val_df = frame[frame["split"] == "val"]
    test_df = frame[frame["split"] == "test"]

    x_train, y_train = _split_xy(train_df, feature_columns, target_column)
    x_val, y_val = _split_xy(val_df, feature_columns, target_column)
    x_test, y_test = _split_xy(test_df, feature_columns, target_column)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_scores = model.predict_proba(x_val)[:, 1]
    tuned_threshold = find_threshold_for_precision(y_val, val_scores, min_precision=0.8)
    val_metrics = compute_metrics(y_val, val_scores, threshold=tuned_threshold)

    test_scores = model.predict_proba(x_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_scores, threshold=tuned_threshold)

    joblib.dump(model, output_dir / "baseline_model.joblib")
    save_preprocessing_artifacts(output_dir, preproc_artifacts)
    save_json(output_dir / "feature_spec.json", spec.to_dict())
    save_json(
        output_dir / "training_metadata.json",
        {
            "model_type": "logistic_regression",
            "feature_columns": feature_columns,
            "split_counts": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
        },
    )
    save_json(output_dir / "thresholds.json", {"decision_threshold": tuned_threshold})
    save_json(output_dir / "metrics_val.json", val_metrics.__dict__)
    save_json(output_dir / "metrics_test.json", test_metrics.__dict__)

    print("Baseline training complete.")
    print(f"Validation PR-AUC: {val_metrics.pr_auc:.4f}")
    print(f"Test PR-AUC: {test_metrics.pr_auc:.4f}")


if __name__ == "__main__":
    main()

