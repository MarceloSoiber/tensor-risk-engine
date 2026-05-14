from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from app.ml.loaders.model_loader import BaselineModelArtifacts, ModelArtifacts, ModelLoader, SequenceModelArtifacts

OOV_INDEX = 1


class RiskInferenceEngine:
    def __init__(self, model_loader: ModelLoader | None = None) -> None:
        self._model_loader = model_loader or ModelLoader()

    def predict(self, features: dict[str, float | str], model_artifacts: ModelArtifacts | None = None) -> float:
        trained_model = model_artifacts or self._model_loader.load_current_baseline_model()
        if isinstance(trained_model, BaselineModelArtifacts):
            return self._predict_with_baseline_model(trained_model, features)
        if isinstance(trained_model, SequenceModelArtifacts):
            return self._predict_with_sequence_model(trained_model, features)

        return self._predict_with_heuristic(features)

    def _predict_with_baseline_model(self, artifacts: BaselineModelArtifacts, features: dict[str, float | str]) -> float:
        numeric_values = dict(artifacts.numeric_fill_values)
        for column in artifacts.numeric_columns:
            if column in features:
                numeric_values[column] = float(features[column])

        numeric_frame = pd.DataFrame(
            [[numeric_values.get(column, 0.0) for column in artifacts.numeric_columns]],
            columns=artifacts.numeric_columns,
        )
        scaled_numeric_matrix = artifacts.scaler.transform(numeric_frame).astype(np.float32)
        scaled_numeric = dict(zip(artifacts.numeric_columns, scaled_numeric_matrix[0], strict=True))

        model_input = []
        for column in artifacts.feature_columns:
            if column in scaled_numeric:
                model_input.append(float(scaled_numeric[column]))
            elif column in artifacts.categorical_index_columns:
                model_input.append(float(self._category_index_for_column(column, artifacts, features)))
            else:
                model_input.append(0.0)

        score = artifacts.model.predict_proba(np.array([model_input], dtype=np.float32))[0, 1]
        return max(0.0, min(1.0, float(score)))

    def _predict_with_sequence_model(self, artifacts: SequenceModelArtifacts, features: dict[str, float | str]) -> float:
        numeric_values = dict(artifacts.numeric_fill_values)
        for column in artifacts.numeric_columns:
            if column in features:
                numeric_values[column] = float(features[column])

        numeric_frame = pd.DataFrame(
            [[numeric_values.get(column, 0.0) for column in artifacts.numeric_columns]],
            columns=artifacts.numeric_columns,
        )
        scaled_numeric_matrix = artifacts.scaler.transform(numeric_frame).astype(np.float32)

        categorical_values = [
            self._category_index_for_sequence_column(index_column, artifacts, features)
            for index_column in artifacts.categorical_index_columns
        ]

        seq_len = max(int(artifacts.seq_len), 1)
        x_num = np.zeros((1, seq_len, len(artifacts.numeric_columns)), dtype=np.float32)
        x_cat = np.zeros((1, seq_len, len(artifacts.categorical_index_columns)), dtype=np.int64)
        x_num[0, -1, :] = scaled_numeric_matrix[0]
        x_cat[0, -1, :] = np.asarray(categorical_values, dtype=np.int64)

        with torch.no_grad():
            logits = artifacts.model(
                torch.from_numpy(x_num),
                torch.from_numpy(x_cat),
                torch.tensor([1], dtype=torch.long),
            )
            score = torch.sigmoid(logits).detach().cpu().numpy()[0]
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _category_index_for_column(
        index_column: str,
        artifacts: BaselineModelArtifacts,
        features: dict[str, float | str],
    ) -> int:
        source_column = index_column.removesuffix("_idx")
        raw_index = features.get(index_column)
        if raw_index is not None:
            return int(float(raw_index))

        mapping = artifacts.category_mappings.get(source_column, {})
        raw_value = str(features.get(source_column, "__MISSING__"))
        return mapping.get(raw_value, OOV_INDEX)

    @staticmethod
    def _category_index_for_sequence_column(
        index_column: str,
        artifacts: SequenceModelArtifacts,
        features: dict[str, float | str],
    ) -> int:
        source_column = index_column.removesuffix("_idx")
        mapping = artifacts.category_mappings.get(source_column, {})
        raw_value = str(features.get(source_column, "__MISSING__"))
        return mapping.get(raw_value, OOV_INDEX)

    @staticmethod
    def _predict_with_heuristic(features: dict[str, float | str]) -> float:
        amount_score = min(max(float(features["log1p_amt"]) / np.log1p(5000.0), 0.0), 1.0)
        velocity_1h_score = min(max(float(features["tx_count_1h"]) / 20.0, 0.0), 1.0)
        velocity_24h_score = min(max(float(features["tx_count_24h"]) / 100.0, 0.0), 1.0)
        distance_score = min(max(float(features["geo_distance_km"]) / 500.0, 0.0), 1.0)
        amount_delta_score = min(max(float(features["amount_delta_ratio_24h"]) / 5.0, 0.0), 1.0)
        night_score = float(features["is_night"])

        risk_score = (
            0.30 * amount_score
            + 0.25 * velocity_1h_score
            + 0.15 * velocity_24h_score
            + 0.15 * distance_score
            + 0.10 * amount_delta_score
            + 0.05 * night_score
        )
        return max(0.0, min(1.0, risk_score))
