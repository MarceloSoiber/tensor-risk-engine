from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.loaders.model_loader import BaselineModelArtifacts, ModelLoader


class RiskInferenceEngine:
    def __init__(self, model_loader: ModelLoader | None = None) -> None:
        self._model_loader = model_loader or ModelLoader()

    def predict(self, features: dict[str, float]) -> float:
        trained_model = self._model_loader.load_current_baseline_model()
        if trained_model is not None:
            return self._predict_with_baseline_model(trained_model, features)

        return self._predict_with_heuristic(features)

    def _predict_with_baseline_model(self, artifacts: BaselineModelArtifacts, features: dict[str, float]) -> float:
        numeric_values = dict(artifacts.numeric_fill_values)
        amount = max(0.0, float(features["raw_amount"]))

        numeric_values["amt"] = amount
        numeric_values["log1p_amt"] = float(np.log1p(amount))
        numeric_values["tx_count_1h"] = max(0.0, float(features["raw_velocity_1h"]))

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
                model_input.append(1.0)
            else:
                model_input.append(0.0)

        score = artifacts.model.predict_proba(np.array([model_input], dtype=np.float32))[0, 1]
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _predict_with_heuristic(features: dict[str, float]) -> float:
        risk_score = (
            0.35 * features["amount"]
            + 0.35 * features["velocity_1h"]
            + 0.25 * features["merchant_risk"]
            + 0.05 * (1.0 - features["device_trust"])
        )
        return max(0.0, min(1.0, risk_score))
