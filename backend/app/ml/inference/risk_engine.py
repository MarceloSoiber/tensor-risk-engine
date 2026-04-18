class RiskInferenceEngine:
    def predict(self, features: dict[str, float]) -> float:
        risk_score = (
            0.35 * features["amount"]
            + 0.35 * features["velocity_1h"]
            + 0.25 * features["merchant_risk"]
            + 0.05 * (1.0 - features["device_trust"])
        )
        return max(0.0, min(1.0, risk_score))
