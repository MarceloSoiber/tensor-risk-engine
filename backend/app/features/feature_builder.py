from app.domain.transaction import Transaction


class FeatureBuilder:
    def build(self, transaction: Transaction) -> dict[str, float]:
        # Normalizacao defensiva reduz impacto de valores extremos.
        normalized_amount = min(transaction.amount / 5000.0, 1.0)
        normalized_velocity = min(transaction.velocity_1h / 20.0, 1.0)

        return {
            "amount": max(0.0, normalized_amount),
            "velocity_1h": max(0.0, normalized_velocity),
            "merchant_risk": max(0.0, min(1.0, transaction.merchant_risk)),
            "device_trust": max(0.0, min(1.0, transaction.device_trust)),
        }
