from dataclasses import dataclass


@dataclass(frozen=True)
class Transaction:
    amount: float
    velocity_1h: int
    merchant_risk: float
    device_trust: float
