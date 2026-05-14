from dataclasses import dataclass


@dataclass(frozen=True)
class RiskScore:
    value: float
