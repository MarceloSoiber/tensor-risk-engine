from dataclasses import dataclass
from enum import StrEnum


class DecisionType(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class Decision:
    outcome: DecisionType
    reasons: list[str]
