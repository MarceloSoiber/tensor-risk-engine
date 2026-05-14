from app.domain.decision import DecisionType
from app.services.risk_service import RiskService


def test_trained_model_threshold_cannot_approve_high_risk_score() -> None:
    decision = RiskService._make_decision(score=0.93, decision_threshold=0.9965)

    assert decision.outcome == DecisionType.REJECT
    assert decision.reasons == ["trained_model_high_risk_guardrail"]


def test_trained_model_threshold_cannot_reject_low_risk_score() -> None:
    decision = RiskService._make_decision(score=0.12, decision_threshold=0.10)

    assert decision.outcome == DecisionType.APPROVE
    assert decision.reasons == ["trained_model_low_risk_guardrail"]


def test_trained_model_threshold_handles_middle_risk_score() -> None:
    decision = RiskService._make_decision(score=0.55, decision_threshold=0.50)

    assert decision.outcome == DecisionType.REJECT
    assert decision.reasons == ["trained_model_above_threshold"]
