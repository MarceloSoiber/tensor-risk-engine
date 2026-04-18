from app.core.config import settings
from app.domain.decision import Decision, DecisionType
from app.domain.risk_score import RiskScore
from app.domain.transaction import Transaction
from app.features.feature_builder import FeatureBuilder
from app.ml.inference.risk_engine import RiskInferenceEngine
from app.ml.loaders.model_loader import ModelLoader
from app.repositories.transaction_repository import TransactionRepository


class RiskService:
    def __init__(
        self,
        repository: TransactionRepository,
        feature_builder: FeatureBuilder,
        inference_engine: RiskInferenceEngine,
        model_loader: ModelLoader,
    ) -> None:
        self._repository = repository
        self._feature_builder = feature_builder
        self._inference_engine = inference_engine
        self._model_loader = model_loader

    def evaluate(self, transaction: Transaction) -> tuple[RiskScore, Decision, str]:
        self._repository.save(transaction)

        features = self._feature_builder.build(transaction)
        score_value = self._inference_engine.predict(features)
        risk_score = RiskScore(value=score_value)

        decision = self._make_decision(score_value)
        model_version = self._model_loader.load_model_version()

        return risk_score, decision, model_version

    @staticmethod
    def _make_decision(score: float) -> Decision:
        if score <= settings.risk_score_approve_max:
            return Decision(outcome=DecisionType.APPROVE, reasons=["low_risk_profile"])

        if score >= settings.risk_score_reject_min:
            return Decision(outcome=DecisionType.REJECT, reasons=["high_risk_profile"])

        return Decision(outcome=DecisionType.REVIEW, reasons=["manual_review_required"])
