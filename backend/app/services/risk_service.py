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

    def evaluate(self, transaction: Transaction, training_job_id: str | None = None) -> tuple[RiskScore, Decision, str]:
        features = self._feature_builder.build(transaction)
        selected_model = self._resolve_selected_model(training_job_id)
        score_value = self._inference_engine.predict(features, model_artifacts=selected_model)
        risk_score = RiskScore(value=score_value)

        decision_threshold = selected_model.decision_threshold if selected_model is not None else self._model_loader.load_decision_threshold()
        decision = self._make_decision(score_value, decision_threshold)
        model_version = selected_model.model_version if selected_model is not None else self._model_loader.load_model_version()
        self._repository.save_analysis(transaction, risk_score, decision, model_version)

        return risk_score, decision, model_version

    def _resolve_selected_model(self, training_job_id: str | None):
        if training_job_id is None:
            return None
        selected_model = self._model_loader.load_baseline_model_by_job_id(training_job_id)
        if selected_model is None:
            raise ValueError("Selected training job is unavailable. Use a succeeded baseline training job.")
        return selected_model

    @staticmethod
    def _make_decision(score: float, decision_threshold: float | None = None) -> Decision:
        if decision_threshold is not None:
            if score >= decision_threshold:
                return Decision(outcome=DecisionType.REJECT, reasons=["trained_model_above_threshold"])
            return Decision(outcome=DecisionType.APPROVE, reasons=["trained_model_below_threshold"])

        if score <= settings.risk_score_approve_max:
            return Decision(outcome=DecisionType.APPROVE, reasons=["low_risk_profile"])

        if score >= settings.risk_score_reject_min:
            return Decision(outcome=DecisionType.REJECT, reasons=["high_risk_profile"])

        return Decision(outcome=DecisionType.REVIEW, reasons=["manual_review_required"])
