from fastapi import APIRouter

from app.domain.transaction import Transaction
from app.features.feature_builder import FeatureBuilder
from app.ml.inference.risk_engine import RiskInferenceEngine
from app.ml.loaders.model_loader import ModelLoader
from app.repositories.transaction_repository import InMemoryTransactionRepository
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.risk_service import RiskService

router = APIRouter(prefix="/v1", tags=["prediction"])

risk_service = RiskService(
    repository=InMemoryTransactionRepository(),
    feature_builder=FeatureBuilder(),
    inference_engine=RiskInferenceEngine(),
    model_loader=ModelLoader(),
)


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    transaction = Transaction(
        amount=payload.amount,
        velocity_1h=payload.velocity_1h,
        merchant_risk=payload.merchant_risk,
        device_trust=payload.device_trust,
    )

    risk_score, decision, model_version = risk_service.evaluate(transaction)

    return PredictResponse(
        risk_score=risk_score.value,
        decision=decision.outcome.value,
        reasons=decision.reasons,
        model_version=model_version,
    )
