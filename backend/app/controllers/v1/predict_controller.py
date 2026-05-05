from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.domain.transaction import Transaction
from app.features.feature_builder import FeatureBuilder
from app.ml.inference.risk_engine import RiskInferenceEngine
from app.ml.loaders.model_loader import ModelLoader
from app.repositories.transaction_repository import PostgresTransactionRepository, TransactionPersistenceError
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.risk_service import RiskService

router = APIRouter(prefix="/v1", tags=["prediction"])

model_loader = ModelLoader()

risk_service = RiskService(
    repository=PostgresTransactionRepository(settings.database_url),
    feature_builder=FeatureBuilder(),
    inference_engine=RiskInferenceEngine(model_loader=model_loader),
    model_loader=model_loader,
)


@router.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    transaction = Transaction(
        amount=payload.amount,
        transaction_datetime=payload.transaction_datetime,
        merchant=payload.merchant,
        category=payload.category,
        gender=payload.gender,
        state=payload.state,
        job=payload.job,
        city_population=payload.city_population,
        customer_latitude=payload.customer_latitude,
        customer_longitude=payload.customer_longitude,
        merchant_latitude=payload.merchant_latitude,
        merchant_longitude=payload.merchant_longitude,
        transactions_last_hour=payload.transactions_last_hour,
        transactions_last_24h=payload.transactions_last_24h,
        average_amount_24h=payload.average_amount_24h,
    )

    try:
        risk_score, decision, model_version = risk_service.evaluate(
            transaction,
            training_job_id=payload.training_job_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except TransactionPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction analysis could not be persisted.",
        ) from exc

    return PredictResponse(
        risk_score=risk_score.value,
        decision=decision.outcome.value,
        reasons=decision.reasons,
        model_version=model_version,
    )
