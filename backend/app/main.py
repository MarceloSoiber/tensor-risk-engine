from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.v1.ai_analysis_controller import router as ai_analysis_router
from app.controllers.v1.dashboard_controller import router as dashboard_router
from app.controllers.v1.predict_controller import router as predict_router
from app.controllers.v1.training_controller import router as training_router
from app.controllers.v1.transactions_controller import router as transactions_router
from app.core.config import settings

app = FastAPI(title=settings.app_name, version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(predict_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(transactions_router, prefix="/api")
app.include_router(ai_analysis_router, prefix="/api")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Backend online"}


@app.get("/api/health")
async def legacy_health() -> dict[str, str]:
    return {"status": "ok"}
