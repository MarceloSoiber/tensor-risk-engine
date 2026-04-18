from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.v1.predict_controller import router as predict_router
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


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Backend online"}


@app.get("/api/health")
def legacy_health() -> dict[str, str]:
    return {"status": "ok"}
