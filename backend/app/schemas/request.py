from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., ge=0.0, le=1_000_000.0)
    velocity_1h: int = Field(..., ge=0, le=500)
    merchant_risk: float = Field(..., ge=0.0, le=1.0)
    device_trust: float = Field(..., ge=0.0, le=1.0)
