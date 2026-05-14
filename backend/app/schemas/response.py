from pydantic import BaseModel, ConfigDict, Field


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_score: float = Field(..., ge=0.0, le=1.0)
    decision: str
    reasons: list[str]
    model_version: str
