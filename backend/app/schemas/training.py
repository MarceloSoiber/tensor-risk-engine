from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]
ModelType = Literal["baseline", "sequence"]


class SequenceTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: Literal["gru", "lstm"] = "gru"
    seq_len: int = Field(default=30, ge=5, le=512)
    stride: int = Field(default=1, ge=1, le=64)
    batch_size: int = Field(default=256, ge=8, le=4096)
    epochs: int = Field(default=20, ge=1, le=300)
    lr: float = Field(default=1e-3, gt=0.0, le=1.0)
    patience: int = Field(default=5, ge=1, le=50)
    hidden_size: int = Field(default=128, ge=16, le=2048)
    num_layers: int = Field(default=2, ge=1, le=8)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)


class TrainingJobStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: ModelType = "sequence"
    dataset_path: str | None = None
    feature_spec_path: str | None = None
    run_name: str | None = Field(default=None, min_length=1, max_length=120)
    sequence_config: SequenceTrainingConfig = Field(default_factory=SequenceTrainingConfig)


class DatasetMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_bytes: int = Field(..., ge=0)
    modified_at: datetime


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    model_type: ModelType
    dataset_path: str
    artifacts_dir: str
    log_path: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_epoch: int | None = None
    best_val_pr_auc: float | None = None
    error: str | None = None
    dataset_metadata: DatasetMetadataResponse


class TrainingJobListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[TrainingJobResponse]

