from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.core.config import settings

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]
ModelType = Literal["baseline", "sequence"]


class TrainingJobError(RuntimeError):
    """Base exception for training jobs."""


class TrainingJobConflictError(TrainingJobError):
    """Raised when a training job is already running."""


class TrainingJobNotFoundError(TrainingJobError):
    """Raised when job id is unknown."""


class TrainingJobValidationError(TrainingJobError):
    """Raised when user input is invalid."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


@dataclass
class TrainingJobRecord:
    job_id: str
    model_type: ModelType
    dataset_path: str
    artifacts_dir: str
    log_path: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    dataset_size_bytes: int
    dataset_modified_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_epoch: int | None = None
    best_val_pr_auc: float | None = None
    pid: int | None = None
    return_code: int | None = None
    error: str | None = None
    command: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_type": self.model_type,
            "dataset_path": self.dataset_path,
            "artifacts_dir": self.artifacts_dir,
            "log_path": self.log_path,
            "status": self.status,
            "created_at": _to_iso(self.created_at),
            "updated_at": _to_iso(self.updated_at),
            "started_at": _to_iso(self.started_at),
            "finished_at": _to_iso(self.finished_at),
            "progress_epoch": self.progress_epoch,
            "best_val_pr_auc": self.best_val_pr_auc,
            "pid": self.pid,
            "return_code": self.return_code,
            "error": self.error,
            "command": self.command or [],
            "dataset_metadata": {
                "size_bytes": self.dataset_size_bytes,
                "modified_at": _to_iso(self.dataset_modified_at),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingJobRecord":
        metadata = payload.get("dataset_metadata") or {}
        return cls(
            job_id=str(payload["job_id"]),
            model_type=payload["model_type"],
            dataset_path=str(payload["dataset_path"]),
            artifacts_dir=str(payload["artifacts_dir"]),
            log_path=str(payload["log_path"]),
            status=payload["status"],
            created_at=_from_iso(payload.get("created_at")) or _utc_now(),
            updated_at=_from_iso(payload.get("updated_at")) or _utc_now(),
            started_at=_from_iso(payload.get("started_at")),
            finished_at=_from_iso(payload.get("finished_at")),
            progress_epoch=payload.get("progress_epoch"),
            best_val_pr_auc=payload.get("best_val_pr_auc"),
            pid=payload.get("pid"),
            return_code=payload.get("return_code"),
            error=payload.get("error"),
            command=list(payload.get("command") or []),
            dataset_size_bytes=int(metadata.get("size_bytes", 0)),
            dataset_modified_at=_from_iso(metadata.get("modified_at")) or _utc_now(),
        )


class TrainingJobService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, TrainingJobRecord] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._log_files: dict[str, Any] = {}

        self._training_data_root = Path(settings.training_data_root).resolve()
        self._artifacts_root = Path(settings.training_artifacts_root).resolve()
        self._registry_path = Path(settings.training_jobs_registry_path).resolve()
        self._default_dataset = Path(settings.training_default_dataset_path).resolve()
        self._python_bin = settings.training_python_bin
        self._backend_root = Path(__file__).resolve().parents[2]
        self._feature_spec_root = (self._backend_root / "training" / "specs").resolve()

        self._load_registry()

    def start_job(self, request: dict[str, Any]) -> dict[str, Any]:
        model_type = request.get("model_type", "sequence")
        dataset_path = request.get("dataset_path")
        feature_spec_path = request.get("feature_spec_path")
        run_name = request.get("run_name")
        sequence_config = request.get("sequence_config") or {}

        if model_type not in {"baseline", "sequence"}:
            raise TrainingJobValidationError("model_type must be baseline or sequence.")

        with self._lock:
            self._refresh_all_locked()
            self._ensure_no_running_job_locked()

            dataset = self._resolve_dataset_path(dataset_path)
            dataset_stat = dataset.stat()

            job_id = uuid.uuid4().hex
            artifacts_dir = (self._artifacts_root / "jobs" / job_id).resolve()
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_path = artifacts_dir / "train.log"

            resolved_feature_spec_path = self._resolve_feature_spec_path(feature_spec_path)

            command = self._build_command(
                job_id=job_id,
                model_type=model_type,
                dataset_path=dataset,
                artifacts_dir=artifacts_dir,
                feature_spec_path=resolved_feature_spec_path,
                run_name=run_name,
                sequence_config=sequence_config,
            )

            record = TrainingJobRecord(
                job_id=job_id,
                model_type=model_type,
                dataset_path=str(dataset),
                artifacts_dir=str(artifacts_dir),
                log_path=str(log_path),
                status="queued",
                created_at=_utc_now(),
                updated_at=_utc_now(),
                dataset_size_bytes=int(dataset_stat.st_size),
                dataset_modified_at=datetime.fromtimestamp(dataset_stat.st_mtime, tz=timezone.utc),
                command=command,
            )
            self._jobs[job_id] = record
            self._save_registry_locked()

            try:
                log_file = log_path.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=str(self._backend_root),
                    stdout=log_file,
                    stderr=log_file,
                )
            except Exception as exc:  # pragma: no cover
                record.status = "failed"
                record.error = f"Failed to start training process: {exc}"
                record.updated_at = _utc_now()
                record.finished_at = _utc_now()
                self._save_registry_locked()
                raise TrainingJobError("Failed to start training process.") from exc

            self._processes[job_id] = process
            self._log_files[job_id] = log_file

            record.status = "running"
            record.started_at = _utc_now()
            record.updated_at = _utc_now()
            record.pid = process.pid
            self._save_registry_locked()
            return record.to_dict()

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise TrainingJobNotFoundError(f"Job '{job_id}' was not found.")
            self._refresh_job_locked(record)
            self._save_registry_locked()
            return record.to_dict()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_all_locked()
            self._save_registry_locked()
            ordered = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [job.to_dict() for job in ordered]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise TrainingJobNotFoundError(f"Job '{job_id}' was not found.")

            self._refresh_job_locked(record)
            if record.status in {"succeeded", "failed", "canceled"}:
                return record.to_dict()

            process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

            record.status = "canceled"
            record.finished_at = _utc_now()
            record.updated_at = _utc_now()
            record.return_code = process.returncode if process is not None else None
            self._close_log_locked(job_id)
            self._save_registry_locked()
            return record.to_dict()

    def _resolve_dataset_path(self, dataset_path: str | None) -> Path:
        candidate = Path(dataset_path).resolve() if dataset_path else self._default_dataset
        if not candidate.is_absolute():
            candidate = (self._training_data_root / candidate).resolve()
        try:
            candidate.relative_to(self._training_data_root)
        except ValueError as exc:
            raise TrainingJobValidationError(
                "dataset_path must be inside the allowed training data directory."
            ) from exc

        if not candidate.exists():
            raise TrainingJobValidationError("dataset_path does not exist.")
        if not candidate.is_file():
            raise TrainingJobValidationError("dataset_path must point to a file.")
        if candidate.stat().st_size <= 0:
            raise TrainingJobValidationError("dataset file cannot be empty.")
        return candidate

    def _build_command(
        self,
        *,
        job_id: str,
        model_type: ModelType,
        dataset_path: Path,
        artifacts_dir: Path,
        feature_spec_path: Path | None,
        run_name: str | None,
        sequence_config: dict[str, Any],
    ) -> list[str]:
        if model_type == "baseline":
            module = "training.train_baseline"
            command = [
                self._python_bin,
                "-m",
                module,
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(artifacts_dir),
            ]
        else:
            module = "training.train_sequence"
            command = [
                self._python_bin,
                "-m",
                module,
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(artifacts_dir),
            ]
            seq_map: dict[str, tuple[str, Any]] = {
                "backbone": ("--backbone", str),
                "seq_len": ("--seq-len", int),
                "stride": ("--stride", int),
                "batch_size": ("--batch-size", int),
                "epochs": ("--epochs", int),
                "lr": ("--lr", float),
                "patience": ("--patience", int),
                "hidden_size": ("--hidden-size", int),
                "num_layers": ("--num-layers", int),
                "dropout": ("--dropout", float),
            }
            for key, (flag, caster) in seq_map.items():
                if key in sequence_config:
                    command.extend([flag, str(caster(sequence_config[key]))])

        if feature_spec_path:
            command.extend(["--feature-spec", str(feature_spec_path)])
        return command

    def _resolve_feature_spec_path(self, feature_spec_path: str | None) -> Path | None:
        if not feature_spec_path:
            return None

        candidate = Path(feature_spec_path)
        if not candidate.is_absolute():
            candidate = (self._feature_spec_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        try:
            candidate.relative_to(self._feature_spec_root)
        except ValueError as exc:
            raise TrainingJobValidationError(
                "feature_spec_path must be inside backend/training/specs."
            ) from exc

        if not candidate.exists() or not candidate.is_file():
            raise TrainingJobValidationError("feature_spec_path does not exist.")
        return candidate

    def _refresh_all_locked(self) -> None:
        for record in self._jobs.values():
            self._refresh_job_locked(record)

    def _refresh_job_locked(self, record: TrainingJobRecord) -> None:
        if record.status not in {"queued", "running"}:
            return

        process = self._processes.get(record.job_id)
        if process is None:
            if record.pid is not None and self._pid_is_running(record.pid):
                return
            record.status = "failed"
            record.error = record.error or "Training process was not found."
            record.finished_at = record.finished_at or _utc_now()
            record.updated_at = _utc_now()
            self._close_log_locked(record.job_id)
            return

        return_code = process.poll()
        if return_code is None:
            if record.status == "queued":
                record.status = "running"
                record.started_at = record.started_at or _utc_now()
                record.updated_at = _utc_now()
            return

        record.return_code = return_code
        record.finished_at = _utc_now()
        record.updated_at = _utc_now()
        if record.status != "canceled":
            record.status = "succeeded" if return_code == 0 else "failed"
            if return_code != 0 and not record.error:
                record.error = "Training process finished with non-zero exit code."
        self._close_log_locked(record.job_id)

    def _ensure_no_running_job_locked(self) -> None:
        for record in self._jobs.values():
            self._refresh_job_locked(record)
            if record.status == "running":
                raise TrainingJobConflictError("A training job is already running.")

    def _close_log_locked(self, job_id: str) -> None:
        log_file = self._log_files.pop(job_id, None)
        if log_file is not None and not log_file.closed:
            log_file.close()

    def _save_registry_locked(self) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "updated_at": _utc_now().isoformat(),
            "jobs": [record.to_dict() for record in self._jobs.values()],
        }
        tmp_path = self._registry_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self._registry_path)

    def _load_registry(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        jobs_payload = payload.get("jobs", []) if isinstance(payload, dict) else []
        loaded: dict[str, TrainingJobRecord] = {}
        for item in jobs_payload:
            if not isinstance(item, dict):
                continue
            try:
                record = TrainingJobRecord.from_dict(item)
            except Exception:
                continue
            loaded[record.job_id] = record
        self._jobs = loaded

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
