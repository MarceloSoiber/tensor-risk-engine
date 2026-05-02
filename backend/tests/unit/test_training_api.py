from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from app.controllers.v1 import training_controller
from app.main import app
from app.schemas.training import TrainingJobStartRequest
from app.services import training_job_service
from app.services.training_job_service import TrainingJobService


class FakeProcess:
    _pid_counter = 90000

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.pid = FakeProcess._pid_counter
        FakeProcess._pid_counter += 1
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


@pytest.fixture()
def patched_training_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TrainingJobService, list[FakeProcess]]:
    data_dir = tmp_path / "training-data"
    data_dir.mkdir()
    (data_dir / "fraudTrain.csv").write_text("transaction_id,amount,is_fraud\n1,25.50,0\n", encoding="utf-8")
    (data_dir / "custom.csv").write_text("transaction_id,amount,is_fraud\n2,900.00,1\n", encoding="utf-8")

    processes: list[FakeProcess] = []

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:  # noqa: ARG001
        process = FakeProcess(list(args[0]))
        processes.append(process)
        return process

    service = TrainingJobService(process_factory=fake_popen)
    service._jobs = {}
    monkeypatch.setattr(service, "_training_data_root", data_dir.resolve())
    monkeypatch.setattr(service, "_default_dataset", (data_dir / "fraudTrain.csv").resolve())
    monkeypatch.setattr(service, "_artifacts_root", (tmp_path / "artifacts").resolve())
    monkeypatch.setattr(service, "_registry_path", (tmp_path / "jobs_registry.json").resolve())
    monkeypatch.setattr(training_controller, "training_job_service", service)
    return service, processes


def test_training_job_start_route_is_registered() -> None:
    routes = {
        (route.path, tuple(sorted(route.methods or [])))
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/training/jobs"
    }

    assert ("/api/v1/training/jobs", ("POST",)) in routes


def test_start_training_job_uses_default_dataset(
    patched_training_service: tuple[TrainingJobService, list[FakeProcess]],
) -> None:
    service, processes = patched_training_service

    response = training_controller.start_training_job(
        TrainingJobStartRequest(model_type="baseline", run_name="  monthly baseline refresh  "),
    )
    body = response.model_dump(mode="json")

    print(f"Body: {body}")

    fetched_job = training_controller.get_training_job(body["job_id"])
    fetched_body = fetched_job.model_dump(mode="json")

    print(f"Fetched job: {fetched_job}")

    assert body["status"] == "running"
    assert body["run_name"] == "monthly baseline refresh"
    assert body["model_type"] == "baseline"
    assert body["dataset_path"].endswith("fraudTrain.csv")
    assert body["dataset_metadata"]["size_bytes"] > 0
    assert service._registry_path.exists()
    assert len(processes) == 1
    assert processes[0].command[:3] == ["python", "-m", "training.train_baseline"]
    assert fetched_body["run_name"] == "monthly baseline refresh"


def test_start_training_job_rejects_invalid_path(
    patched_training_service: tuple[TrainingJobService, list[FakeProcess]],
) -> None:
    request = TrainingJobStartRequest(dataset_path="../outside.csv", model_type="sequence")

    with pytest.raises(HTTPException) as exc_info:
        training_controller.start_training_job(request)

    assert exc_info.value.status_code == 422
    assert "allowed training data directory" in exc_info.value.detail


def test_start_training_job_blocks_concurrent_run(
    patched_training_service: tuple[TrainingJobService, list[FakeProcess]],
) -> None:
    first = training_controller.start_training_job(TrainingJobStartRequest(model_type="baseline"))

    with pytest.raises(HTTPException) as exc_info:
        training_controller.start_training_job(TrainingJobStartRequest(model_type="sequence"))

    assert first.status == "running"
    assert exc_info.value.status_code == 409
    assert "already running" in exc_info.value.detail


def test_training_job_integration_completes_process(
    patched_training_service: tuple[TrainingJobService, list[FakeProcess]],
) -> None:
    service, processes = patched_training_service

    start = training_controller.start_training_job(
        TrainingJobStartRequest(
            model_type="sequence",
            dataset_path="custom.csv",
            sequence_config={"epochs": 1, "seq_len": 5, "batch_size": 8},
        ),
    )
    job_id = start.job_id
    processes[0].returncode = 0

    response = training_controller.get_training_job(job_id)
    body = response.model_dump(mode="json")

    assert body["job_id"] == job_id
    assert body["status"] == "succeeded"
    assert body["return_code"] == 0
    assert body["dataset_path"].endswith("custom.csv")
    assert body["command"] == processes[0].command
    assert body["log_path"].startswith(str(service._artifacts_root))
    assert "--epochs" in body["command"]
    assert "1" in body["command"]
    assert service._registry_path.exists()


def test_delete_training_job_removes_record_and_artifacts(
    patched_training_service: tuple[TrainingJobService, list[FakeProcess]],
) -> None:
    service, processes = patched_training_service

    start = training_controller.start_training_job(TrainingJobStartRequest(model_type="baseline"))
    job_id = start.job_id
    artifacts_dir = Path(start.artifacts_dir)
    (artifacts_dir / "training_progress.json").write_text("{}", encoding="utf-8")
    processes[0].returncode = 0
    training_controller.get_training_job(job_id)

    training_controller.delete_training_job(job_id)

    assert job_id not in service._jobs
    assert not artifacts_dir.exists()


def test_delete_training_job_rejects_running_job(
    patched_training_service: tuple[TrainingJobService, list[FakeProcess]],
) -> None:
    training_controller_response = training_controller.start_training_job(TrainingJobStartRequest(model_type="baseline"))

    with pytest.raises(HTTPException) as exc_info:
        training_controller.delete_training_job(training_controller_response.job_id)

    assert exc_info.value.status_code == 409
    assert "cannot be removed" in exc_info.value.detail


def test_service_layer_has_no_csv_parsing_calls() -> None:
    source = inspect.getsource(training_job_service)
    assert "read_csv" not in source
    assert ".head(" not in source
    assert ".sample(" not in source
