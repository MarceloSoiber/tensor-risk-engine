from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
import inspect

from app.controllers.v1 import training_controller
from app.main import app
from app.services import training_job_service
from app.services.training_job_service import TrainingJobService


class FakeProcess:
    _pid_counter = 90000

    def __init__(self) -> None:
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
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def patched_training_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainingJobService:
    data_dir = Path(__file__).resolve().parents[1] / "training" / "data"

    def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:  # noqa: ARG001
        return FakeProcess()

    service = TrainingJobService()
    monkeypatch.setattr(service, "_training_data_root", data_dir.resolve())
    monkeypatch.setattr(service, "_default_dataset", (data_dir / "fraudTrain.csv").resolve())
    monkeypatch.setattr(service, "_artifacts_root", (tmp_path / "artifacts").resolve())
    monkeypatch.setattr(service, "_registry_path", (tmp_path / "jobs_registry.json").resolve())
    monkeypatch.setattr(training_controller, "training_job_service", service)
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    return service


def test_start_training_job_uses_default_dataset(
    client: TestClient,
    patched_training_service: TrainingJobService,
) -> None:
    response = client.post("/api/v1/training/jobs", json={"model_type": "baseline"})
    body = response.json()
    print(f"Body: {body}")
    
    assert response.status_code == 202
    assert body["status"] == "running"
    assert body["model_type"] == "baseline"
    assert body["dataset_path"].endswith("fraudTrain.csv")
    assert body["dataset_metadata"]["size_bytes"] > 0
    assert patched_training_service._registry_path.exists()


# def test_start_training_job_rejects_invalid_path(client: TestClient, patched_training_service: TrainingJobService) -> None:
#     response = client.post(
#         "/api/v1/training/jobs",
#         json={"dataset_path": "../outside.csv", "model_type": "sequence"},
#     )
#     body = response.json()

#     assert response.status_code == 422
#     assert "allowed training data directory" in body["detail"]


# def test_start_training_job_blocks_concurrent_run(
#     client: TestClient,
#     patched_training_service: TrainingJobService,
# ) -> None:
#     first = client.post("/api/v1/training/jobs", json={"model_type": "baseline"})
#     second = client.post("/api/v1/training/jobs", json={"model_type": "sequence"})

#     assert first.status_code == 202
#     assert second.status_code == 409
#     assert "already running" in second.json()["detail"]


# def test_get_training_job_returns_metadata(client: TestClient, patched_training_service: TrainingJobService) -> None:
#     start = client.post("/api/v1/training/jobs", json={"model_type": "sequence"})
#     job_id = start.json()["job_id"]

#     response = client.get(f"/api/v1/training/jobs/{job_id}")
#     body = response.json()

#     assert response.status_code == 200
#     assert body["job_id"] == job_id
#     assert body["status"] == "running"
#     assert body["dataset_metadata"]["size_bytes"] > 0
#     assert "modified_at" in body["dataset_metadata"]


# def test_service_layer_has_no_csv_parsing_calls() -> None:
#     source = inspect.getsource(training_job_service)
#     assert "read_csv" not in source
#     assert ".head(" not in source
#     assert ".sample(" not in source
