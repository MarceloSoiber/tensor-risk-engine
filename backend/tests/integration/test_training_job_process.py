from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from app.services.training_job_service import TrainingJobService


pytestmark = pytest.mark.integration


CSV_HEADER = (
    "Unnamed: 0,trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,"
    "street,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,"
    "merch_long,is_fraud\n"
)


def _write_minimal_training_dataset(path: Path, row_count: int = 24) -> None:
    rows = [CSV_HEADER]
    for index in range(row_count):
        is_fraud = 1 if index % 3 == 0 else 0
        cc_num = 4000000000000000 + (index % 4)
        hour = index % 24
        amount = 12.50 + (index * 3.75) + (120.0 if is_fraud else 0.0)
        unix_time = 1_700_000_000 + (index * 3600)
        rows.append(
            ",".join(
                [
                    str(index),
                    f"2024-01-{(index // 24) + 1:02d} {hour:02d}:00:00",
                    str(cc_num),
                    f"merchant_{index % 5}",
                    "grocery_pos" if index % 2 == 0 else "misc_net",
                    f"{amount:.2f}",
                    "Jane",
                    "Doe",
                    "F" if index % 2 == 0 else "M",
                    "100 Main St",
                    "Testville",
                    "CA" if index % 2 == 0 else "NY",
                    "90001",
                    f"{34.0 + (index * 0.01):.6f}",
                    f"{-118.0 - (index * 0.01):.6f}",
                    str(10_000 + index),
                    f"job_{index % 4}",
                    "1990-01-01",
                    f"tx_{index:04d}",
                    str(unix_time),
                    f"{34.1 + (index * 0.01):.6f}",
                    f"{-118.1 - (index * 0.01):.6f}",
                    str(is_fraud),
                ]
            )
            + "\n"
        )
    path.write_text("".join(rows), encoding="utf-8")


def _assert_process_is_visible(pid: int) -> None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise AssertionError(f"Training process {pid} is not visible to the OS.") from exc


def test_start_job_runs_real_training_process_and_writes_observable_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "training-data"
    artifacts_root = tmp_path / "training-artifacts"
    registry_path = artifacts_root / "jobs_registry.json"
    data_root.mkdir()
    dataset_path = data_root / "fraudTrain.csv"
    _write_minimal_training_dataset(dataset_path)

    service = TrainingJobService()
    service._jobs = {}
    service._processes = {}
    service._log_files = {}
    service._training_data_root = data_root.resolve()
    service._default_dataset = dataset_path.resolve()
    service._artifacts_root = artifacts_root.resolve()
    service._registry_path = registry_path.resolve()
    service._python_bin = sys.executable

    started = service.start_job({"model_type": "baseline"})

    job_id = started["job_id"]
    pid = started["pid"]
    log_path = Path(started["log_path"])
    artifacts_dir = Path(started["artifacts_dir"])

    print(f"\njob_id={job_id}")
    print(f"pid={pid}")
    print(f"artifacts_dir={artifacts_dir}")
    print(f"log_path={log_path}")
    print(f"registry_path={registry_path}")
    print(f"command={' '.join(started['command'])}")

    assert started["status"] == "running"
    assert isinstance(pid, int)
    _assert_process_is_visible(pid)
    assert artifacts_dir.exists()
    assert log_path.exists()
    assert registry_path.exists()

    deadline = time.monotonic() + 90
    final_state = started
    while time.monotonic() < deadline:
        final_state = service.get_job(job_id)
        print(f"status={final_state['status']} return_code={final_state['return_code']}")
        if final_state["status"] in {"succeeded", "failed", "canceled"}:
            break
        time.sleep(0.5)

    assert final_state["status"] == "succeeded"
    assert final_state["return_code"] == 0
    assert log_path.read_text(encoding="utf-8")
    assert (artifacts_dir / "baseline_model.joblib").exists()
    assert (artifacts_dir / "metrics_test.json").exists()
    assert "Baseline training complete." in log_path.read_text(encoding="utf-8")

    consulted_job = service.get_job(job_id)
    listed_jobs = service.list_jobs()
    metrics_test = json.loads((Path(consulted_job["artifacts_dir"]) / "metrics_test.json").read_text(encoding="utf-8"))
    metadata = json.loads((Path(consulted_job["artifacts_dir"]) / "training_metadata.json").read_text(encoding="utf-8"))

    print(f"consulted_status={consulted_job['status']}")
    print(f"consulted_artifacts_dir={consulted_job['artifacts_dir']}")
    print(f"test_metrics={metrics_test}")

    assert consulted_job["job_id"] == job_id
    assert consulted_job["status"] == "succeeded"
    assert consulted_job["return_code"] == 0
    assert consulted_job["error"] is None
    assert consulted_job["dataset_metadata"]["size_bytes"] == dataset_path.stat().st_size
    assert consulted_job["log_path"] == str(log_path)
    assert any(item["job_id"] == job_id for item in listed_jobs)
    assert metadata["model_type"] == "logistic_regression"
    assert metadata["split_counts"]["train"] > 0
    assert metrics_test.keys() >= {"pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"}
