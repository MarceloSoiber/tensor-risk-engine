from __future__ import annotations

import pandas as pd

from training.pipeline import run_data_pipeline
from training.sequences import SequenceConfig, build_sequence_arrays


def _raw_sequence_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        rows.append(
            {
                "Unnamed: 0": index,
                "trans_date_trans_time": f"2024-01-01 {index:02d}:00:00",
                "cc_num": 4000000000000000 + (index % 3),
                "merchant": f"merchant_{index % 2}",
                "category": "grocery_pos",
                "amt": 25.0 + index,
                "first": "Jane",
                "last": "Doe",
                "gender": "F",
                "street": "100 Main St",
                "city": "Testville",
                "state": "CA",
                "zip": "90001",
                "lat": 34.0,
                "long": -118.0,
                "city_pop": 10000,
                "job": "Engineer",
                "dob": "1990-01-01",
                "trans_num": f"tx_{index}",
                "unix_time": 1_700_000_000 + (index * 3600),
                "merch_lat": 34.1,
                "merch_long": -118.1,
                "is_fraud": 1 if index % 4 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_data_pipeline_preserves_sequence_grouping_columns(tmp_path) -> None:
    dataset_path = tmp_path / "fraudTrain.csv"
    _raw_sequence_frame().to_csv(dataset_path, index=False)

    frame, spec, artifacts = run_data_pipeline(dataset_path=dataset_path)
    arrays = build_sequence_arrays(
        frame,
        spec=spec,
        numeric_columns=artifacts.numeric_columns,
        categorical_index_columns=artifacts.categorical_index_columns,
        config=SequenceConfig(seq_len=5, stride=1),
        split_value="train",
    )

    assert spec.entity_id.columns[0] in frame.columns
    assert spec.time_column in frame.columns
    assert arrays.x_num.shape[0] > 0
    assert arrays.x_cat.shape[0] == arrays.x_num.shape[0]
