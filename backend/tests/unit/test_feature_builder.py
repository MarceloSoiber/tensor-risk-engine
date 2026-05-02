from datetime import datetime

import pytest

from app.domain.transaction import Transaction
from app.features.feature_builder import FeatureBuilder, OOV_INDEX


def _transaction(**overrides: object) -> Transaction:
    payload = {
        "amount": 250.0,
        "transaction_datetime": datetime.fromisoformat("2020-06-21T03:15:00"),
        "merchant": "known_merchant",
        "category": "grocery_pos",
        "gender": "F",
        "state": "NY",
        "job": "Engineer",
        "city_population": 125000,
        "customer_latitude": 40.7128,
        "customer_longitude": -74.0060,
        "merchant_latitude": 40.7306,
        "merchant_longitude": -73.9352,
        "transactions_last_hour": 2,
        "transactions_last_24h": 8,
        "average_amount_24h": 75.0,
    }
    payload.update(overrides)
    return Transaction(**payload)


def test_build_derives_model_ready_numeric_features() -> None:
    features = FeatureBuilder().build(_transaction())

    assert features["amt"] == 250.0
    assert features["log1p_amt"] == pytest.approx(5.525452939)
    assert features["hour"] == 3.0
    assert features["day_of_week"] == 6.0
    assert features["is_weekend"] == 1.0
    assert features["is_night"] == 1.0
    assert features["tx_count_1h"] == 2.0
    assert features["tx_count_24h"] == 8.0
    assert features["amt_mean_24h"] == 75.0
    assert features["geo_distance_km"] == pytest.approx(6.283, rel=0.01)


def test_build_with_category_indices_uses_oov_for_unknown_values() -> None:
    mappings = {
        "merchant": {"known_merchant": 2},
        "category": {"grocery_pos": 4},
        "gender": {"F": 2},
        "state": {"NY": 33},
        "job": {"Doctor": 10},
    }

    features = FeatureBuilder().build_with_category_indices(_transaction(), mappings)

    assert features["merchant_idx"] == 2.0
    assert features["category_idx"] == 4.0
    assert features["gender_idx"] == 2.0
    assert features["state_idx"] == 33.0
    assert features["job_idx"] == float(OOV_INDEX)
