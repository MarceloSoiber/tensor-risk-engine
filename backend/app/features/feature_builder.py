from __future__ import annotations

from datetime import UTC
from math import asin, cos, log1p, pi, radians, sin, sqrt

from app.domain.transaction import Transaction

OOV_INDEX = 1
EARTH_RADIUS_KM = 6371.0088
EPSILON = 1e-8


class FeatureBuilder:
    def build(self, transaction: Transaction) -> dict[str, float | str]:
        utc_datetime = transaction.transaction_datetime
        if utc_datetime.tzinfo is None:
            utc_datetime = utc_datetime.replace(tzinfo=UTC)
        else:
            utc_datetime = utc_datetime.astimezone(UTC)
        transaction_datetime = utc_datetime.replace(tzinfo=None)

        hour = float(transaction_datetime.hour)
        day_of_week = float(transaction_datetime.weekday())
        month = float(transaction_datetime.month)
        hour_angle = 2.0 * pi * (hour / 24.0)
        day_angle = 2.0 * pi * (day_of_week / 7.0)

        amount = max(0.0, float(transaction.amount))
        average_amount_24h = max(0.0, float(transaction.average_amount_24h))
        amount_delta_ratio = self._amount_delta_ratio(amount, average_amount_24h)

        return {
            "amt": amount,
            "lat": float(transaction.customer_latitude),
            "long": float(transaction.customer_longitude),
            "city_pop": float(transaction.city_population),
            "merch_lat": float(transaction.merchant_latitude),
            "merch_long": float(transaction.merchant_longitude),
            "unix_time": float(utc_datetime.timestamp()),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": 1.0 if transaction_datetime.weekday() in {5, 6} else 0.0,
            "month": month,
            "is_night": 1.0 if transaction_datetime.hour in {0, 1, 2, 3, 4, 5} else 0.0,
            "hour_sin": sin(hour_angle),
            "hour_cos": cos(hour_angle),
            "dow_sin": sin(day_angle),
            "dow_cos": cos(day_angle),
            "geo_distance_km": self._haversine_distance_km(
                transaction.customer_latitude,
                transaction.customer_longitude,
                transaction.merchant_latitude,
                transaction.merchant_longitude,
            ),
            "time_since_prev_tx": 0.0,
            "tx_count_1h": float(transaction.transactions_last_hour),
            "tx_count_24h": float(transaction.transactions_last_24h),
            "amt_mean_24h": average_amount_24h,
            "amt_std_24h": 0.0,
            "amt_zscore_24h": 0.0,
            "log1p_amt": log1p(amount),
            "log1p_city_pop": log1p(max(0.0, float(transaction.city_population))),
            "amount_delta_ratio_24h": amount_delta_ratio,
            "merchant": transaction.merchant,
            "category": transaction.category,
            "gender": transaction.gender,
            "state": transaction.state,
            "job": transaction.job,
        }

    def build_with_category_indices(
        self,
        transaction: Transaction,
        category_mappings: dict[str, dict[str, int]],
    ) -> dict[str, float | str]:
        features = self.build(transaction)
        for column, mapping in category_mappings.items():
            raw_value = str(features.get(column, "__MISSING__"))
            features[f"{column}_idx"] = float(mapping.get(raw_value, OOV_INDEX))
        return features

    @staticmethod
    def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)

        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        a = sin(delta_lat / 2.0) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2.0) ** 2
        a = max(0.0, min(1.0, a))
        return EARTH_RADIUS_KM * 2.0 * asin(sqrt(a))

    @staticmethod
    def _amount_delta_ratio(amount: float, average_amount_24h: float) -> float:
        if average_amount_24h <= EPSILON:
            return 1.0 if amount > 0.0 else 0.0
        return max(0.0, (amount - average_amount_24h) / average_amount_24h)
