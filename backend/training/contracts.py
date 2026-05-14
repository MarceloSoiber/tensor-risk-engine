"""Training feature and schema contracts for the fraud dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class FeatureSpecError(ValueError):
    """Raised when the feature specification is invalid."""


def _normalize_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise FeatureSpecError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise FeatureSpecError(f"{field_name} cannot be empty.")
    return cleaned


def _normalize_optional_str(value: Any, *, field_name: str) -> str:
    if value is None:
        return ""
    return _normalize_str(value, field_name=field_name)


def _normalize_str_list(values: Any, *, field_name: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise FeatureSpecError(f"{field_name} must be a list of strings.")

    normalized = [_normalize_str(item, field_name=f"{field_name} item") for item in values]
    if len(set(normalized)) != len(normalized):
        raise FeatureSpecError(f"{field_name} cannot contain duplicates.")
    return normalized


def _ensure_subset(subset: Sequence[str], superset: Sequence[str], *, field_name: str) -> None:
    missing = [item for item in subset if item not in superset]
    if missing:
        raise FeatureSpecError(
            f"{field_name} contains values that are not declared in the parent set: {missing}."
        )


@dataclass(frozen=True, slots=True)
class TargetContract:
    name: str
    positive_label: int = 1
    negative_label: int = 0
    type: str = "binary"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TargetContract":
        return cls(
            name=_normalize_str(data.get("name"), field_name="target.name"),
            positive_label=int(data.get("positive_label", 1)),
            negative_label=int(data.get("negative_label", 0)),
            type=_normalize_str(data.get("type", "binary"), field_name="target.type"),
        )

    def validate(self) -> None:
        if self.type != "binary":
            raise FeatureSpecError("target.type must be binary.")
        if self.positive_label == self.negative_label:
            raise FeatureSpecError("target labels must be different.")


@dataclass(frozen=True, slots=True)
class EntityIdContract:
    columns: list[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EntityIdContract":
        return cls(
            columns=_normalize_str_list(data.get("columns", []), field_name="entity_id.columns"),
            description=_normalize_optional_str(data.get("description", ""), field_name="entity_id.description"),
        )

    def validate(self) -> None:
        if not self.columns:
            raise FeatureSpecError("entity_id.columns must contain at least one column.")


@dataclass(frozen=True, slots=True)
class TimeColumnContract:
    name: str
    format: str = "%Y-%m-%d %H:%M:%S"
    timezone: str = "naive"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TimeColumnContract":
        return cls(
            name=_normalize_str(data.get("name"), field_name="time_col.name"),
            format=_normalize_optional_str(data.get("format", "%Y-%m-%d %H:%M:%S"), field_name="time_col.format"),
            timezone=_normalize_optional_str(data.get("timezone", "naive"), field_name="time_col.timezone"),
        )

    def validate(self) -> None:
        if not self.format:
            raise FeatureSpecError("time_col.format cannot be empty.")
        if self.timezone not in {"naive", "UTC"}:
            raise FeatureSpecError("time_col.timezone must be either naive or UTC.")


@dataclass(frozen=True, slots=True)
class SplitContract:
    strategy: str = "time_based"
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = False
    stratify_target: bool = False
    group_by_entity: bool = True
    random_seed: int = 42

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SplitContract":
        return cls(
            strategy=_normalize_optional_str(data.get("strategy", "time_based"), field_name="split.strategy"),
            train_ratio=float(data.get("train_ratio", 0.7)),
            validation_ratio=float(data.get("validation_ratio", 0.15)),
            test_ratio=float(data.get("test_ratio", 0.15)),
            shuffle=bool(data.get("shuffle", False)),
            stratify_target=bool(data.get("stratify_target", False)),
            group_by_entity=bool(data.get("group_by_entity", True)),
            random_seed=int(data.get("random_seed", 42)),
        )

    def validate(self) -> None:
        if self.strategy != "time_based":
            raise FeatureSpecError("split.strategy must be time_based.")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(value <= 0 for value in ratios):
            raise FeatureSpecError("split ratios must be greater than zero.")
        total = sum(ratios)
        if abs(total - 1.0) > 1e-6:
            raise FeatureSpecError("split ratios must sum to 1.0.")
        if self.shuffle:
            raise FeatureSpecError("time-based splits must not shuffle data.")


@dataclass(frozen=True, slots=True)
class SequenceContract:
    enabled: bool = False
    sort_by: list[str] = field(default_factory=list)
    window_size: int = 10
    step_size: int = 1
    min_history_size: int = 2
    feature_columns: list[str] = field(default_factory=list)
    target_shift: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SequenceContract":
        return cls(
            enabled=bool(data.get("enabled", False)),
            sort_by=_normalize_str_list(data.get("sort_by", []), field_name="sequence.sort_by"),
            window_size=int(data.get("window_size", 10)),
            step_size=int(data.get("step_size", 1)),
            min_history_size=int(data.get("min_history_size", 2)),
            feature_columns=_normalize_str_list(
                data.get("feature_columns", []), field_name="sequence.feature_columns"
            ),
            target_shift=int(data.get("target_shift", 1)),
        )

    def validate(self) -> None:
        if self.window_size <= 0:
            raise FeatureSpecError("sequence.window_size must be positive.")
        if self.step_size <= 0:
            raise FeatureSpecError("sequence.step_size must be positive.")
        if self.min_history_size <= 0:
            raise FeatureSpecError("sequence.min_history_size must be positive.")
        if self.target_shift <= 0:
            raise FeatureSpecError("sequence.target_shift must be positive.")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    target: TargetContract
    entity_id: EntityIdContract
    time_col: TimeColumnContract
    id_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)
    log1p_columns: list[str] = field(default_factory=list)
    high_cardinality_categorical: list[str] = field(default_factory=list)
    low_cardinality_categorical: list[str] = field(default_factory=list)
    split: SplitContract = field(default_factory=SplitContract)
    sequence: SequenceContract = field(default_factory=SequenceContract)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FeatureSpec":
        spec_path = Path(path) if path is not None else Path(__file__).with_name("specs") / "feature_spec.json"
        with spec_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FeatureSpec":
        required_sections = {
            "target",
            "entity_id",
            "time_col",
            "id_columns",
            "numeric_columns",
            "categorical_columns",
            "drop_columns",
            "log1p_columns",
            "high_cardinality_categorical",
            "low_cardinality_categorical",
            "split",
            "sequence",
        }
        extra_sections = set(data.keys()) - required_sections
        missing_sections = required_sections - set(data.keys())
        if extra_sections:
            raise FeatureSpecError(f"Unexpected top-level sections: {sorted(extra_sections)}.")
        if missing_sections:
            raise FeatureSpecError(f"Missing top-level sections: {sorted(missing_sections)}.")

        spec = cls(
            target=TargetContract.from_mapping(data["target"]),
            entity_id=EntityIdContract.from_mapping(data["entity_id"]),
            time_col=TimeColumnContract.from_mapping(data["time_col"]),
            id_columns=_normalize_str_list(data.get("id_columns", []), field_name="id_columns"),
            numeric_columns=_normalize_str_list(data.get("numeric_columns", []), field_name="numeric_columns"),
            categorical_columns=_normalize_str_list(
                data.get("categorical_columns", []), field_name="categorical_columns"
            ),
            drop_columns=_normalize_str_list(data.get("drop_columns", []), field_name="drop_columns"),
            log1p_columns=_normalize_str_list(data.get("log1p_columns", []), field_name="log1p_columns"),
            high_cardinality_categorical=_normalize_str_list(
                data.get("high_cardinality_categorical", []),
                field_name="high_cardinality_categorical",
            ),
            low_cardinality_categorical=_normalize_str_list(
                data.get("low_cardinality_categorical", []),
                field_name="low_cardinality_categorical",
            ),
            split=SplitContract.from_mapping(data["split"]),
            sequence=SequenceContract.from_mapping(data["sequence"]),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        self.target.validate()
        self.entity_id.validate()
        self.time_col.validate()
        self.split.validate()
        self.sequence.validate()

        for field_name in (
            "id_columns",
            "numeric_columns",
            "categorical_columns",
            "drop_columns",
            "log1p_columns",
            "high_cardinality_categorical",
            "low_cardinality_categorical",
        ):
            value = getattr(self, field_name)
            if len(set(value)) != len(value):
                raise FeatureSpecError(f"{field_name} cannot contain duplicates.")

        _ensure_subset(self.log1p_columns, self.numeric_columns, field_name="log1p_columns")
        _ensure_subset(
            self.high_cardinality_categorical,
            self.categorical_columns,
            field_name="high_cardinality_categorical",
        )
        _ensure_subset(
            self.low_cardinality_categorical,
            self.categorical_columns,
            field_name="low_cardinality_categorical",
        )

        overlap = set(self.high_cardinality_categorical) & set(self.low_cardinality_categorical)
        if overlap:
            raise FeatureSpecError(
                f"high_cardinality_categorical and low_cardinality_categorical must be disjoint: {sorted(overlap)}."
            )

        protected = {self.target.name, self.time_col.name, *self.entity_id.columns}
        conflicts = protected & set(self.drop_columns)
        if conflicts:
            raise FeatureSpecError(f"Protected columns cannot be in drop_columns: {sorted(conflicts)}.")

        if self.target.name in self.id_columns:
            raise FeatureSpecError("target.name cannot be listed in id_columns.")

        if self.sequence.enabled:
            _ensure_subset(self.sequence.sort_by, self.entity_id.columns + [self.time_col.name], field_name="sequence.sort_by")
            _ensure_subset(self.sequence.feature_columns, self.model_feature_columns, field_name="sequence.feature_columns")

    @property
    def model_feature_columns(self) -> list[str]:
        return [column for column in self.numeric_columns + self.categorical_columns if column not in self.drop_columns and column not in self.id_columns]

    @property
    def feature_columns(self) -> list[str]:
        return self.model_feature_columns

    @property
    def categorical_feature_columns(self) -> list[str]:
        return [column for column in self.categorical_columns if column not in self.drop_columns and column not in self.id_columns]

    @property
    def numeric_feature_columns(self) -> list[str]:
        return [column for column in self.numeric_columns if column not in self.drop_columns and column not in self.id_columns]

    @property
    def high_cardinality_categorical_columns(self) -> list[str]:
        return list(self.high_cardinality_categorical)

    @property
    def low_cardinality_categorical_columns(self) -> list[str]:
        return list(self.low_cardinality_categorical)

    @property
    def id_feature_columns(self) -> list[str]:
        return list(self.id_columns)

    @property
    def target_column(self) -> str:
        return self.target.name

    @property
    def time_column(self) -> str:
        return self.time_col.name

    @property
    def excluded_columns(self) -> list[str]:
        ordered: list[str] = []
        for column in [*self.drop_columns, *self.id_columns, self.target.name, self.time_col.name, *self.entity_id.columns]:
            if column not in ordered:
                ordered.append(column)
        return ordered

    @property
    def all_excluded_columns(self) -> list[str]:
        return self.excluded_columns

    @property
    def model_input_columns(self) -> list[str]:
        return self.feature_columns

    @property
    def utility_columns(self) -> list[str]:
        return self.excluded_columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": {
                "name": self.target.name,
                "positive_label": self.target.positive_label,
                "negative_label": self.target.negative_label,
                "type": self.target.type,
            },
            "entity_id": {
                "columns": list(self.entity_id.columns),
                "description": self.entity_id.description,
            },
            "time_col": {
                "name": self.time_col.name,
                "format": self.time_col.format,
                "timezone": self.time_col.timezone,
            },
            "id_columns": list(self.id_columns),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "drop_columns": list(self.drop_columns),
            "log1p_columns": list(self.log1p_columns),
            "high_cardinality_categorical": list(self.high_cardinality_categorical),
            "low_cardinality_categorical": list(self.low_cardinality_categorical),
            "split": {
                "strategy": self.split.strategy,
                "train_ratio": self.split.train_ratio,
                "validation_ratio": self.split.validation_ratio,
                "test_ratio": self.split.test_ratio,
                "shuffle": self.split.shuffle,
                "stratify_target": self.split.stratify_target,
                "group_by_entity": self.split.group_by_entity,
                "random_seed": self.split.random_seed,
            },
            "sequence": {
                "enabled": self.sequence.enabled,
                "sort_by": list(self.sequence.sort_by),
                "window_size": self.sequence.window_size,
                "step_size": self.sequence.step_size,
                "min_history_size": self.sequence.min_history_size,
                "feature_columns": list(self.sequence.feature_columns),
                "target_shift": self.sequence.target_shift,
            },
        }


def load_feature_spec(path: str | Path | None = None) -> FeatureSpec:
    return FeatureSpec.load(path)


__all__ = [
    "EntityIdContract",
    "FeatureSpec",
    "FeatureSpecError",
    "SequenceContract",
    "SplitContract",
    "TargetContract",
    "TimeColumnContract",
    "load_feature_spec",
]
