from __future__ import annotations

import math

import pandas as pd
import pytest

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    EXAMPLE_INPUT,
    NUMERIC_FEATURES,
    engineer_features,
    map_icd_to_group,
    parse_age_to_midpoint,
)


@pytest.mark.parametrize(
    ("code", "group"),
    [
        ("428", "circulatory"),
        ("785", "circulatory"),
        ("486", "respiratory"),
        ("786", "respiratory"),
        ("571.2", "digestive"),
        ("250.83", "diabetes"),
        ("250", "diabetes"),
        ("251", "other"),
        ("820", "injury"),
        ("715", "musculoskeletal"),
        ("599", "genitourinary"),
        ("788", "genitourinary"),
        ("174", "neoplasms"),
        ("V45", "other"),
        ("E909", "other"),
        (None, "missing"),
        (float("nan"), "missing"),
    ],
)
def test_icd_mapping(code: object, group: str) -> None:
    assert map_icd_to_group(code) == group


def test_age_midpoint() -> None:
    assert parse_age_to_midpoint("[60-70)") == 65.0
    assert math.isnan(parse_age_to_midpoint(None))
    assert math.isnan(parse_age_to_midpoint("sessenta"))


def test_engineer_features_from_raw_row() -> None:
    raw = pd.DataFrame([EXAMPLE_INPUT])
    features = engineer_features(raw)

    assert list(features.columns) == [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    row = features.iloc[0]
    assert row["age_mid"] == 65.0
    assert row["medication_ratio"] == 12 / 5
    assert row["prior_visits"] == 1
    assert row["diag_1_group"] == "circulatory"
    assert row["admission_source_id"] == "7"
    assert list(raw.columns) == list(EXAMPLE_INPUT)


def test_test_not_done_is_distinct_from_missing() -> None:
    raw = pd.DataFrame([{"A1Cresult": "None"}, {"A1Cresult": None}])
    a1c = engineer_features(raw)["A1Cresult"]

    assert a1c.iloc[0] == "None"
    assert pd.isna(a1c.iloc[1])


def test_engineer_features_tolerates_all_missing_row() -> None:
    features = engineer_features(pd.DataFrame([{"age": None}]))

    assert len(features) == 1
    assert features[list(NUMERIC_FEATURES)].isna().all(axis=None)
    assert features[["diag_1_group", "diag_2_group", "diag_3_group"]].eq("missing").all(axis=None)
