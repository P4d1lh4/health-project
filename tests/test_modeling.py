from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.artifacts import load_artifacts
from src.data_loading import build_cohort
from src.feature_engineering import EXAMPLE_INPUT, RAW_INPUT_COLUMNS
from src.modeling import choose_threshold, split_by_patient
from tests.synthetic import make_raw_frame


def test_no_patient_in_both_train_and_test() -> None:
    cohort = build_cohort(make_raw_frame(900, seed=1))
    train, test = split_by_patient(cohort, test_size=0.2, random_state=0)

    assert set(train["patient_nbr"]).isdisjoint(test["patient_nbr"])
    assert len(train) + len(test) == len(cohort)
    assert 0.1 < len(test) / len(cohort) < 0.3


def test_threshold_maximizes_f1() -> None:
    y = np.array([0, 0, 0, 1, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    assert choose_threshold(y, proba) == 0.4


def test_saved_pipeline_predicts_from_raw_dict_with_missing_fields(trained_model_dir: Path) -> None:
    artifacts = load_artifacts(trained_model_dir)
    partial = {k: v for k, v in EXAMPLE_INPUT.items() if k not in {"age", "A1Cresult", "diag_2"}}

    for record in (EXAMPLE_INPUT, partial, {"medical_specialty": "Especialidade-Nunca-Vista"}, {}):
        result = artifacts.predict(record)
        assert 0.0 < result.probability < 1.0
        assert result.prediction == int(result.probability >= result.threshold)


def test_saved_pipeline_accepts_raw_columns_directly(trained_model_dir: Path) -> None:
    pipeline = load_artifacts(trained_model_dir).pipeline
    raw = make_raw_frame(5, seed=3)[list(RAW_INPUT_COLUMNS)]

    assert pipeline.predict_proba(raw).shape == (5, 2)


def test_metadata_and_reports(trained_model_dir: Path) -> None:
    metadata = json.loads((trained_model_dir / "metadata.json").read_text(encoding="utf-8"))
    reports_dir = trained_model_dir.parent / "reports"
    metrics = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    importances = pd.read_csv(reports_dir / "feature_importance.csv")

    assert metadata["sklearn_version"] == sklearn.__version__
    assert metadata["raw_input_columns"] == list(RAW_INPUT_COLUMNS)
    assert metadata["threshold"] == metrics["test"]["threshold"]
    assert sorted(importances["feature"]) == sorted(RAW_INPUT_COLUMNS)

    predictions = pd.read_csv(reports_dir / "test_predictions.csv")
    assert len(predictions) == metrics["test_rows"]
    assert predictions["y_true"].mean() == metrics["test"]["prevalence"]
    assert predictions["proba"].between(0, 1).all()
