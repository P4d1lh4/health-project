from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.patient import form_columns
from src.data_loading import build_cohort
from src.dimensions import DIMENSIONS, ORDERED_DIMENSIONS, build_dimensions
from src.feature_engineering import RAW_INPUT_COLUMNS
from tests.synthetic import make_raw_frame

APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_patient_form_covers_every_model_input_once() -> None:
    assert sorted(form_columns()) == sorted(RAW_INPUT_COLUMNS)


def test_dimensions_label_every_row() -> None:
    cohort = build_cohort(make_raw_frame(300))
    dimensions = build_dimensions(cohort)

    assert len(dimensions) == len(cohort)
    assert dimensions.columns.tolist() == [d.name for d in DIMENSIONS]
    assert not dimensions.isna().any().any()
    assert dimensions["Destino da alta"].isin(["Casa", "Enfermagem especializada (SNF)", "Casa com assistência domiciliar"]).all()
    assert "Internações no ano anterior" in ORDERED_DIMENSIONS


@pytest.mark.skipif(
    not (APP_PATH.parents[1] / "reports" / "test_predictions.csv").exists()
    or not (APP_PATH.parents[1] / "data" / "diabetic_data.csv").exists(),
    reason="precisa do dataset e do modelo treinado em disco",
)
def test_dashboard_renders_and_predicts_without_errors() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not app.exception

    next(button for button in app.button if button.label == "Calcular risco").click().run()
    assert not app.exception
    assert any("Risco de readmissão em 30 dias" == metric.label for metric in app.metric)
    assert isinstance(app.session_state["patient_record"], dict)
    assert pd.notna(app.session_state["patient_record"]["number_inpatient"])
