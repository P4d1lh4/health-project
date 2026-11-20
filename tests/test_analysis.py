from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.analysis import (
    bucket_top,
    concentration,
    rate_by,
    risk_bands,
    segment_ranking,
    threshold_for_share,
    threshold_summary,
)
from pathlib import Path

from src.data_loading import TARGET_COLUMN, build_cohort, load_raw_data
from src.explain import explain_record, source_feature
from src.feature_engineering import EXAMPLE_INPUT, RAW_INPUT_COLUMNS
from src.labels import DISCHARGE_LABELS, MEDICAL_SPECIALTY_LABELS, label_codes, value_label
from src.modeling import build_pipeline
from tests.synthetic import make_raw_frame

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetic_data.csv"


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sexo": ["F", "F", "F", "F", "M", "M", "M", "M"],
            TARGET_COLUMN: [1, 1, 0, 0, 1, 0, 0, 0],
        }
    )


def test_rate_by_reports_rate_share_and_lift() -> None:
    cohort = _cohort()
    table = rate_by(cohort, cohort["sexo"], min_rows=1).set_index("grupo")

    assert table.loc["F", "taxa"] == 0.5
    assert table.loc["M", "taxa"] == 0.25
    assert table.loc["F", "participacao"] == 0.5
    assert table.loc["F", "lift"] == pytest.approx(0.5 / 0.375)


def test_rate_by_drops_small_groups() -> None:
    cohort = _cohort()
    groups = pd.Series(["a"] * 7 + ["b"])

    assert rate_by(cohort, groups, min_rows=2)["grupo"].tolist() == ["a"]


def test_bucket_top_groups_the_tail_and_keeps_order() -> None:
    table = bucket_top(pd.Series([0, 1, 2, 5, 9]), top=2)

    assert table.tolist() == ["0", "1", "2+", "2+", "2+"]
    assert table.cat.categories.tolist() == ["0", "1", "2+"]


def test_segment_ranking_orders_by_lift() -> None:
    cohort = _cohort()
    ranking = segment_ranking(cohort, {"Sexo": cohort["sexo"]}, min_rows=1)

    assert ranking["grupo"].tolist() == ["F", "M"]
    assert ranking["dimensao"].unique().tolist() == ["Sexo"]


def test_concentration() -> None:
    cohort = _cohort()

    assert concentration(cohort, cohort["sexo"] == "F") == (0.5, pytest.approx(2 / 3))


def test_risk_bands_are_equal_sized_and_concentrate_positives_at_the_top() -> None:
    proba = np.linspace(0.01, 0.99, 100)
    y = (proba > 0.8).astype(int)
    bands = risk_bands(y, proba, bins=10)

    assert bands["atendimentos"].tolist() == [10] * 10
    assert bands["faixa"].tolist() == list(range(1, 11))
    assert bands.loc[bands["faixa"] == 10, "taxa"].item() == 1.0
    assert bands["participacao_readmissoes"].sum() == pytest.approx(1.0)


def test_threshold_summary_counts_per_thousand() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    summary = threshold_summary(y, proba, threshold=0.5)

    assert summary["sinalizados"] == 0.2
    assert summary["precisao"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["alertas_por_mil"] == 200
    assert summary["encontradas_por_mil"] == 100
    assert summary["falsos_alertas_por_mil"] == 100
    assert summary["perdidas_por_mil"] == 100


def test_threshold_for_share_flags_about_the_requested_share() -> None:
    proba = np.linspace(0, 1, 1001)
    threshold = threshold_for_share(proba, 0.2)

    assert (proba >= threshold).mean() == pytest.approx(0.2, abs=0.002)


def test_labels_fall_back_for_unknown_and_missing_values() -> None:
    codes = label_codes(pd.Series([1, 99, np.nan]), DISCHARGE_LABELS)

    assert codes.tolist() == ["Casa", "Código 99", "Não informado"]
    assert value_label("insulin", "Up") == "Dose aumentada"
    assert value_label("medical_specialty", "InternalMedicine") == "Clínica médica"
    assert value_label("medical_specialty", "Especialidade-Nova") == "Especialidade-Nova"
    assert value_label("race", None) == "Não informado"


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="precisa do dataset em disco")
def test_every_specialty_in_the_dataset_has_a_translation() -> None:
    specialties = load_raw_data(DATASET_PATH)["medical_specialty"].dropna().unique()

    assert sorted(set(specialties) - set(MEDICAL_SPECIALTY_LABELS)) == []


def test_source_feature_picks_the_longest_matching_name() -> None:
    assert source_feature("num__number_inpatient") == "number_inpatient"
    assert source_feature("cat__diag_1_group_circulatory") == "diag_1_group"
    assert source_feature("cat__discharge_disposition_id_22") == "discharge_disposition_id"


def test_explanation_adds_up_to_the_model_log_odds() -> None:
    cohort = build_cohort(make_raw_frame(600))
    classifier = XGBClassifier(n_estimators=20, max_depth=3, random_state=0)
    pipeline = build_pipeline(classifier).fit(cohort[list(RAW_INPUT_COLUMNS)], cohort[TARGET_COLUMN])

    def log_odds(record: dict) -> float:
        proba = pipeline.predict_proba(pd.DataFrame([record]))[0, 1]
        return float(np.log(proba / (1 - proba)))

    low = EXAMPLE_INPUT
    high = {**EXAMPLE_INPUT, "number_inpatient": 9, "discharge_disposition_id": 3}

    # O termo base é igual para todo registro, então a diferença entre as somas
    # das contribuições tem de ser a diferença entre os log-odds previstos.
    explained_gap = explain_record(pipeline, high).sum() - explain_record(pipeline, low).sum()
    assert explained_gap == pytest.approx(log_odds(high) - log_odds(low), abs=1e-4)
    assert log_odds(high) != pytest.approx(log_odds(low))


def test_explanation_is_unavailable_for_other_classifiers(trained_model_dir) -> None:
    from src.artifacts import load_artifacts

    assert explain_record(load_artifacts(trained_model_dir).pipeline, EXAMPLE_INPUT) is None
