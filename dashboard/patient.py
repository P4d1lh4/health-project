from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.charts import decimal, pct
from dashboard.views import show_chart, tile
from src.artifacts import ModelArtifacts
from src.explain import explain_record
from src.feature_engineering import DIAGNOSIS_COLUMNS, RAW_INPUT_SCHEMA, InputColumn, InputKind
from src.labels import ADMISSION_TYPE_LABELS, DISCHARGE_LABELS, feature_label, value_label

RECORD_KEY = "patient_record"
NOT_INFORMED = None

# Como o formulário distribui as colunas de entrada do modelo: três colunas de tela, cada uma com seus grupos.
FORM_LAYOUT: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...] = (
    (
        ("Perfil", ("age", "gender", "race")),
        (
            "Internação",
            ("admission_type_id", "admission_source_id", "discharge_disposition_id", "time_in_hospital", "medical_specialty"),
        ),
    ),
    (
        ("Histórico no ano anterior", ("number_inpatient", "number_emergency", "number_outpatient")),
        ("Volume de cuidado", ("num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses")),
        ("Diagnósticos (CID-9)", DIAGNOSIS_COLUMNS),
    ),
    (
        ("Exames", ("A1Cresult", "max_glu_serum")),
        (
            "Medicação",
            ("insulin", "metformin", "glyburide", "glipizide", "pioglitazone", "rosiglitazone", "change", "diabetesMed"),
        ),
    ),
)

CODE_LABELS: Mapping[str, Mapping[int, str]] = {
    "discharge_disposition_id": DISCHARGE_LABELS,
    "admission_type_id": ADMISSION_TYPE_LABELS,
}
_SCHEMA = {column.name: column for column in RAW_INPUT_SCHEMA}


def form_columns() -> list[str]:
    return [name for screen_column in FORM_LAYOUT for _, names in screen_column for name in names]


def _category_options(column: InputColumn, cohort: pd.DataFrame | None) -> list[str | None]:
    if cohort is None:
        return [str(column.example), NOT_INFORMED]
    values = cohort[column.name].dropna().astype(str).value_counts().index.tolist()
    example = str(column.example)
    return [*([] if example in values else [example]), *values, NOT_INFORMED]


def _widget(column: InputColumn, cohort: pd.DataFrame | None) -> Any:
    label = feature_label(column.name)
    if column.kind is InputKind.COUNT:
        return st.number_input(label, min_value=0, value=int(column.example), step=1)
    if column.name in CODE_LABELS:
        codes = CODE_LABELS[column.name]
        options = list(codes)
        return st.selectbox(label, options, index=options.index(int(column.example)), format_func=codes.get)
    if column.kind is InputKind.CODE:
        return st.number_input(f"{label} (código)", min_value=1, value=int(column.example), step=1)
    if column.name in DIAGNOSIS_COLUMNS:
        return st.text_input(label, value=str(column.example)).strip() or NOT_INFORMED
    options = _category_options(column, cohort)
    return st.selectbox(
        label,
        options,
        index=options.index(str(column.example)),
        format_func=lambda value: value_label(column.name, value),
    )


def _form(cohort: pd.DataFrame | None) -> dict[str, Any] | None:
    record: dict[str, Any] = {}
    with st.form("paciente"):
        for screen_column, groups in zip(st.columns(3), FORM_LAYOUT):
            with screen_column:
                for title, names in groups:
                    st.markdown(f"**{title}**")
                    for name in names:
                        record[name] = _widget(_SCHEMA[name], cohort)
        submitted = st.form_submit_button("Calcular risco", type="primary")
    return record if submitted else None


def _result(artifacts: ModelArtifacts, record: Mapping[str, Any], test_proba: np.ndarray | None, prevalence: float | None) -> None:
    result = artifacts.predict(record)
    left, right = st.columns([1, 2])
    with left:
        tile(st, "Risco de readmissão em 30 dias", pct(result.probability))
        if prevalence:
            st.markdown(f"**{decimal(result.probability / prevalence, 1)}×** a taxa média de {pct(prevalence)}.")
        if test_proba is not None:
            st.markdown(f"Maior que o de **{pct(float((test_proba < result.probability).mean()), 0)}** dos atendimentos do teste.")
        if result.prediction:
            st.warning(f"Sinalizar: risco igual ou acima do limiar de {pct(result.threshold)}.", icon="⚠️")
        else:
            st.info(f"Não sinalizar: risco abaixo do limiar de {pct(result.threshold)}.", icon="ℹ️")
    with right:
        contributions = explain_record(artifacts.pipeline, record)
        if contributions is None:
            st.caption("A explicação por atributo só está disponível quando o modelo é XGBoost.")
            return
        st.markdown("##### O que mais pesou nesta previsão")
        show_chart(charts.contribution_chart(contributions))
        st.caption("Efeito de cada atributo em relação a um paciente médio. Explica o modelo, não a causa da readmissão.")


def patient(
    artifacts: ModelArtifacts, cohort: pd.DataFrame | None, test_proba: np.ndarray | None, prevalence: float | None
) -> None:
    st.markdown("Preencha os dados da alta para ver o risco previsto e o que mais pesou nele.")
    submitted = _form(cohort)
    if submitted is not None:
        st.session_state[RECORD_KEY] = submitted
    record = st.session_state.get(RECORD_KEY)
    if record is None:
        return
    st.divider()
    _result(artifacts, record, test_proba, prevalence)
    st.caption("Estudo com dados históricos de hospitais dos EUA (1999-2008). Não usar para decisão clínica.")
