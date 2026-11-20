from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class InputKind(Enum):
    """Tipo de cada coluna bruta aceita pelo modelo."""

    COUNT = "count"
    CODE = "code"
    CATEGORY = "category"


@dataclass(frozen=True)
class InputColumn:
    name: str
    kind: InputKind
    example: str | int


# Contrato único entre treino, API e dashboard: colunas brutas do CSV que o modelo consome.
RAW_INPUT_SCHEMA: tuple[InputColumn, ...] = (
    InputColumn("age", InputKind.CATEGORY, "[60-70)"),
    InputColumn("gender", InputKind.CATEGORY, "Female"),
    InputColumn("race", InputKind.CATEGORY, "Caucasian"),
    InputColumn("admission_type_id", InputKind.CODE, 1),
    InputColumn("discharge_disposition_id", InputKind.CODE, 1),
    InputColumn("admission_source_id", InputKind.CODE, 7),
    InputColumn("time_in_hospital", InputKind.COUNT, 5),
    InputColumn("num_lab_procedures", InputKind.COUNT, 40),
    InputColumn("num_procedures", InputKind.COUNT, 1),
    InputColumn("num_medications", InputKind.COUNT, 12),
    InputColumn("number_outpatient", InputKind.COUNT, 1),
    InputColumn("number_emergency", InputKind.COUNT, 0),
    InputColumn("number_inpatient", InputKind.COUNT, 0),
    InputColumn("number_diagnoses", InputKind.COUNT, 9),
    InputColumn("A1Cresult", InputKind.CATEGORY, "None"),
    InputColumn("max_glu_serum", InputKind.CATEGORY, "None"),
    InputColumn("insulin", InputKind.CATEGORY, "Steady"),
    InputColumn("metformin", InputKind.CATEGORY, "No"),
    InputColumn("glyburide", InputKind.CATEGORY, "No"),
    InputColumn("glipizide", InputKind.CATEGORY, "No"),
    InputColumn("pioglitazone", InputKind.CATEGORY, "No"),
    InputColumn("rosiglitazone", InputKind.CATEGORY, "No"),
    InputColumn("change", InputKind.CATEGORY, "Ch"),
    InputColumn("diabetesMed", InputKind.CATEGORY, "Yes"),
    InputColumn("diag_1", InputKind.CATEGORY, "428"),
    InputColumn("diag_2", InputKind.CATEGORY, "250.01"),
    InputColumn("diag_3", InputKind.CATEGORY, "401"),
    InputColumn("medical_specialty", InputKind.CATEGORY, "InternalMedicine"),
)

RAW_INPUT_COLUMNS: tuple[str, ...] = tuple(c.name for c in RAW_INPUT_SCHEMA)
EXAMPLE_INPUT: dict[str, str | int] = {c.name: c.example for c in RAW_INPUT_SCHEMA}

DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")
COUNT_COLUMNS = tuple(c.name for c in RAW_INPUT_SCHEMA if c.kind is InputKind.COUNT)
PASSTHROUGH_CATEGORY_COLUMNS = tuple(
    c.name
    for c in RAW_INPUT_SCHEMA
    if c.kind is not InputKind.COUNT and c.name not in DIAGNOSIS_COLUMNS and c.name != "age"
)

NUMERIC_FEATURES: tuple[str, ...] = ("age_mid", *COUNT_COLUMNS, "medication_ratio", "prior_visits")
CATEGORICAL_FEATURES: tuple[str, ...] = (
    *PASSTHROUGH_CATEGORY_COLUMNS,
    *(f"{c}_group" for c in DIAGNOSIS_COLUMNS),
)

# Faixas de ICD-9 usadas por Strack et al. (2014), artigo de origem do dataset.
_ICD9_NUMERIC_GROUPS: tuple[tuple[float, float, str], ...] = (
    (390, 460, "circulatory"),
    (785, 786, "circulatory"),
    (460, 520, "respiratory"),
    (786, 787, "respiratory"),
    (520, 580, "digestive"),
    (787, 788, "digestive"),
    (250, 251, "diabetes"),
    (800, 1000, "injury"),
    (710, 740, "musculoskeletal"),
    (580, 630, "genitourinary"),
    (788, 789, "genitourinary"),
    (140, 240, "neoplasms"),
)

_AGE_RANGE = re.compile(r"^\[(\d+)-(\d+)\)$")


def parse_age_to_midpoint(age: object) -> float:
    """Converte a faixa etária do dataset (ex.: "[60-70)") no ponto médio (65.0); NaN se inválida."""
    if not isinstance(age, str):
        return np.nan
    match = _AGE_RANGE.match(age.strip())
    if match is None:
        return np.nan
    low, high = (float(g) for g in match.groups())
    return (low + high) / 2.0


def map_icd_to_group(code: object) -> str:
    """Mapeia um código ICD-9 para um grupo clínico amplo; "missing" se ausente."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "missing"
    text = str(code).strip().upper()
    if not text:
        return "missing"
    if text.startswith(("E", "V")):
        return "other"
    try:
        value = float(text)
    except ValueError:
        return "other"
    for low, high, group in _ICD9_NUMERIC_GROUPS:
        if low <= value < high:
            return group
    return "other"


def _as_category(series: pd.Series) -> pd.Series:
    # NaN (e não None) para o SimpleImputer reconhecer o valor ausente em colunas object.
    return series.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe colunas brutas de RAW_INPUT_COLUMNS (colunas ausentes viram NaN) e
    devolve o DataFrame de atributos que o modelo consome. Não altera `raw`.
    """
    frame = raw.reindex(columns=list(RAW_INPUT_COLUMNS))
    counts = frame[list(COUNT_COLUMNS)].apply(pd.to_numeric, errors="coerce")

    numeric = pd.DataFrame(
        {
            "age_mid": frame["age"].map(parse_age_to_midpoint).astype(float),
            **{c: counts[c] for c in COUNT_COLUMNS},
            "medication_ratio": counts["num_medications"] / counts["time_in_hospital"].clip(lower=1),
            "prior_visits": counts[["number_outpatient", "number_emergency", "number_inpatient"]].sum(
                axis=1, min_count=1
            ),
        },
        index=frame.index,
    )
    categorical = pd.DataFrame(
        {
            **{c: _as_category(frame[c]) for c in PASSTHROUGH_CATEGORY_COLUMNS},
            **{f"{c}_group": frame[c].map(map_icd_to_group).astype(object) for c in DIAGNOSIS_COLUMNS},
        },
        index=frame.index,
    )
    return pd.concat([numeric, categorical], axis=1)[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
