from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .analysis import bucket_top
from .feature_engineering import map_icd_to_group
from .labels import (
    ADMISSION_TYPE_LABELS,
    DIAGNOSIS_GROUP_LABELS,
    DISCHARGE_LABELS,
    feature_label,
    label_codes,
    label_values,
)


@dataclass(frozen=True)
class Dimension:
    """Um recorte da coorte: nome exibido, como agrupar e se os grupos têm ordem natural."""

    name: str
    groups: Callable[[pd.DataFrame], pd.Series]
    ordered: bool


def _count(column: str, top: int) -> Dimension:
    return Dimension(feature_label(column), lambda df: bucket_top(df[column], top), ordered=True)


def _category(column: str) -> Dimension:
    return Dimension(feature_label(column), lambda df: label_values(df[column], column), ordered=False)


DIMENSIONS: tuple[Dimension, ...] = (
    _count("number_inpatient", 5),
    _count("number_emergency", 4),
    _count("number_outpatient", 4),
    _count("time_in_hospital", 10),
    _count("number_diagnoses", 9),
    Dimension(feature_label("age"), lambda df: df["age"].fillna("Não informado"), ordered=True),
    Dimension(
        feature_label("discharge_disposition_id"),
        lambda df: label_codes(df["discharge_disposition_id"], DISCHARGE_LABELS),
        ordered=False,
    ),
    Dimension(
        feature_label("admission_type_id"),
        lambda df: label_codes(df["admission_type_id"], ADMISSION_TYPE_LABELS),
        ordered=False,
    ),
    Dimension(
        feature_label("diag_1"),
        lambda df: df["diag_1"].map(map_icd_to_group).map(DIAGNOSIS_GROUP_LABELS),
        ordered=False,
    ),
    _category("A1Cresult"),
    _category("insulin"),
    _category("change"),
    _category("diabetesMed"),
    _category("gender"),
    _category("race"),
    _category("medical_specialty"),
)

ORDERED_DIMENSIONS: frozenset[str] = frozenset(d.name for d in DIMENSIONS if d.ordered)


def build_dimensions(cohort: pd.DataFrame) -> pd.DataFrame:
    """Uma coluna por dimensão, com os rótulos de grupo de cada atendimento da coorte."""
    return pd.DataFrame({d.name: d.groups(cohort) for d in DIMENSIONS}, index=cohort.index)
