from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from .feature_engineering import CATEGORICAL_FEATURES, RAW_INPUT_COLUMNS


def source_feature(transformed_name: str) -> str:
    """Atributo de origem de uma coluna do pré-processador (ex.: "cat__insulin_Up" -> "insulin")."""
    block, rest = transformed_name.split("__", 1)
    if block == "num":
        return rest
    candidates = [c for c in CATEGORICAL_FEATURES if rest == c or rest.startswith(f"{c}_")]
    return max(candidates, key=len)


def explain_record(pipeline: Pipeline, record: Mapping[str, Any]) -> pd.Series | None:
    """
    Contribuição de cada atributo para a previsão de um registro, em log-odds
    (valores SHAP exatos do XGBoost). Positivo aumenta o risco. Devolve None
    quando o classificador não é XGBoost.
    """
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, xgb.XGBClassifier):
        return None
    row = pd.DataFrame([{c: record.get(c) for c in RAW_INPUT_COLUMNS}])
    # A matriz segue esparsa, como no treino: o XGBoost trata o zero implícito como
    # ausente, e a versão densa percorreria outros ramos das árvores.
    transformed = pipeline[:-1].transform(row)
    contributions = classifier.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[0]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    # O último valor é o termo base do modelo, não um atributo.
    by_feature = pd.Series(contributions[:-1]).groupby([source_feature(n) for n in names]).sum()
    return by_feature.sort_values(key=abs, ascending=False)
