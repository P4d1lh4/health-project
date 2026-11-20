from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from .feature_engineering import RAW_INPUT_COLUMNS
from .modeling import METADATA_FILENAME, MODEL_FILENAME


@dataclass(frozen=True)
class Prediction:
    probability: float
    threshold: float
    prediction: int


@dataclass(frozen=True)
class ModelArtifacts:
    pipeline: Pipeline
    model_name: str
    threshold: float

    def predict(self, record: Mapping[str, Any]) -> Prediction:
        """Prediz a partir de um registro bruto; campos ausentes viram NaN e são imputados."""
        row = pd.DataFrame([{c: record.get(c) for c in RAW_INPUT_COLUMNS}])
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        return Prediction(probability, self.threshold, int(probability >= self.threshold))


def artifacts_exist(model_dir: Path) -> bool:
    return (model_dir / MODEL_FILENAME).exists() and (model_dir / METADATA_FILENAME).exists()


def load_artifacts(model_dir: Path) -> ModelArtifacts:
    metadata = json.loads((model_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    if list(metadata["raw_input_columns"]) != list(RAW_INPUT_COLUMNS):
        raise ValueError(
            "As colunas de entrada do modelo salvo divergem de RAW_INPUT_COLUMNS; treine o modelo novamente."
        )
    return ModelArtifacts(
        pipeline=load(model_dir / MODEL_FILENAME),
        model_name=metadata["model_name"],
        threshold=float(metadata["threshold"]),
    )
