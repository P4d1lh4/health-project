from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, NonNegativeInt, create_model

from src.artifacts import ModelArtifacts, artifacts_exist, load_artifacts
from src.feature_engineering import RAW_INPUT_SCHEMA, InputKind

APP_TITLE = "API de Previsão de Readmissão Hospitalar"
APP_VERSION = "0.2.0"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

logger = logging.getLogger(__name__)

_FIELD_TYPES: dict[InputKind, Any] = {
    InputKind.COUNT: Optional[NonNegativeInt],
    InputKind.CODE: Optional[int],
    InputKind.CATEGORY: Optional[str],
}

PatientRecord = create_model(
    "PatientRecord",
    __config__=ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{c.name: c.example for c in RAW_INPUT_SCHEMA}]},
    ),
    **{c.name: (_FIELD_TYPES[c.kind], None) for c in RAW_INPUT_SCHEMA},
)


class PredictionResponse(BaseModel):
    model: str
    probability_readmit_30: float
    threshold: float
    prediction: int


def create_app(model_dir: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if artifacts_exist(model_dir):
            app.state.artifacts = load_artifacts(model_dir)
        else:
            logger.warning("Artefatos não encontrados em %s; /predict responderá 503.", model_dir)
            app.state.artifacts = None
        yield

    app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

    def _artifacts(request: Request) -> ModelArtifacts:
        artifacts: ModelArtifacts | None = request.app.state.artifacts
        if artifacts is None:
            raise HTTPException(status_code=503, detail="Modelo não carregado. Treine o modelo primeiro.")
        return artifacts

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        artifacts: ModelArtifacts | None = request.app.state.artifacts
        if artifacts is None:
            return {"status": "model_not_loaded", "model": None}
        return {"status": "ok", "model": artifacts.model_name, "threshold": artifacts.threshold}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(record: PatientRecord, request: Request) -> PredictionResponse:  # type: ignore[valid-type]
        artifacts = _artifacts(request)
        try:
            result = artifacts.predict(record.model_dump())
        except Exception:
            logger.exception("Falha na inferência")
            raise HTTPException(status_code=500, detail="Erro interno ao calcular a predição.")
        return PredictionResponse(
            model=artifacts.model_name,
            probability_readmit_30=result.probability,
            threshold=result.threshold,
            prediction=result.prediction,
        )

    return app


app = create_app(Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)))
