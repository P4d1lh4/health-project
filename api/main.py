from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from joblib import load


APP_TITLE = "Hospital Readmission Prediction API"
APP_VERSION = "0.1.0"
MODEL_PATH = os.path.join("models", "best_model.joblib")
METADATA_PATH = os.path.join("models", "metadata.json")

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

_PIPELINE = None
_RAW_FEATURE_COLS = None
_MODEL_NAME = None


def _load_artifacts() -> None:
	global _PIPELINE, _RAW_FEATURE_COLS, _MODEL_NAME
	if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
		return
	_PIPELINE = load(MODEL_PATH)
	with open(METADATA_PATH, "r", encoding="utf-8") as f:
		meta = json.load(f)
	_MODEL_NAME = meta.get("best_model_name")
	_RAW_FEATURE_COLS = meta.get("raw_feature_columns", [])


@app.on_event("startup")
def on_startup() -> None:
	_load_artifacts()


@app.get("/health")
def health() -> Dict[str, Any]:
	status = "ok" if _PIPELINE is not None else "model_not_loaded"
	return {"status": status, "model": _MODEL_NAME}


@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
	if _PIPELINE is None or _RAW_FEATURE_COLS is None:
		raise HTTPException(status_code=503, detail="Modelo não carregado. Treine o modelo primeiro.")

	# Constrói DataFrame com as colunas esperadas
	row = {}
	for c in _RAW_FEATURE_COLS:
		row[c] = payload.get(c, None)
	X = pd.DataFrame([row])

	try:
		proba = float(_PIPELINE.predict_proba(X)[:, 1][0])
		pred = int(proba >= 0.5)
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Erro ao inferir: {e}")

	return {
		"model": _MODEL_NAME,
		"probability_readmit_30": proba,
		"prediction": pred,
	}


