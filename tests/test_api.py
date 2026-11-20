from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.feature_engineering import EXAMPLE_INPUT


@pytest.fixture()
def client(trained_model_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(trained_model_dir)) as test_client:
        yield test_client


def test_predict_returns_probability_and_stored_threshold(client: TestClient, trained_model_dir: Path) -> None:
    stored = json.loads((trained_model_dir / "metadata.json").read_text(encoding="utf-8"))

    response = client.post("/predict", json=EXAMPLE_INPUT)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 < body["probability_readmit_30"] < 1.0
    assert body["threshold"] == stored["threshold"]
    assert body["model"] == stored["model_name"]
    assert body["prediction"] == int(body["probability_readmit_30"] >= body["threshold"])


def test_predict_accepts_partial_payload(client: TestClient) -> None:
    response = client.post("/predict", json={"age": "[70-80)", "number_inpatient": 2})

    assert response.status_code == 200


def test_predict_rejects_negative_count(client: TestClient) -> None:
    response = client.post("/predict", json={**EXAMPLE_INPUT, "number_inpatient": -1})

    assert response.status_code == 422


def test_predict_rejects_unknown_field(client: TestClient) -> None:
    response = client.post("/predict", json={**EXAMPLE_INPUT, "age_mid": 65})

    assert response.status_code == 422


def test_predict_without_model_returns_503(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as test_client:
        assert test_client.get("/health").json()["status"] == "model_not_loaded"
        assert test_client.post("/predict", json=EXAMPLE_INPUT).status_code == 503
