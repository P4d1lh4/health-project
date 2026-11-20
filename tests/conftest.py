from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from src.data_loading import build_cohort
from src.modeling import train_and_save
from tests.synthetic import make_raw_frame


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("treino")
    model_dir = root / "models"
    train_and_save(
        build_cohort(make_raw_frame(600)),
        model_dir=model_dir,
        reports_dir=root / "reports",
        n_splits=2,
        classifiers={"LogisticRegression": LogisticRegression(max_iter=1000)},
    )
    return model_dir
