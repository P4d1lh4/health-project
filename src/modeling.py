from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data_loading import GROUP_COLUMN, TARGET_COLUMN
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_INPUT_COLUMNS, engineer_features

MODEL_FILENAME = "best_model.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"
IMPORTANCE_FILENAME = "feature_importance.csv"
PREDICTIONS_FILENAME = "test_predictions.csv"

CV_FOLDS = 5
# Categorias com menos de 0,1% das linhas viram "infrequent" (ex.: medical_specialty raras).
MIN_CATEGORY_FREQUENCY = 0.001
PERMUTATION_REPEATS = 3


def candidate_classifiers(random_state: int) -> dict[str, BaseEstimator]:
    """Sem class_weight: o desbalanceamento é tratado pelo limiar e pela PR-AUC."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def build_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Pipeline completo: colunas brutas -> atributos -> pré-processamento -> classificador."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_FREQUENCY),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(engineer_features)),
            ("preprocessor", preprocessor),
            ("classifier", clone(classifier)),
        ]
    )


def split_by_patient(
    cohort: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino/teste por `patient_nbr`, estratificado pelo alvo; nenhum paciente fica nos dois."""
    if not 0 < test_size < 0.5:
        raise ValueError(f"test_size deve estar em (0, 0.5); recebido {test_size}.")
    n_splits = round(1 / test_size)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(
        splitter.split(cohort, cohort[TARGET_COLUMN], groups=cohort[GROUP_COLUMN])
    )
    return cohort.iloc[train_idx].reset_index(drop=True), cohort.iloc[test_idx].reset_index(drop=True)


@dataclass(frozen=True)
class CandidateResult:
    fold_roc_auc: tuple[float, ...]
    fold_pr_auc: tuple[float, ...]
    oof_proba: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "roc_auc_mean": float(np.mean(self.fold_roc_auc)),
            "roc_auc_std": float(np.std(self.fold_roc_auc)),
            "pr_auc_mean": float(np.mean(self.fold_pr_auc)),
            "pr_auc_std": float(np.std(self.fold_pr_auc)),
        }


def cross_validate_candidate(
    classifier: BaseEstimator, train: pd.DataFrame, n_splits: int, random_state: int
) -> CandidateResult:
    """Validação cruzada agrupada por paciente, com probabilidades out-of-fold."""
    X = train[list(RAW_INPUT_COLUMNS)]
    y = train[TARGET_COLUMN].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(train), np.nan)
    roc, pr = [], []
    for fit_idx, val_idx in splitter.split(X, y, groups=train[GROUP_COLUMN]):
        pipeline = build_pipeline(classifier).fit(X.iloc[fit_idx], y[fit_idx])
        proba = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        roc.append(roc_auc_score(y[val_idx], proba))
        pr.append(average_precision_score(y[val_idx], proba))
    return CandidateResult(tuple(roc), tuple(pr), oof)


def choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Limiar que maximiza o F1 nas probabilidades out-of-fold."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "prevalence": float(np.mean(y_true)),
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def compute_permutation_importances(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, random_state: int
) -> pd.DataFrame:
    """O pipeline recebe colunas brutas, então há uma importância por coluna de RAW_INPUT_COLUMNS."""
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=random_state,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def train_and_save(
    cohort: pd.DataFrame,
    model_dir: Path,
    reports_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = CV_FOLDS,
    classifiers: dict[str, BaseEstimator] | None = None,
) -> dict[str, Any]:
    """
    Seleciona o modelo por validação cruzada agrupada, escolhe o limiar, reajusta
    no treino completo, avalia uma vez no teste e grava os artefatos, incluindo
    as probabilidades previstas no teste.
    """
    candidates = classifiers if classifiers is not None else candidate_classifiers(random_state)
    train, test = split_by_patient(cohort, test_size, random_state)
    y_train = train[TARGET_COLUMN].to_numpy()

    results = {
        name: cross_validate_candidate(clf, train, n_splits, random_state)
        for name, clf in candidates.items()
    }
    cv_summary = {name: result.summary() for name, result in results.items()}
    best_name = max(cv_summary, key=lambda name: cv_summary[name]["roc_auc_mean"])
    threshold = choose_threshold(y_train, results[best_name].oof_proba)

    X_train = train[list(RAW_INPUT_COLUMNS)]
    X_test = test[list(RAW_INPUT_COLUMNS)]
    y_test = test[TARGET_COLUMN].to_numpy()
    pipeline = build_pipeline(candidates[best_name]).fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold)
    predictions = pd.DataFrame({"y_true": y_test, "proba": test_proba})
    importances = compute_permutation_importances(pipeline, X_test, y_test, random_state)

    counts = {
        "cohort_rows": int(len(cohort)),
        "cohort_patients": int(cohort[GROUP_COLUMN].nunique()),
        "train_rows": int(len(train)),
        "train_patients": int(train[GROUP_COLUMN].nunique()),
        "test_rows": int(len(test)),
        "test_patients": int(test[GROUP_COLUMN].nunique()),
    }
    metadata = {
        "model_name": best_name,
        "raw_input_columns": list(RAW_INPUT_COLUMNS),
        "threshold": threshold,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **counts,
    }
    metrics = {
        "best_model": best_name,
        "split": (
            f"teste = {test_size:.0%} dos pacientes (StratifiedGroupKFold por {GROUP_COLUMN}); "
            f"seleção por CV de {n_splits} folds agrupada por paciente no treino"
        ),
        "cv_scores": cv_summary,
        "threshold_selection": "máximo F1 nas probabilidades out-of-fold do modelo vencedor",
        "test": test_metrics,
        **counts,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    dump(pipeline, model_dir / MODEL_FILENAME)
    _write_json(model_dir / METADATA_FILENAME, metadata)
    _write_json(reports_dir / METRICS_FILENAME, metrics)
    importances.to_csv(reports_dir / IMPORTANCE_FILENAME, index=False, encoding="utf-8")
    predictions.to_csv(reports_dir / PREDICTIONS_FILENAME, index=False, encoding="utf-8")
    return metrics
