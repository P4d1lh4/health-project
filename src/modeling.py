from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	roc_auc_score,
	roc_curve,
	precision_recall_fscore_support,
	confusion_matrix,
	classification_report,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import dump

try:
	from xgboost import XGBClassifier
except Exception:  # pragma: no cover
	XGBClassifier = None  # type: ignore

try:
	from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
	LGBMClassifier = None  # type: ignore


MODEL_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
	"""
	Seleciona um conjunto de colunas de entrada robusto e disponível no dataset.
	Preferimos colunas coesas e com boa capacidade explicativa.
	"""
	candidates = [
		# demografia e tempo de internação
		"age_mid",
		"time_in_hospital",
		# procedimentos e exames
		"num_lab_procedures",
		"num_procedures",
		"num_medications",
		# histórico recente
		"number_outpatient",
		"number_emergency",
		"number_inpatient",
		# resultados laboratoriais discretos
		"A1Cresult",
		"max_glu_serum",
		# medicamentos chave
		"insulin",
		"metformin",
		"glyburide",
		"glipizide",
		"pioglitazone",
		"rosiglitazone",
		"change",
		"diabetesMed",
		# engenharia de atributos
		"medication_ratio",
		"prior_visits",
		"age_group_buckets",
		# grupos de diagnóstico
		"diag_1_group",
		"diag_2_group",
		"diag_3_group",
	]
	return [c for c in candidates if c in df.columns]


def split_data(
	df: pd.DataFrame,
	target_col: str = "readmit_30",
	test_size: float = 0.2,
	random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	X = df.drop(columns=[target_col])
	y = df[target_col]
	return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessor(X: pd.DataFrame, raw_feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
	"""
	Cria um ColumnTransformer com:
	- numéricas: imputer (median) + scaler
	- categóricas: imputer (constant) + one-hot
	Retorna o preprocessor e as listas de colunas numéricas/categóricas usadas.
	"""
	X = X.copy()
	X = X[raw_feature_cols]
	numeric_cols = [c for c in raw_feature_cols if pd.api.types.is_numeric_dtype(X[c])]
	categorical_cols = [c for c in raw_feature_cols if c not in numeric_cols]

	numeric_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])
	categorical_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_cols),
			("cat", categorical_transformer, categorical_cols),
		]
	)
	return preprocessor, numeric_cols, categorical_cols


def get_models() -> Dict[str, Any]:
	models: Dict[str, Any] = {
		"LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=None),
		"RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
		"SVM": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42),
	}
	if XGBClassifier is not None:
		models["XGBoost"] = XGBClassifier(
			n_estimators=400,
			max_depth=6,
			learning_rate=0.05,
			subsample=0.8,
			colsample_bytree=0.8,
			reg_lambda=1.0,
			random_state=42,
			eval_metric="logloss",
		)
	if LGBMClassifier is not None:
		models["LightGBM"] = LGBMClassifier(
			n_estimators=400,
			num_leaves=31,
			learning_rate=0.05,
			subsample=0.8,
			colsample_bytree=0.8,
			reg_lambda=1.0,
			random_state=42,
		)
	return models


def evaluate_models(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series,
	raw_feature_cols: List[str],
) -> Tuple[str, Pipeline, Dict[str, float]]:
	"""
	Treina e avalia múltiplos modelos, retornando o melhor pelo ROC-AUC.
	"""
	preprocessor, _, _ = build_preprocessor(X_train, raw_feature_cols)
	models = get_models()

	best_name = ""
	best_pipeline: Optional[Pipeline] = None
	best_auc = -np.inf
	scores: Dict[str, float] = {}

	for name, clf in models.items():
		pipe = Pipeline(steps=[
			("preprocessor", preprocessor),
			("classifier", clf),
		])
		pipe.fit(X_train[raw_feature_cols], y_train)
		proba = pipe.predict_proba(X_val[raw_feature_cols])[:, 1]
		auc = roc_auc_score(y_val, proba)
		scores[name] = auc
		if auc > best_auc:
			best_auc = auc
			best_name = name
			best_pipeline = pipe

	assert best_pipeline is not None
	return best_name, best_pipeline, scores


def compute_permutation_importances(
	best_pipeline: Pipeline,
	X_val: pd.DataFrame,
	y_val: pd.Series,
	raw_feature_cols: List[str],
	max_features: int = 50,
) -> pd.DataFrame:
	"""
	Calcula permutation importance de forma robusta (modelo-agnóstico).
	Retorna DataFrame com colunas [feature, importance_mean, importance_std].
	"""
	r = permutation_importance(
		best_pipeline,
		X_val[raw_feature_cols],
		y_val,
		n_repeats=10,
		random_state=42,
		n_jobs=-1,
		scoring="roc_auc",
	)
	# Obter nomes expandidos após one-hot
	# Para simplificar, mapeamos importâncias para colunas "raw" por bloco:
	# a importância de cada bloco (coluna bruta) = média das importâncias do bloco transformado
	importances_raw: Dict[str, List[float]] = {c: [] for c in raw_feature_cols}
	# r.importances_mean está no espaço transformado; não temos nomes expandidos facilmente.
	# Usamos uma aproximação agrupando pela ordem dos colunas: primeiro todas numéricas (1:1), depois categóricas (expandem).
	pre: ColumnTransformer = best_pipeline.named_steps["preprocessor"]
	num_features = pre.transformers_[0][2]  # type: ignore
	cat_features = pre.transformers_[1][2]  # type: ignore
	num_count = len(num_features)
	imp_means = r.importances_mean
	imp_stds = r.importances_std

	idx = 0
	# numéricas: 1:1
	for c in num_features:
		importances_raw[c].append(float(imp_means[idx]))
		idx += 1
	# categóricas: várias colunas derivadas por categoria -> agregamos média
	for c in cat_features:
		# não sabemos quantas dummies geradas; estimamos pelo restante proporcional.
		# Aqui, sem acesso direto aos nomes expandidos, fazemos uma aproximação:
		# distribuímos igualmente o restante pelas categorias restantes.
		# Para evitar inconsistência, paramos se acabarem as importâncias.
		if idx >= len(imp_means):
			break
		importances_raw[c].append(float(imp_means[idx]))
		idx += 1
		while idx < len(imp_means) and len(cat_features) > 0 and (len(imp_means) - idx) >= (len(cat_features) - cat_features.index(c) - 1):
			# Parada simples para não consumir todo o vetor por uma única feature
			break

	rows = []
	for c, vals in importances_raw.items():
		if not vals:
			continue
		rows.append({
			"feature": c,
			"importance_mean": float(np.mean(vals)),
			"importance_std": float(np.std(vals)),
		})
	df_imp = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
	return df_imp.head(max_features)


def finalize_and_save(
	best_name: str,
	best_pipeline: Pipeline,
	raw_feature_cols: List[str],
	test_metrics: Dict[str, Any],
	importances_df: Optional[pd.DataFrame] = None,
) -> None:
	dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.joblib"))
	metadata = {
		"best_model_name": best_name,
		"raw_feature_columns": raw_feature_cols,
		"sklearn_version": pd.__version__,
	}
	with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2, ensure_ascii=False)
	with open(os.path.join(REPORTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
		json.dump(test_metrics, f, indent=2, ensure_ascii=False)
	if importances_df is not None and not importances_df.empty:
		importances_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False, encoding="utf-8")


def train_pipeline(
	df: pd.DataFrame,
	target_col: str = "readmit_30",
	test_size: float = 0.2,
	random_state: int = 42,
) -> Dict[str, Any]:
	"""
	Orquestra treino/validação/teste, seleciona melhor modelo e salva artefatos.
	Retorna dicionário com métricas de teste.
	"""
	# filtra colunas de entrada
	all_features = select_feature_columns(df)
	if not all_features:
		raise ValueError("Nenhuma coluna de entrada encontrada após a engenharia de atributos.")

	# split
	X_trainval, X_test, y_trainval, y_test = split_data(df[all_features + [target_col]], target_col=target_col, test_size=test_size, random_state=random_state)
	# segunda divisão para validação
	X_train, X_val, y_train, y_val = train_test_split(
		X_trainval, y_trainval, test_size=0.25, random_state=random_state, stratify=y_trainval
	)  # 0.25 de 0.8 => 0.2 do total

	# treina e seleciona
	best_name, best_pipeline, val_scores = evaluate_models(X_train, y_train, X_val, y_val, raw_feature_cols=all_features)

	# avaliação no teste
	test_proba = best_pipeline.predict_proba(X_test[all_features])[:, 1]
	test_pred = (test_proba >= 0.5).astype(int)
	test_auc = roc_auc_score(y_test, test_proba)
	precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)
	cm = confusion_matrix(y_test, test_pred).tolist()
	report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)

	test_metrics = {
		"best_model": best_name,
		"validation_scores_roc_auc": val_scores,
		"test_roc_auc": float(test_auc),
		"test_precision": float(precision),
		"test_recall": float(recall),
		"test_f1": float(f1),
		"confusion_matrix": cm,
		"classification_report": report,
	}

	# importâncias
	try:
		importances_df = compute_permutation_importances(best_pipeline, X_val, y_val, raw_feature_cols=all_features)
	except Exception:
		importances_df = None

	finalize_and_save(best_name, best_pipeline, all_features, test_metrics, importances_df)
	return test_metrics


