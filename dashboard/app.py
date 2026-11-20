import os
import json
from typing import Any, Dict

import pandas as pd
import streamlit as st
from joblib import load
from src.kaggle_utils import ensure_diabetic_csv


st.set_page_config(page_title="Hospital Readmission Prediction", layout="wide")

MODEL_PATH = os.path.join("models", "best_model.joblib")
METADATA_PATH = os.path.join("models", "metadata.json")
DATA_PATH_DEFAULT = os.path.join("data", "diabetic_data.csv")
REPORTS_METRICS = os.path.join("reports", "metrics.json")
REPORTS_IMPORTANCE = os.path.join("reports", "feature_importance.csv")


@st.cache_resource
def load_artifacts():
	if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
		return None, None, None
	model = load(MODEL_PATH)
	with open(METADATA_PATH, "r", encoding="utf-8") as f:
		meta = json.load(f)
	return model, meta.get("raw_feature_columns", []), meta.get("best_model_name", "")


def load_metrics():
	if os.path.exists(REPORTS_METRICS):
		with open(REPORTS_METRICS, "r", encoding="utf-8") as f:
			return json.load(f)
	return None


def load_importances():
	if os.path.exists(REPORTS_IMPORTANCE):
		return pd.read_csv(REPORTS_IMPORTANCE)
	return None


def sidebar_prediction_form(feature_cols):
	st.sidebar.header("Entrada para Predição")
	user_inputs: Dict[str, Any] = {}
	# Campos comuns e seguros de expor
	common_fields = [
		"age", "time_in_hospital", "num_lab_procedures", "num_medications",
		"number_outpatient", "number_emergency", "number_inpatient",
		"insulin", "metformin", "glyburide", "glipizide",
		"diag_1", "diag_2", "diag_3",
	]
	for c in common_fields:
		if c not in feature_cols and c not in ["age", "diag_1", "diag_2", "diag_3"]:
			# Exibimos mesmo que não esteja na lista de colunas brutas; o pipeline lida com engenharia
			pass
	if "age" in common_fields:
		user_inputs["age"] = st.sidebar.selectbox("age (faixa)", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)
	user_inputs["time_in_hospital"] = st.sidebar.number_input("time_in_hospital", min_value=0, max_value=365, value=5, step=1)
	user_inputs["num_lab_procedures"] = st.sidebar.number_input("num_lab_procedures", min_value=0, max_value=500, value=40, step=1)
	user_inputs["num_medications"] = st.sidebar.number_input("num_medications", min_value=0, max_value=200, value=12, step=1)
	user_inputs["number_outpatient"] = st.sidebar.number_input("number_outpatient", min_value=0, max_value=200, value=0, step=1)
	user_inputs["number_emergency"] = st.sidebar.number_input("number_emergency", min_value=0, max_value=200, value=0, step=1)
	user_inputs["number_inpatient"] = st.sidebar.number_input("number_inpatient", min_value=0, max_value=200, value=0, step=1)
	user_inputs["insulin"] = st.sidebar.selectbox("insulin", ["No", "Up", "Down", "Steady"])
	user_inputs["metformin"] = st.sidebar.selectbox("metformin", ["No", "Up", "Down", "Steady"])
	user_inputs["glyburide"] = st.sidebar.selectbox("glyburide", ["No", "Up", "Down", "Steady"])
	user_inputs["glipizide"] = st.sidebar.selectbox("glipizide", ["No", "Up", "Down", "Steady"])
	user_inputs["diag_1"] = st.sidebar.text_input("diag_1 (ICD)", value="250.13")
	user_inputs["diag_2"] = st.sidebar.text_input("diag_2 (ICD)", value="401.9")
	user_inputs["diag_3"] = st.sidebar.text_input("diag_3 (ICD)", value="V45.81")
	return user_inputs


def main():
	st.title("Previsão de Readmissão Hospitalar (<= 30 dias)")

	# Carregar artefatos
	model, feature_cols, model_name = load_artifacts()

	# Painel esquerdo: upload de dataset para EDA básica
	st.sidebar.header("Dados para EDA / Kaggle")
	data_path = st.sidebar.text_input("Caminho do CSV", value=DATA_PATH_DEFAULT)
	with st.sidebar.expander("Baixar do Kaggle"):
		kaggle_dataset = st.text_input("Slug Kaggle (OWNER/DATASET)", value="", help="Ex.: owner/dataset-name")
		kaggle_url = st.text_input("URL do dataset Kaggle", value="", help="Use URL de dataset (não link de Code)")
		if st.button("Baixar CSV do Kaggle"):
			try:
				final_path = ensure_diabetic_csv(
					target_path=DATA_PATH_DEFAULT,
					kaggle_dataset=kaggle_dataset or None,
					kaggle_url=kaggle_url or None,
				)
				st.success(f"Baixado em: {final_path}")
			except Exception as e:
				st.error(f"Falha no download: {e}")
	df = None
	if os.path.exists(data_path):
		try:
			df = pd.read_csv(data_path, na_values=["?", "Unknown/Invalid"])
		except Exception:
			df = None

	# Seção de métricas e importâncias (se existirem)
	metrics = load_metrics()
	importances = load_importances()

	col1, col2 = st.columns(2)
	with col1:
		st.subheader("Status do Modelo")
		if model is None:
			st.error("Modelo não encontrado. Execute o treino em scripts/train.py.")
		else:
			st.success(f"Modelo carregado: {model_name}")
			if metrics:
				st.json({
			"test_roc_auc": metrics.get("test_roc_auc"),
			"test_precision": metrics.get("test_precision"),
			"test_recall": metrics.get("test_recall"),
			"test_f1": metrics.get("test_f1"),
		})
	with col2:
		st.subheader("Importâncias de Atributos")
		if importances is not None and not importances.empty:
			st.dataframe(importances.head(20))
		else:
			st.info("Importâncias não disponíveis.")

	# EDA rápida
	st.subheader("EDA Rápida (opcional)")
	if df is not None:
		if "readmitted" in df.columns:
			st.write("Distribuição do alvo (readmitted):")
			st.bar_chart(df["readmitted"].value_counts())
		cols_to_show = [c for c in ["time_in_hospital", "num_lab_procedures", "num_medications"] if c in df.columns]
		if cols_to_show:
			st.write("Estatísticas descritivas:")
			st.dataframe(df[cols_to_show].describe().T)
	else:
		st.info("Carregue o CSV em data/ ou informe o caminho na barra lateral para EDA.")

	# Formulário de predição
	user_inputs = sidebar_prediction_form(feature_cols or [])
	if st.sidebar.button("Prever"):
		if model is None:
			st.error("Modelo não carregado.")
		else:
			# Model espera colunas brutas; combinamos com engenharia no pipeline
			row = {}
			for c in (feature_cols or []):
				row[c] = user_inputs.get(c, None)
			# adicionamos campos que geram features derivadas
			for extra in ["age", "diag_1", "diag_2", "diag_3"]:
				if extra not in row and extra in user_inputs:
					row[extra] = user_inputs.get(extra)
			X = pd.DataFrame([row])
			try:
				proba = float(model.predict_proba(X)[:, 1][0])
				pred = int(proba >= 0.5)
				st.success(f"Probabilidade de readmissão (<=30 dias): {proba:.3f} | Pred: {pred}")
			except Exception as e:
				st.error(f"Erro ao inferir: {e}")


if __name__ == "__main__":
	main()


