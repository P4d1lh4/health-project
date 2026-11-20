from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import numpy as np


MISSING_TOKENS = ["?", "Unknown/Invalid", "UNKNOWN", "NA", "N/A", ""]


def parse_age_to_midpoint(age_str: Optional[str]) -> Optional[float]:
	"""
	Converte a faixa etária do dataset (ex.: "[60-70)") para o ponto médio (ex.: 65.0).
	Retorna None se não for possível converter.
	"""
	if age_str is None or not isinstance(age_str, str):
		return None
	s = age_str.strip()
	if not s or s[0] != "[" or "-" not in s:
		return None
	try:
		# Formato típico: "[60-70)"
		range_part = s.strip("[]()")
		lo, hi = range_part.split("-")
		return (float(lo) + float(hi)) / 2.0
	except Exception:
		return None


def load_raw_data(csv_path: str) -> pd.DataFrame:
	"""
	Carrega o CSV, marcando tokens de missing e sem converter automaticamente tipos problemáticos.
	"""
	df = pd.read_csv(
		csv_path,
		na_values=MISSING_TOKENS,
		low_memory=False,
	)
	return df


def _make_binary_target(df: pd.DataFrame, source_col: str = "readmitted", target_col: str = "readmit_30") -> pd.DataFrame:
	"""
	Transforma a coluna `readmitted` em alvo binário:
	- 1 se "<30"
	- 0 caso contrário
	"""
	if source_col not in df.columns:
		raise ValueError(f"Coluna de origem do alvo '{source_col}' não encontrada no DataFrame.")
	df = df.copy()
	df[target_col] = (df[source_col].astype(str).str.strip() == "<30").astype(int)
	return df


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Normaliza tipos básicos e corrige colunas numéricas/categóricas comuns do dataset.
	- Converte age -> age_mid (ponto médio numeric)
	- Corrige gênero inválido
	- Garante tipos numéricos em colunas contínuas
	"""
	df = df.copy()

	# Idade para ponto médio
	if "age" in df.columns:
		df["age_mid"] = df["age"].apply(parse_age_to_midpoint)

	# Corrigir gênero inválido
	if "gender" in df.columns:
		df["gender"] = df["gender"].replace({"Unknown/Invalid": np.nan})

	# Colunas comumente numéricas no dataset
	numeric_like_cols = [
		"time_in_hospital",
		"num_lab_procedures",
		"num_procedures",
		"num_medications",
		"number_outpatient",
		"number_emergency",
		"number_inpatient",
		"number_diagnoses",
	]
	for c in numeric_like_cols:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors="coerce")

	return df


def clean_data(
	df: pd.DataFrame,
	make_binary_target: bool = True,
	source_target_col: str = "readmitted",
	binary_target_col: str = "readmit_30",
) -> pd.DataFrame:
	"""
	Limpeza e padronização inicial:
	- Remove duplicados
	- (Opcional) cria alvo binário a partir de `readmitted`
	- Padroniza tipos e valores inválidos
	"""
	df = df.copy()
	df = df.drop_duplicates()

	if make_binary_target:
		df = _make_binary_target(df, source_col=source_target_col, target_col=binary_target_col)

	df = standardize_types(df)
	return df


def load_and_clean(csv_path: str) -> pd.DataFrame:
	"""
	Atalho: carrega o CSV e aplica limpeza padrão, retornando DataFrame pronto
	para engenharia de atributos e modelagem.
	"""
	df = load_raw_data(csv_path)
	df = clean_data(df)
	return df


