from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np


def _parse_icd_as_float(code: Optional[str]) -> Optional[float]:
	if code is None or (isinstance(code, float) and np.isnan(code)):
		return None
	try:
		return float(str(code).strip())
	except Exception:
		return None


def map_icd_to_group(code: Optional[str]) -> str:
	"""
	Mapeia um código ICD para um grupo clínico amplo.
	Regras aproximadas usadas na literatura do dataset:
	- 390–459, 785: circulatory
	- 460–519, 786: respiratory
	- 520–579, 787: digestive
	- 250: diabetes
	- 800–999: injury
	- 710–739: musculoskeletal
	- 580–629: genitourinary
	- 140–239: neoplasms
	- E/V codes: other
	- caso contrário: other
	"""
	if code is None or (isinstance(code, float) and np.isnan(code)):
		return "other"
	s = str(code).strip().upper()

	# E/V codes
	if s.startswith("E") or s.startswith("V"):
		return "other"

	num = _parse_icd_as_float(s)
	if num is None:
		return "other"

	if 390 <= num <= 459 or num == 785:
		return "circulatory"
	if 460 <= num <= 519 or num == 786:
		return "respiratory"
	if 520 <= num <= 579 or num == 787:
		return "digestive"
	if 250 <= num < 251:
		return "diabetes"
	if 800 <= num <= 999:
		return "injury"
	if 710 <= num <= 739:
		return "musculoskeletal"
	if 580 <= num <= 629:
		return "genitourinary"
	if 140 <= num <= 239:
		return "neoplasms"
	return "other"


def add_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	for col in ["diag_1", "diag_2", "diag_3"]:
		if col in df.columns:
			df[f"{col}_group"] = df[col].apply(map_icd_to_group)
	return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()

	# medication_ratio
	if "num_medications" in df.columns:
		den = df["time_in_hospital"].replace(0, 1) if "time_in_hospital" in df.columns else 1
		df["medication_ratio"] = df["num_medications"] / den

	# prior_visits
	components = []
	for c in ["number_outpatient", "number_emergency", "number_inpatient"]:
		if c in df.columns:
			components.append(df[c].fillna(0))
	if components:
		df["prior_visits"] = sum(components)

	return df


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	# se age_mid não existir, tenta criar a partir de age
	if "age_mid" not in df.columns and "age" in df.columns:
		from .data_loading import parse_age_to_midpoint
		df["age_mid"] = df["age"].apply(parse_age_to_midpoint)

	# Faixas: [0–30], [30–60], [60+]
	def _bucket_age(x: Optional[float]) -> str:
		try:
			if x is None or np.isnan(x):
				return "unknown"
			if x < 30:
				return "0-30"
			if x < 60:
				return "30-60"
			return "60+"
		except Exception:
			return "unknown"

	if "age_mid" in df.columns:
		df["age_group_buckets"] = df["age_mid"].apply(_bucket_age)
	return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Pipeline de engenharia de atributos:
	- grupos de diagnóstico (ICD)
	- variáveis derivadas (medication_ratio, prior_visits)
	- faixas etárias
	"""
	df = add_diagnosis_groups(df)
	df = add_derived_features(df)
	df = add_age_groups(df)
	return df


