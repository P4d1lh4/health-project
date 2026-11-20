from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "readmit_30"
GROUP_COLUMN = "patient_nbr"

# "None" em A1Cresult/max_glu_serum significa "exame não realizado" e precisa
# sobreviver como categoria, por isso a lista padrão de NA do pandas é desligada.
MISSING_TOKENS = ("?", "Unknown/Invalid")

# Altas por óbito ou para hospice: o paciente não tem como ser readmitido.
DEATH_OR_HOSPICE_DISCHARGE_IDS = frozenset({11, 13, 14, 19, 20, 21})


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV bruto tratando apenas `?` e `Unknown/Invalid` como ausentes."""
    return pd.read_csv(
        csv_path,
        keep_default_na=False,
        na_values=list(MISSING_TOKENS),
        dtype={"diag_1": str, "diag_2": str, "diag_3": str},
        low_memory=False,
    )


def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove altas por óbito/hospice e adiciona o alvo `readmit_30`
    (1 se `readmitted == "<30"`). Mantém todos os atendimentos e `patient_nbr`
    para que a divisão treino/teste seja feita por paciente.
    """
    eligible = df.loc[~df["discharge_disposition_id"].isin(DEATH_OR_HOSPICE_DISCHARGE_IDS)]
    target = (eligible["readmitted"].astype(str).str.strip() == "<30").astype(int)
    return eligible.assign(**{TARGET_COLUMN: target}).reset_index(drop=True)
