from __future__ import annotations

import numpy as np
import pandas as pd

AGES = ["[40-50)", "[50-60)", "[60-70)", "[70-80)"]
DIAGS = ["428", "250.01", "401", "V45", "786", None]


def make_raw_frame(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Frame sintético com o formato do CSV bruto; pacientes com vários atendimentos."""
    rng = np.random.default_rng(seed)
    inpatient = rng.integers(0, 4, n_rows)
    readmit_soon = rng.random(n_rows) < 0.05 + 0.08 * inpatient
    return pd.DataFrame(
        {
            "encounter_id": np.arange(n_rows),
            "patient_nbr": rng.integers(0, n_rows // 3, n_rows),
            "age": rng.choice(AGES, n_rows),
            "gender": rng.choice(["Female", "Male"], n_rows),
            "race": rng.choice(["Caucasian", "AfricanAmerican", None], n_rows),
            "admission_type_id": rng.integers(1, 4, n_rows),
            "discharge_disposition_id": rng.choice([1, 1, 1, 3, 6, 11, 13], n_rows),
            "admission_source_id": rng.choice([1, 7], n_rows),
            "time_in_hospital": rng.integers(1, 14, n_rows),
            "num_lab_procedures": rng.integers(1, 80, n_rows),
            "num_procedures": rng.integers(0, 6, n_rows),
            "num_medications": rng.integers(1, 30, n_rows),
            "number_outpatient": rng.integers(0, 3, n_rows),
            "number_emergency": rng.integers(0, 3, n_rows),
            "number_inpatient": inpatient,
            "number_diagnoses": rng.integers(1, 10, n_rows),
            "A1Cresult": rng.choice(["None", ">8", "Norm"], n_rows),
            "max_glu_serum": rng.choice(["None", ">200", "Norm"], n_rows),
            "insulin": rng.choice(["No", "Steady", "Up"], n_rows),
            "metformin": rng.choice(["No", "Steady"], n_rows),
            "glyburide": rng.choice(["No", "Steady"], n_rows),
            "glipizide": rng.choice(["No", "Steady"], n_rows),
            "pioglitazone": rng.choice(["No", "Steady"], n_rows),
            "rosiglitazone": rng.choice(["No", "Steady"], n_rows),
            "change": rng.choice(["No", "Ch"], n_rows),
            "diabetesMed": rng.choice(["Yes", "No"], n_rows),
            "diag_1": rng.choice(DIAGS, n_rows),
            "diag_2": rng.choice(DIAGS, n_rows),
            "diag_3": rng.choice(DIAGS, n_rows),
            "medical_specialty": rng.choice(["InternalMedicine", "Cardiology", None], n_rows),
            "readmitted": np.where(readmit_soon, "<30", rng.choice(["NO", ">30"], n_rows)),
        }
    )
