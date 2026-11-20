from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loading import DEATH_OR_HOSPICE_DISCHARGE_IDS, build_cohort, load_raw_data


def test_none_stays_a_category_and_question_mark_is_missing(tmp_path: Path) -> None:
    csv = tmp_path / "amostra.csv"
    csv.write_text(
        "patient_nbr,race,gender,A1Cresult,max_glu_serum,diag_1\n"
        "1,?,Unknown/Invalid,None,None,250.01\n"
        "2,Caucasian,Female,>8,Norm,V45\n",
        encoding="utf-8",
    )
    df = load_raw_data(str(csv))

    assert df["A1Cresult"].tolist() == ["None", ">8"]
    assert df["max_glu_serum"].tolist() == ["None", "Norm"]
    assert pd.isna(df.loc[0, "race"])
    assert pd.isna(df.loc[0, "gender"])
    assert df["diag_1"].tolist() == ["250.01", "V45"]


def test_cohort_drops_death_and_hospice_and_builds_target() -> None:
    raw = pd.DataFrame(
        {
            "patient_nbr": [1, 1, 2, 3, 4, 5, 6, 7],
            "discharge_disposition_id": [1, 3, 11, 13, 14, 19, 20, 21],
            "readmitted": ["<30", ">30", "<30", "NO", "NO", "<30", "NO", "NO"],
        }
    )
    cohort = build_cohort(raw)

    assert not cohort["discharge_disposition_id"].isin(DEATH_OR_HOSPICE_DISCHARGE_IDS).any()
    assert cohort["discharge_disposition_id"].tolist() == [1, 3]
    assert cohort["readmit_30"].tolist() == [1, 0]
    assert cohort["patient_nbr"].tolist() == [1, 1]
    assert "readmit_30" not in raw.columns
