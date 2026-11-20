from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .data_loading import TARGET_COLUMN

MIN_GROUP_ROWS = 200
PER_THOUSAND = 1000


def bucket_top(series: pd.Series, top: int) -> pd.Series:
    """Contagens a partir de `top` viram um único grupo "top+", como categoria ordenada."""
    labels = [*(str(v) for v in range(int(series.min()), top)), f"{top}+"]
    text = series.clip(upper=top).map(lambda v: f"{top}+" if v == top else str(int(v)))
    return pd.Series(pd.Categorical(text, categories=labels, ordered=True), index=series.index)


def rate_by(cohort: pd.DataFrame, groups: pd.Series, min_rows: int = MIN_GROUP_ROWS) -> pd.DataFrame:
    """
    Uma linha por grupo com atendimentos, readmissões, taxa, participação nos
    atendimentos e lift (taxa do grupo / taxa geral). Grupos pequenos ficam de fora.
    """
    overall = cohort[TARGET_COLUMN].mean()
    table = (
        cohort[TARGET_COLUMN]
        .groupby(groups.rename("grupo"), observed=True)
        .agg(atendimentos="size", readmissoes="sum")
        .reset_index()
    )
    table = table.assign(
        grupo=table["grupo"].astype(str),
        taxa=table["readmissoes"] / table["atendimentos"],
        participacao=table["atendimentos"] / len(cohort),
    )
    table = table.assign(lift=table["taxa"] / overall)
    return table[table["atendimentos"] >= min_rows].reset_index(drop=True)


def segment_ranking(
    cohort: pd.DataFrame, dimensions: Mapping[str, pd.Series], min_rows: int
) -> pd.DataFrame:
    """Junta os grupos de várias dimensões e ordena do maior para o menor lift."""
    tables = [
        rate_by(cohort, groups, min_rows).assign(dimensao=name) for name, groups in dimensions.items()
    ]
    return pd.concat(tables, ignore_index=True).sort_values("lift", ascending=False).reset_index(drop=True)


def concentration(cohort: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    """Fração dos atendimentos e fração das readmissões que estão dentro de `mask`."""
    readmissions = cohort[TARGET_COLUMN].sum()
    share_of_rows = float(mask.mean())
    share_of_readmissions = float(cohort.loc[mask, TARGET_COLUMN].sum() / readmissions) if readmissions else 0.0
    return share_of_rows, share_of_readmissions


def risk_bands(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """
    Divide os atendimentos em faixas de mesmo tamanho pelo risco previsto
    (faixa 1 = menor risco). Serve para calibração (prob_media x taxa) e para
    medir quanto das readmissões cada faixa concentra.
    """
    frame = pd.DataFrame({"y": y_true, "proba": proba})
    # rank(method="first") evita faixas de tamanhos diferentes quando há probabilidades empatadas.
    band = (pd.qcut(frame["proba"].rank(method="first"), bins, labels=False) + 1).rename("faixa")
    table = (
        frame.groupby(band)
        .agg(atendimentos=("y", "size"), readmissoes=("y", "sum"), prob_media=("proba", "mean"), taxa=("y", "mean"))
        .reset_index()
    )
    return table.assign(
        participacao_readmissoes=table["readmissoes"] / frame["y"].sum(),
        lift=table["taxa"] / frame["y"].mean(),
    )


def threshold_for_share(proba: np.ndarray, share: float) -> float:
    """Limiar que sinaliza aproximadamente a fração `share` de maior risco."""
    return float(np.quantile(proba, 1 - share))


def threshold_summary(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Resultado de sinalizar quem tem probabilidade >= limiar, em frações e por mil altas."""
    flagged = proba >= threshold
    positives = y_true == 1
    hits = int((flagged & positives).sum())
    scale = PER_THOUSAND / len(y_true)
    return {
        "limiar": threshold,
        "sinalizados": float(flagged.mean()),
        "precisao": hits / int(flagged.sum()) if flagged.any() else float("nan"),
        "recall": hits / int(positives.sum()) if positives.any() else float("nan"),
        "alertas_por_mil": float(flagged.sum() * scale),
        "encontradas_por_mil": hits * scale,
        "falsos_alertas_por_mil": float((flagged & ~positives).sum() * scale),
        "perdidas_por_mil": float((~flagged & positives).sum() * scale),
    }


def threshold_curve(y_true: np.ndarray, proba: np.ndarray, shares: np.ndarray) -> pd.DataFrame:
    """Precisão e recall ao sinalizar cada fração `shares` de maior risco."""
    rows = [
        {"sinalizados_alvo": float(share), **threshold_summary(y_true, proba, threshold_for_share(proba, share))}
        for share in shares
    ]
    return pd.DataFrame(rows)
