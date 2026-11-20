from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.charts import decimal, integer, pct
from src.analysis import (
    MIN_GROUP_ROWS,
    concentration,
    rate_by,
    risk_bands,
    segment_ranking,
    threshold_curve,
    threshold_for_share,
    threshold_summary,
)
from src.data_loading import TARGET_COLUMN
from src.dimensions import ORDERED_DIMENSIONS
from src.labels import feature_label

MAX_NOMINAL_GROUPS = 12
PANORAMA_DISCHARGE_GROUPS = 8
RANKING_ROWS = 10
RANKING_MIN_ROWS = 500
REPEAT_INPATIENT_VISITS = 2
CAPACITY_SCENARIOS = (0.05, 0.10, 0.20, 0.30, 0.50)


@dataclass(frozen=True)
class Reports:
    metrics: dict[str, Any]
    importances: pd.DataFrame
    predictions: pd.DataFrame

    @property
    def y_true(self) -> np.ndarray:
        return self.predictions["y_true"].to_numpy()

    @property
    def proba(self) -> np.ndarray:
        return self.predictions["proba"].to_numpy()


def tile(column: Any, label: str, value: str, help: str | None = None, **metric_args: Any) -> None:
    with column.container(border=True):
        st.metric(label, value, help=help, **metric_args)


def show_chart(chart: Any) -> None:
    st.altair_chart(chart, use_container_width=True, theme="streamlit")


def _min_rows(cohort: pd.DataFrame) -> int:
    """Com filtros a coorte encolhe; o mínimo por grupo acompanha, sem cair abaixo de 30."""
    return int(max(30, min(MIN_GROUP_ROWS, len(cohort) // 100)))


def _group_table(cohort: pd.DataFrame, groups: pd.Series, ordered: bool, limit: int) -> pd.DataFrame:
    table = rate_by(cohort, groups, _min_rows(cohort))
    if ordered:
        return table
    return table.nlargest(limit, "atendimentos").sort_values("taxa", ascending=False).reset_index(drop=True)


def _takeaway(table: pd.DataFrame, ordered: bool) -> str:
    if ordered:
        first, last = table.iloc[0], table.iloc[-1]
        return f"Vai de {pct(first['taxa'])} no grupo {first['grupo']} a {pct(last['taxa'])} no grupo {last['grupo']}."
    top, bottom = table.iloc[0], table.iloc[-1]
    return f"Maior taxa: {top['grupo']} ({pct(top['taxa'])}). Menor: {bottom['grupo']} ({pct(bottom['taxa'])})."


def _data_expander(table: pd.DataFrame) -> None:
    # Valores já formatados como texto: o st.dataframe não formata números no padrão brasileiro.
    view = pd.DataFrame(
        {
            "Grupo": table["grupo"],
            "Atendimentos": table["atendimentos"].map(integer),
            "Readmissões": table["readmissoes"].map(integer),
            "Taxa": table["taxa"].map(pct),
            "Participação": table["participacao"].map(pct),
            "Em relação à média": table["lift"].map(lambda v: f"{decimal(v, 2)}×"),
        }
    )
    with st.expander("Ver dados"):
        st.dataframe(view, hide_index=True, use_container_width=True)


def _rate_section(title: str, table: pd.DataFrame, overall: float, ordered: bool) -> None:
    st.markdown(f"##### {title}")
    if table.empty:
        st.info("Nenhum grupo tem atendimentos suficientes com os filtros atuais.")
        return
    st.caption(f"{_takeaway(table, ordered)} {_mean_note(overall)}")
    show_chart(charts.rate_chart(table, overall, horizontal=not ordered))
    _data_expander(table)


def _mean_note(overall: float) -> str:
    return f"A linha cinza é a taxa média, {pct(overall)}."


def panorama(cohort: pd.DataFrame, dimensions: pd.DataFrame) -> None:
    overall = cohort[TARGET_COLUMN].mean()
    share_rows, share_readmissions = concentration(
        cohort, cohort["number_inpatient"] >= REPEAT_INPATIENT_VISITS
    )
    st.markdown(
        f"Pacientes com duas ou mais internações no ano anterior são **{pct(share_rows)}** dos atendimentos "
        f"e respondem por **{pct(share_readmissions)}** das readmissões em 30 dias."
    )
    inpatient = feature_label("number_inpatient")
    discharge = feature_label("discharge_disposition_id")
    left, right = st.columns(2)
    with left:
        table = _group_table(cohort, dimensions[inpatient], ordered=True, limit=MAX_NOMINAL_GROUPS)
        _rate_section(f"Por {inpatient.lower()}", table, overall, ordered=True)
    with right:
        table = _group_table(cohort, dimensions[discharge], ordered=False, limit=PANORAMA_DISCHARGE_GROUPS)
        _rate_section(f"Por {discharge.lower()}", table, overall, ordered=False)


def risk_factors(cohort: pd.DataFrame, dimensions: pd.DataFrame) -> None:
    overall = cohort[TARGET_COLUMN].mean()
    name = st.selectbox("Variável", list(dimensions.columns))
    ordered = name in ORDERED_DIMENSIONS
    table = _group_table(cohort, dimensions[name], ordered, MAX_NOMINAL_GROUPS)
    if table.empty:
        st.info("Nenhum grupo tem atendimentos suficientes com os filtros atuais.")
    else:
        st.caption(f"{_takeaway(table, ordered)} {_mean_note(overall)}")
        left, right = st.columns(2)
        with left:
            show_chart(charts.rate_chart(table, overall, horizontal=not ordered))
        with right:
            show_chart(charts.share_chart(table, horizontal=not ordered))
        _data_expander(table)

    st.markdown("##### Segmentos de maior risco")
    min_rows = max(_min_rows(cohort), min(RANKING_MIN_ROWS, len(cohort) // 20))
    ranking = segment_ranking(cohort, {c: dimensions[c] for c in dimensions.columns}, min_rows)
    top = ranking.head(RANKING_ROWS)
    top = top.assign(grupo=top["dimensao"] + ": " + top["grupo"])
    if top.empty:
        st.info("Nenhum grupo tem atendimentos suficientes com os filtros atuais.")
        return
    st.caption(
        f"Os {len(top)} grupos de maior taxa entre todas as variáveis, com pelo menos {integer(min_rows)} atendimentos. "
        f"O primeiro tem {decimal(top.iloc[0]['lift'], 1)}× a taxa média. {_mean_note(overall)}"
    )
    show_chart(charts.rate_chart(top, overall, horizontal=True))
    _data_expander(top)


def model_performance(reports: Reports) -> None:
    test = reports.metrics["test"]
    bands = risk_bands(reports.y_true, reports.proba)
    top, bottom = bands.iloc[-1], bands.iloc[0]
    tiles = st.columns(4)
    tile(tiles[0], "ROC-AUC no teste", decimal(test["roc_auc"]), "Chance de o modelo dar risco maior a um readmitido do que a um não readmitido. 0,5 é acaso; 1,0 é perfeito.")
    tile(tiles[1], "PR-AUC no teste", decimal(test["pr_auc"]), f"Precisão média ao longo dos limiares. A referência é a prevalência, {pct(test['prevalence'])}.")
    tile(tiles[2], "Brier", decimal(test["brier"]), "Erro quadrático médio das probabilidades. Quanto menor, melhor.")
    tile(tiles[3], "Captura no top 10%", pct(top["participacao_readmissoes"]), "Parcela das readmissões do teste que cai na faixa de maior risco previsto.")

    left, right = st.columns(2)
    with left:
        st.markdown("##### O modelo ordena o risco")
        st.caption(
            f"Os 10% de maior risco têm taxa de {pct(top['taxa'])}, {decimal(top['lift'], 1)}× a média. "
            f"Os 10% de menor risco têm {pct(bottom['taxa'])}. {_mean_note(test['prevalence'])}"
        )
        show_chart(charts.risk_band_chart(bands, test["prevalence"]))
    with right:
        gap = (bands["prob_media"] - bands["taxa"]).abs().max()
        st.markdown("##### As probabilidades correspondem ao observado")
        st.caption(
            f"O maior desvio entre o previsto e o observado nas dez faixas é de {decimal(gap * 100, 1)} p.p. "
            "Sobre a linha cinza, a previsão seria perfeita."
        )
        show_chart(charts.calibration_chart(bands))

    best = reports.metrics["best_model"]
    left, right = st.columns(2)
    with left:
        st.markdown(f"##### {best} venceu por margem pequena")
        st.caption("Validação cruzada de 5 partições no treino, agrupada por paciente. As barras mostram um desvio-padrão.")
        show_chart(charts.model_comparison_chart(reports.metrics["cv_scores"], best))
    with right:
        first, second = reports.importances.iloc[0], reports.importances.iloc[1]
        st.markdown("##### Poucas colunas carregam o sinal")
        st.caption(
            f"{feature_label(first['feature'])} e {feature_label(second['feature']).lower()} ficam bem à frente das demais."
        )
        show_chart(charts.importance_chart(reports.importances))
    st.caption(
        f"Teste com {integer(reports.metrics['test_rows'])} atendimentos de {integer(reports.metrics['test_patients'])} pacientes "
        "que o modelo nunca viu. ROC-AUC entre 0,64 e 0,68 é o típico para este dataset."
    )


def threshold_and_capacity(reports: Reports, saved_threshold: float) -> None:
    y_true, proba = reports.y_true, reports.proba
    saved = threshold_summary(y_true, proba, saved_threshold)
    st.markdown(
        "O modelo dá um risco a cada alta. Quantas a equipe acompanha depende da capacidade, "
        "e essa escolha define quantas readmissões são encontradas."
    )
    capacity = st.slider(
        "Parcela das altas que a equipe consegue acompanhar (%)",
        min_value=5,
        max_value=60,
        value=int(round(saved["sinalizados"] * 100)),
    )
    share = capacity / 100
    chosen = threshold_summary(y_true, proba, threshold_for_share(proba, share))
    prevalence = float(y_true.mean())

    tiles = st.columns(4)
    tile(tiles[0], "Limiar de risco", pct(chosen["limiar"]), "Altas com risco previsto igual ou acima deste valor são sinalizadas.")
    tile(tiles[1], "Precisão", pct(chosen["precisao"]), "Dos sinalizados, quantos são readmitidos em 30 dias.")
    tile(tiles[2], "Recall", pct(chosen["recall"]), "Das readmissões, quantas estão entre os sinalizados.")
    tile(tiles[3], "Ganho sobre o acaso", f"{decimal(chosen['precisao'] / prevalence, 1)}×", "Precisão do modelo dividida pela taxa média de readmissão.")

    random_hits = chosen["alertas_por_mil"] * prevalence
    st.markdown(
        f"**A cada 1.000 altas:** {integer(chosen['alertas_por_mil'])} são sinalizadas e "
        f"{integer(chosen['encontradas_por_mil'])} delas são readmitidas. "
        f"Ficam de fora {integer(chosen['perdidas_por_mil'])} readmissões. "
        f"Escolhendo o mesmo número de pacientes ao acaso, seriam encontradas cerca de {integer(random_hits)}."
    )

    curve = threshold_curve(y_true, proba, np.arange(0.05, 0.61, 0.01))
    show_chart(charts.threshold_chart(curve, chosen["sinalizados"]))
    st.caption(
        f"A linha cinza marca a capacidade escolhida. O limiar salvo no modelo ({pct(saved_threshold)}) "
        f"maximiza o F1 e sinaliza {pct(saved['sinalizados'])} das altas."
    )

    scenarios = threshold_curve(y_true, proba, np.array(CAPACITY_SCENARIOS))
    with st.expander("Ver cenários de capacidade"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Altas sinalizadas": scenarios["sinalizados"].map(pct),
                    "Limiar": scenarios["limiar"].map(pct),
                    "Precisão": scenarios["precisao"].map(pct),
                    "Recall": scenarios["recall"].map(pct),
                    "Encontradas por mil altas": scenarios["encontradas_por_mil"].map(integer),
                    "Perdidas por mil altas": scenarios["perdidas_por_mil"].map(integer),
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
