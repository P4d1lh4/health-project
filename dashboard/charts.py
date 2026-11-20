from __future__ import annotations

from typing import Mapping

import altair as alt
import pandas as pd

from src.labels import feature_label

# Paleta validada para daltonismo e contraste em fundo claro (azul/laranja e azul/vermelho).
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
ORANGE = "#eb6834"
RED = "#e34948"
MUTED = "#898781"

BAR_RADIUS = 4
BAND_PADDING = 0.35
CHART_HEIGHT = 340
ROW_HEIGHT = 30
# O Streamlit encaixa eixos e legenda dentro da altura pedida, então eles entram na conta.
AXIS_BAND = 70
LEGEND_BAND = 40
LABEL_LIMIT = 240
RAISES_RISK = "Aumenta o risco"
LOWERS_RISK = "Reduz o risco"


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}".replace(".", ",")


def integer(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _percent_axis(title: str | None) -> alt.Axis:
    return alt.Axis(format=".0%", title=title)


def _reference_line(value: float, horizontal: bool) -> alt.Chart:
    """Linha cinza de referência. O que ela marca vai na legenda do gráfico, para não cobrir as barras."""
    rule = alt.Chart(pd.DataFrame({"valor": [value]})).mark_rule(color=MUTED, strokeWidth=1.5)
    return rule.encode(x="valor:Q") if horizontal else rule.encode(y="valor:Q")


def _group_tooltips(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(
        tip_taxa=table["taxa"].map(pct),
        tip_atendimentos=table["atendimentos"].map(integer),
        tip_participacao=table["participacao"].map(pct),
        tip_lift=table["lift"].map(lambda v: f"{decimal(v, 2)}×"),
    )


_GROUP_TOOLTIP = [
    alt.Tooltip("grupo:N", title="Grupo"),
    alt.Tooltip("tip_taxa:N", title="Taxa de readmissão"),
    alt.Tooltip("tip_atendimentos:N", title="Atendimentos"),
    alt.Tooltip("tip_participacao:N", title="Participação"),
    alt.Tooltip("tip_lift:N", title="Em relação à média"),
]


def _group_bars(table: pd.DataFrame, field: str, title: str, color: str, horizontal: bool) -> alt.Chart:
    band = alt.Scale(paddingInner=BAND_PADDING)
    category = dict(sort=None, title=None, scale=band)
    if horizontal:
        encoding = dict(
            y=alt.Y("grupo:N", axis=alt.Axis(labelLimit=LABEL_LIMIT), **category),
            x=alt.X(f"{field}:Q", axis=_percent_axis(title)),
        )
    else:
        encoding = dict(
            x=alt.X("grupo:N", axis=alt.Axis(labelAngle=0, labelOverlap=True), **category),
            y=alt.Y(f"{field}:Q", axis=_percent_axis(title)),
        )
    return (
        alt.Chart(_group_tooltips(table))
        .mark_bar(color=color, cornerRadiusEnd=BAR_RADIUS)
        .encode(tooltip=_GROUP_TOOLTIP, **encoding)
    )


def _height(rows: int, horizontal: bool) -> int:
    return AXIS_BAND + ROW_HEIGHT * rows if horizontal else CHART_HEIGHT


def rate_chart(table: pd.DataFrame, overall: float, horizontal: bool) -> alt.LayerChart:
    """Taxa de readmissão por grupo, com a taxa geral como linha de referência."""
    bars = _group_bars(table, "taxa", "Taxa de readmissão em 30 dias", BLUE, horizontal)
    return (bars + _reference_line(overall, horizontal)).properties(height=_height(len(table), horizontal))


def share_chart(table: pd.DataFrame, horizontal: bool) -> alt.Chart:
    """Participação de cada grupo nos atendimentos, na mesma ordem do gráfico de taxa."""
    bars = _group_bars(table, "participacao", "Participação nos atendimentos", BLUE_LIGHT, horizontal)
    return bars.properties(height=_height(len(table), horizontal))


def risk_band_chart(bands: pd.DataFrame, overall: float) -> alt.LayerChart:
    """Taxa observada em cada faixa de risco previsto (10 = maior risco)."""
    data = bands.assign(
        tip_taxa=bands["taxa"].map(pct),
        tip_prob=bands["prob_media"].map(pct),
        tip_share=bands["participacao_readmissoes"].map(pct),
        tip_n=bands["atendimentos"].map(integer),
    )
    bars = (
        alt.Chart(data)
        .mark_bar(color=BLUE, cornerRadiusEnd=BAR_RADIUS)
        .encode(
            x=alt.X(
                "faixa:O",
                title="Faixa de risco previsto (1 = menor, 10 = maior)",
                axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(paddingInner=BAND_PADDING),
            ),
            y=alt.Y("taxa:Q", axis=_percent_axis("Taxa observada de readmissão")),
            tooltip=[
                alt.Tooltip("faixa:O", title="Faixa"),
                alt.Tooltip("tip_taxa:N", title="Taxa observada"),
                alt.Tooltip("tip_prob:N", title="Risco previsto médio"),
                alt.Tooltip("tip_share:N", title="Parcela das readmissões"),
                alt.Tooltip("tip_n:N", title="Atendimentos"),
            ],
        )
    )
    return (bars + _reference_line(overall, horizontal=False)).properties(height=CHART_HEIGHT)


def calibration_chart(bands: pd.DataFrame) -> alt.LayerChart:
    """Risco previsto contra taxa observada por faixa; sobre a diagonal, a previsão está certa."""
    limit = float(max(bands["prob_media"].max(), bands["taxa"].max()) * 1.08)
    scale = alt.Scale(domain=[0, limit], nice=False)
    data = bands.assign(tip_prob=bands["prob_media"].map(pct), tip_taxa=bands["taxa"].map(pct))
    diagonal = pd.DataFrame({"x": [0.0, limit], "y": [0.0, limit]})
    perfect = alt.Chart(diagonal).mark_line(color=MUTED, strokeWidth=1.5).encode(x="x:Q", y="y:Q")
    model = (
        alt.Chart(data)
        .mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(color=BLUE, size=70))
        .encode(
            x=alt.X("prob_media:Q", scale=scale, axis=_percent_axis("Risco previsto (média da faixa)")),
            y=alt.Y("taxa:Q", scale=scale, axis=_percent_axis("Taxa observada de readmissão")),
            tooltip=[
                alt.Tooltip("faixa:O", title="Faixa"),
                alt.Tooltip("tip_prob:N", title="Risco previsto"),
                alt.Tooltip("tip_taxa:N", title="Taxa observada"),
            ],
        )
    )
    return (perfect + model).properties(height=CHART_HEIGHT)


def model_comparison_chart(cv_scores: Mapping[str, Mapping[str, float]], best: str) -> alt.LayerChart:
    """ROC-AUC médio de cada candidato na validação cruzada, com um desvio-padrão para cada lado."""
    data = pd.DataFrame(
        [
            {
                "modelo": name,
                "media": scores["roc_auc_mean"],
                "minimo": scores["roc_auc_mean"] - scores["roc_auc_std"],
                "maximo": scores["roc_auc_mean"] + scores["roc_auc_std"],
                "destaque": "Escolhido" if name == best else "Descartado",
                "tip": f"{decimal(scores['roc_auc_mean'])} ± {decimal(scores['roc_auc_std'])}",
            }
            for name, scores in cv_scores.items()
        ]
    ).sort_values("media", ascending=False)
    color = alt.Color(
        "destaque:N", scale=alt.Scale(domain=["Escolhido", "Descartado"], range=[BLUE, MUTED]), legend=None
    )
    y = alt.Y("modelo:N", sort=None, title=None)
    tooltip = [alt.Tooltip("modelo:N", title="Modelo"), alt.Tooltip("tip:N", title="ROC-AUC na validação")]
    base = alt.Chart(data)
    interval = base.mark_rule(strokeWidth=2).encode(
        y=y,
        x=alt.X("minimo:Q", scale=alt.Scale(zero=False, padding=20), title="ROC-AUC na validação cruzada (média ± desvio)"),
        x2="maximo:Q",
        color=color,
        tooltip=tooltip,
    )
    mean = base.mark_point(filled=True, size=110, opacity=1).encode(y=y, x="media:Q", color=color, tooltip=tooltip)
    return (interval + mean).properties(height=_height(len(data), horizontal=True))


def importance_chart(importances: pd.DataFrame, top: int = 10) -> alt.Chart:
    """Quanto o ROC-AUC do teste cai ao embaralhar cada coluna de entrada."""
    data = importances.head(top).assign(
        fator=lambda df: df["feature"].map(feature_label),
        tip=lambda df: df["importance_mean"].map(lambda v: decimal(v, 4)),
    )
    return (
        alt.Chart(data)
        .mark_bar(color=BLUE, cornerRadiusEnd=BAR_RADIUS)
        .encode(
            y=alt.Y(
                "fator:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelLimit=LABEL_LIMIT),
                scale=alt.Scale(paddingInner=BAND_PADDING),
            ),
            x=alt.X("importance_mean:Q", title="Queda de ROC-AUC ao embaralhar a coluna"),
            tooltip=[alt.Tooltip("fator:N", title="Coluna"), alt.Tooltip("tip:N", title="Queda de ROC-AUC")],
        )
        .properties(height=_height(len(data), horizontal=True))
    )


def threshold_chart(curve: pd.DataFrame, chosen_share: float) -> alt.LayerChart:
    """Precisão e recall conforme a fração de altas sinalizadas."""
    names = {"precisao": "Precisão", "recall": "Recall"}
    data = curve.melt(id_vars="sinalizados", value_vars=list(names), var_name="metrica", value_name="valor")
    data = data.assign(
        metrica=data["metrica"].map(names),
        tip_valor=data["valor"].map(pct),
        tip_sinalizados=data["sinalizados"].map(pct),
    )
    color = alt.Color(
        "metrica:N",
        scale=alt.Scale(domain=list(names.values()), range=[BLUE, ORANGE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    x = alt.X(
        "sinalizados:Q",
        axis=alt.Axis(format=".0%", title="Altas sinalizadas para acompanhamento", tickCount=6),
    )
    y = alt.Y("valor:Q", axis=_percent_axis(None))
    base = alt.Chart(data)
    lines = base.mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    hover = base.mark_point(size=160, opacity=0).encode(
        x=x,
        y=y,
        tooltip=[
            alt.Tooltip("metrica:N", title="Métrica"),
            alt.Tooltip("tip_valor:N", title="Valor"),
            alt.Tooltip("tip_sinalizados:N", title="Altas sinalizadas"),
        ],
    )
    return (lines + hover + _reference_line(chosen_share, horizontal=True)).properties(height=CHART_HEIGHT)


def contribution_chart(contributions: pd.Series, top: int = 8) -> alt.LayerChart:
    """Atributos que mais empurraram a previsão de um paciente para cima ou para baixo."""
    data = pd.DataFrame(
        {
            "fator": [feature_label(name) for name in contributions.index[:top]],
            "contribuicao": contributions.to_numpy()[:top],
        }
    )
    data = data.assign(
        efeito=data["contribuicao"].map(lambda v: RAISES_RISK if v > 0 else LOWERS_RISK),
        tip=data["contribuicao"].map(lambda v: f"{v:+.2f}".replace(".", ",")),
    )
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=BAR_RADIUS)
        .encode(
            y=alt.Y(
                "fator:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelLimit=LABEL_LIMIT),
                scale=alt.Scale(paddingInner=BAND_PADDING),
            ),
            x=alt.X("contribuicao:Q", title="Efeito na previsão (log-odds)"),
            color=alt.Color(
                "efeito:N",
                scale=alt.Scale(domain=[RAISES_RISK, LOWERS_RISK], range=[RED, BLUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("fator:N", title="Atributo"),
                alt.Tooltip("efeito:N", title="Efeito"),
                alt.Tooltip("tip:N", title="Log-odds"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED, strokeWidth=1).encode(x="x:Q")
    return (bars + zero).properties(height=LEGEND_BAND + _height(len(data), horizontal=True))
