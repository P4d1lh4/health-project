import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.artifacts import ModelArtifacts, artifacts_exist, load_artifacts  # noqa: E402
from src.data_loading import load_raw_data  # noqa: E402
from src.feature_engineering import RAW_INPUT_SCHEMA, InputColumn, InputKind  # noqa: E402
from src.kaggle_utils import ensure_diabetic_csv  # noqa: E402
from src.modeling import IMPORTANCE_FILENAME, METRICS_FILENAME  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH_DEFAULT = PROJECT_ROOT / "data" / "diabetic_data.csv"
MISSING_OPTION = "(não informado)"

st.set_page_config(page_title="Previsão de Readmissão Hospitalar", layout="wide")


@st.cache_resource
def cached_artifacts() -> ModelArtifacts | None:
    return load_artifacts(MODEL_DIR) if artifacts_exist(MODEL_DIR) else None


@st.cache_data
def cached_dataset(path: str) -> pd.DataFrame:
    return load_raw_data(path)


def load_metrics() -> dict[str, Any] | None:
    path = REPORTS_DIR / METRICS_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_importances() -> pd.DataFrame | None:
    path = REPORTS_DIR / IMPORTANCE_FILENAME
    return pd.read_csv(path) if path.exists() else None


def category_options(column: InputColumn, df: pd.DataFrame | None) -> list[str]:
    if df is None or column.name not in df.columns:
        return [str(column.example)]
    values = df[column.name].dropna().astype(str).value_counts().index.tolist()
    return values if str(column.example) in values else [str(column.example), *values]


def input_widget(column: InputColumn, df: pd.DataFrame | None) -> Any:
    if column.kind is InputKind.COUNT:
        return st.sidebar.number_input(column.name, min_value=0, value=int(column.example), step=1)
    if column.kind is InputKind.CODE:
        return st.sidebar.number_input(column.name, min_value=1, value=int(column.example), step=1)
    options = [*category_options(column, df), MISSING_OPTION]
    choice = st.sidebar.selectbox(column.name, options, index=options.index(str(column.example)))
    return None if choice == MISSING_OPTION else choice


def prediction_form(df: pd.DataFrame | None) -> dict[str, Any]:
    st.sidebar.header("Entrada para Predição")
    return {column.name: input_widget(column, df) for column in RAW_INPUT_SCHEMA}


def kaggle_download_panel() -> None:
    with st.sidebar.expander("Baixar do Kaggle"):
        kaggle_dataset = st.text_input("Slug Kaggle (OWNER/DATASET)", value="", help="Ex.: owner/dataset-name")
        kaggle_url = st.text_input("URL do dataset Kaggle", value="", help="Use URL de dataset (não link de Code)")
        if st.button("Baixar CSV do Kaggle"):
            try:
                final_path = ensure_diabetic_csv(
                    target_path=str(DATA_PATH_DEFAULT),
                    kaggle_dataset=kaggle_dataset or None,
                    kaggle_url=kaggle_url or None,
                )
            except (RuntimeError, FileNotFoundError) as error:
                st.error(f"Falha no download: {error}")
            else:
                st.success(f"Baixado em: {final_path}")


def model_panel(artifacts: ModelArtifacts | None, metrics: dict[str, Any] | None) -> None:
    st.subheader("Status do Modelo")
    if artifacts is None:
        st.error("Modelo não encontrado. Execute o treino em scripts/train.py.")
        return
    st.success(f"Modelo carregado: {artifacts.model_name}")
    if metrics is None:
        return
    test = metrics["test"]
    cols = st.columns(4)
    cols[0].metric("ROC-AUC (teste)", f"{test['roc_auc']:.3f}")
    cols[1].metric("PR-AUC (teste)", f"{test['pr_auc']:.3f}", help=f"Prevalência: {test['prevalence']:.3f}")
    cols[2].metric("Brier (teste)", f"{test['brier']:.4f}")
    cols[3].metric("Limiar", f"{artifacts.threshold:.3f}")
    st.caption(
        f"No limiar: precisão {test['precision']:.3f}, recall {test['recall']:.3f}, F1 {test['f1']:.3f}. "
        f"{metrics['split']}."
    )


def main() -> None:
    st.title("Previsão de Readmissão Hospitalar (< 30 dias)")
    artifacts = cached_artifacts()

    st.sidebar.header("Dados para EDA / Kaggle")
    data_path = st.sidebar.text_input("Caminho do CSV", value=str(DATA_PATH_DEFAULT))
    kaggle_download_panel()
    df = cached_dataset(data_path) if Path(data_path).exists() else None

    col1, col2 = st.columns(2)
    with col1:
        model_panel(artifacts, load_metrics())
    with col2:
        st.subheader("Importância por Permutação (queda de ROC-AUC)")
        importances = load_importances()
        if importances is None or importances.empty:
            st.info("Importâncias não disponíveis.")
        else:
            st.dataframe(importances.head(20))

    st.subheader("EDA Rápida")
    if df is None:
        st.info("Coloque o CSV em data/ ou informe o caminho na barra lateral para EDA.")
    else:
        st.write("Distribuição do alvo (readmitted):")
        st.bar_chart(df["readmitted"].value_counts())
        st.write("Estatísticas descritivas:")
        st.dataframe(df[["time_in_hospital", "num_lab_procedures", "num_medications"]].describe().T)

    record = prediction_form(df)
    if st.sidebar.button("Prever"):
        if artifacts is None:
            st.error("Modelo não carregado.")
            return
        result = artifacts.predict(record)
        label = "readmissão provável" if result.prediction else "readmissão improvável"
        st.success(
            f"Probabilidade de readmissão (< 30 dias): {result.probability:.3f} "
            f"| limiar {result.threshold:.3f} | {label}"
        )


if __name__ == "__main__":
    main()
