import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import views  # noqa: E402
from dashboard.charts import decimal, integer, pct  # noqa: E402
from dashboard.patient import patient  # noqa: E402
from dashboard.views import Reports, tile  # noqa: E402
from src.artifacts import ModelArtifacts, artifacts_exist, load_artifacts  # noqa: E402
from src.data_loading import GROUP_COLUMN, TARGET_COLUMN, build_cohort, load_raw_data  # noqa: E402
from src.dimensions import build_dimensions  # noqa: E402
from src.kaggle_utils import ensure_diabetic_csv  # noqa: E402
from src.labels import feature_label  # noqa: E402
from src.modeling import IMPORTANCE_FILENAME, METRICS_FILENAME, PREDICTIONS_FILENAME  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH_DEFAULT = PROJECT_ROOT / "data" / "diabetic_data.csv"
FILTER_COLUMNS = ("age", "admission_type_id", "gender")
TRAIN_HINT = "Rode `python scripts/train.py` para gerar o modelo e os relatórios."

st.set_page_config(page_title="Readmissão hospitalar em 30 dias", layout="wide")


@st.cache_resource
def cached_artifacts() -> ModelArtifacts | None:
    return load_artifacts(MODEL_DIR) if artifacts_exist(MODEL_DIR) else None


@st.cache_data
def cached_cohort(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = build_cohort(load_raw_data(path))
    return cohort, build_dimensions(cohort)


def load_reports() -> Reports | None:
    paths = [REPORTS_DIR / name for name in (METRICS_FILENAME, IMPORTANCE_FILENAME, PREDICTIONS_FILENAME)]
    if not all(path.exists() for path in paths):
        return None
    return Reports(
        metrics=json.loads(paths[0].read_text(encoding="utf-8")),
        importances=pd.read_csv(paths[1]),
        predictions=pd.read_csv(paths[2]),
    )


def sidebar() -> str:
    st.sidebar.header("Fonte de dados")
    data_path = st.sidebar.text_input("Caminho do CSV", value=str(DATA_PATH_DEFAULT))
    with st.sidebar.expander("Baixar do Kaggle"):
        kaggle_dataset = st.text_input("Slug Kaggle (OWNER/DATASET)", value="")
        kaggle_url = st.text_input("URL do dataset Kaggle", value="")
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
    st.sidebar.header("Sobre")
    st.sidebar.caption(
        "Atendimentos de pacientes diabéticos em 130 hospitais dos EUA, de 1999 a 2008 (UCI). "
        "Altas por óbito ou hospice ficam fora da coorte. "
        "O modelo é avaliado em 20% dos pacientes, separados antes do treino."
    )
    return data_path


def filter_row(dimensions: pd.DataFrame) -> pd.Series:
    """Uma linha de filtros acima de tudo o que ela recorta; vazio significa todos."""
    mask = pd.Series(True, index=dimensions.index)
    for column, name in zip(st.columns(len(FILTER_COLUMNS)), map(feature_label, FILTER_COLUMNS)):
        options = sorted(dimensions[name].astype(str).unique())
        selected = column.multiselect(name, options, placeholder="Todos")
        if selected:
            mask &= dimensions[name].astype(str).isin(selected)
    return mask


def kpi_row(selected: pd.DataFrame, cohort: pd.DataFrame) -> None:
    rate, overall = selected[TARGET_COLUMN].mean(), cohort[TARGET_COLUMN].mean()
    filtered = len(selected) < len(cohort)
    delta = f"{(rate - overall) * 100:+.1f} p.p. vs. coorte".replace(".", ",", 1) if filtered else None
    tiles = st.columns(5)
    tile(tiles[0], "Atendimentos", integer(len(selected)))
    tile(tiles[1], "Pacientes", integer(selected[GROUP_COLUMN].nunique()))
    tile(tiles[2], "Readmissões", integer(selected[TARGET_COLUMN].sum()))
    tile(tiles[3], "Taxa de readmissão", pct(rate), delta=delta, delta_color="inverse")
    tile(tiles[4], "Internação média", f"{decimal(selected['time_in_hospital'].mean(), 1)} dias")


def main() -> None:
    data_path = sidebar()
    st.title("Readmissão hospitalar em 30 dias")
    st.caption("Onde o risco se concentra, o quanto o modelo acerta e o que muda conforme a capacidade de acompanhamento.")

    artifacts = cached_artifacts()
    reports = load_reports()
    cohort = dimensions = selected = selected_dimensions = None
    if Path(data_path).exists():
        cohort, dimensions = cached_cohort(data_path)
        mask = filter_row(dimensions)
        selected, selected_dimensions = cohort[mask], dimensions[mask]
        if selected.empty:
            st.warning("Nenhum atendimento com os filtros escolhidos.")
        else:
            kpi_row(selected, cohort)
            st.caption("Os filtros valem para Panorama e Fatores de risco. O modelo é sempre medido no conjunto de teste completo.")
    else:
        st.info("CSV não encontrado. Coloque o arquivo em `data/` ou informe o caminho na barra lateral.")

    panorama_tab, factors_tab, model_tab, threshold_tab, patient_tab = st.tabs(
        ["Panorama", "Fatores de risco", "Desempenho do modelo", "Limiar e capacidade", "Paciente"]
    )
    has_rows = selected is not None and not selected.empty
    with panorama_tab:
        if has_rows:
            views.panorama(selected, selected_dimensions)
    with factors_tab:
        if has_rows:
            views.risk_factors(selected, selected_dimensions)
    with model_tab:
        if reports is None:
            st.info(TRAIN_HINT)
        else:
            views.model_performance(reports)
    with threshold_tab:
        if reports is None or artifacts is None:
            st.info(TRAIN_HINT)
        else:
            views.threshold_and_capacity(reports, artifacts.threshold)
    with patient_tab:
        if artifacts is None:
            st.info(TRAIN_HINT)
        else:
            test_proba = reports.proba if reports else None
            prevalence = reports.metrics["test"]["prevalence"] if reports else None
            patient(artifacts, cohort, test_proba, prevalence)


if __name__ == "__main__":
    main()
