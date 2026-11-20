import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import build_cohort, load_raw_data  # noqa: E402
from src.kaggle_utils import ensure_diabetic_csv  # noqa: E402
from src.modeling import train_and_save  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelos para previsão de readmissão hospitalar (<30 dias).")
    parser.add_argument("--data_path", type=str, default="data/diabetic_data.csv", help="Caminho para o CSV de dados.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporção de pacientes no conjunto de teste.")
    parser.add_argument("--random_state", type=int, default=42, help="Semente aleatória.")
    parser.add_argument("--model_dir", type=Path, default=PROJECT_ROOT / "models", help="Diretório dos artefatos do modelo.")
    parser.add_argument("--reports_dir", type=Path, default=PROJECT_ROOT / "reports", help="Diretório dos relatórios.")
    parser.add_argument("--kaggle_dataset", type=str, default=None, help="Slug Kaggle do dataset (ex.: OWNER/DATASET).")
    parser.add_argument("--kaggle_url", type=str, default=None, help="URL do dataset no Kaggle (não use link de Code).")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        ensure_diabetic_csv(
            target_path=args.data_path,
            kaggle_dataset=args.kaggle_dataset,
            kaggle_url=args.kaggle_url,
        )

    started = time.perf_counter()
    cohort = build_cohort(load_raw_data(args.data_path))
    metrics = train_and_save(
        cohort,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Tempo total de treino: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
