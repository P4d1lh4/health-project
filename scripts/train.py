import argparse
import os
import json
import sys
from pathlib import Path

# garante que o diretório raiz do projeto esteja no sys.path para importar 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_and_clean
from src.feature_engineering import engineer_features
from src.modeling import train_pipeline
from src.kaggle_utils import ensure_diabetic_csv


def main() -> None:
	parser = argparse.ArgumentParser(description="Treina modelos para previsão de readmissão hospitalar (<=30 dias).")
	parser.add_argument("--data_path", type=str, default="data/diabetic_data.csv", help="Caminho para o CSV de dados.")
	parser.add_argument("--test_size", type=float, default=0.2, help="Proporção do conjunto de teste.")
	parser.add_argument("--random_state", type=int, default=42, help="Semente aleatória.")
	parser.add_argument("--kaggle_dataset", type=str, default=None, help="Slug Kaggle do dataset (ex.: OWNER/DATASET).")
	parser.add_argument("--kaggle_url", type=str, default=None, help="URL do dataset no Kaggle (não use link de Code).")
	args = parser.parse_args()

	# Se o arquivo não existir, tenta baixar do Kaggle
	if not os.path.exists(args.data_path):
		ensure_diabetic_csv(
			target_path=args.data_path,
			kaggle_dataset=args.kaggle_dataset,
			kaggle_url=args.kaggle_url,
		)

	df = load_and_clean(args.data_path)
	df = engineer_features(df)

	metrics = train_pipeline(df, target_col="readmit_30", test_size=args.test_size, random_state=args.random_state)
	print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()


