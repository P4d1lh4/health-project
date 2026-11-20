from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional


def _has_kaggle_cli() -> bool:
	try:
		subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
		return True
	except Exception:
		return False


def _unzip_first_csv(zip_path: Path, target_csv_name: str, out_dir: Path) -> Optional[Path]:
	with zipfile.ZipFile(zip_path, "r") as zf:
		# tenta extrair exatamente o arquivo alvo
		names = zf.namelist()
		# busca por nome exato
		for name in names:
			if Path(name).name == target_csv_name:
				zf.extract(name, out_dir)
				extracted = out_dir / name
				# move para raiz de data/
				final_path = out_dir / target_csv_name
				if extracted != final_path:
					final_path.parent.mkdir(parents=True, exist_ok=True)
					shutil.move(str(extracted), str(final_path))
				return final_path
		# fallback: extrai tudo e procura
		zf.extractall(out_dir)
		for name in names:
			if Path(name).name == target_csv_name:
				extracted = out_dir / name
				final_path = out_dir / target_csv_name
				if extracted != final_path:
					shutil.move(str(extracted), str(final_path))
				return final_path
	return None


def ensure_diabetic_csv(
	target_path: str = "data/diabetic_data.csv",
	kaggle_dataset: Optional[str] = None,
	kaggle_url: Optional[str] = None,
) -> str:
	"""
	Garante que data/diabetic_data.csv exista. Tenta na ordem:
	1) Se já existir, retorna.
	2) Se `kaggle_dataset` fornecido (ex.: 'OWNER/DATASET'), usa Kaggle CLI para baixar apenas o CSV.
	3) Se `kaggle_url` fornecido (URL de dataset Kaggle), usa opendatasets para baixar e localizar o arquivo.

	Requer credenciais Kaggle configuradas em %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
	ou ~/.kaggle/kaggle.json (Linux/Mac).
	"""
	target = Path(target_path)
	if target.exists():
		return str(target)

	target.parent.mkdir(parents=True, exist_ok=True)

	# 2) Kaggle com slug explícito
	if kaggle_dataset:
		if _has_kaggle_cli():
			try:
				# baixa somente o arquivo desejado
				cmd = [
					"kaggle",
					"datasets",
					"download",
					"-d",
					kaggle_dataset,
					"-f",
					"diabetic_data.csv",
					"-p",
					str(target.parent),
					"--force",
				]
				subprocess.run(cmd, check=True)
				zips = sorted(target.parent.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
				if zips:
					found = _unzip_first_csv(zips[0], "diabetic_data.csv", target.parent)
					if found and Path(found).exists():
						return str(found)
			except Exception:
				# fallback para opendatasets
				pass
		# fallback usando opendatasets com URL construída a partir do slug
		try:
			import opendatasets as od  # type: ignore
			download_dir = target.parent
			url = f"https://www.kaggle.com/datasets/{kaggle_dataset}"
			od.download(url, data_dir=str(download_dir))
			for p in download_dir.rglob("diabetic_data.csv"):
				final_target = download_dir / "diabetic_data.csv"
				if p != final_target:
					shutil.move(str(p), str(final_target))
				return str(final_target)
		except Exception:
			pass

	# 3) opendatasets com URL de dataset
	if kaggle_url:
		try:
			import opendatasets as od  # type: ignore
			download_dir = target.parent
			od.download(kaggle_url, data_dir=str(download_dir))
			# procurar o arquivo baixado recursivamente
			for p in download_dir.rglob("diabetic_data.csv"):
				final_target = download_dir / "diabetic_data.csv"
				if p != final_target:
					shutil.move(str(p), str(final_target))
				return str(final_target)
		except Exception as e:
			raise RuntimeError(f"Falha ao baixar via opendatasets: {e}") from e

	raise FileNotFoundError(
		"Não foi possível obter 'data/diabetic_data.csv'. "
		"Forneça --kaggle_dataset OWNER/DATASET (ex.: 'OWNER/NOME') ou --kaggle_url de dataset Kaggle."
	)


