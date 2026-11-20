# Projeto: Previsão de Readmissão Hospitalar (Diabetes)

Este projeto constrói um pipeline de Machine Learning para prever a probabilidade de um paciente ser readmitido em até 30 dias após a alta, a partir do dataset `diabetic_data.csv` (Kaggle/UCI).

## Estrutura

```
Health-Project/
├─ data/                    # coloque aqui o diabetic_data.csv
├─ models/                  # artefatos treinados (.joblib, metadata)
├─ reports/                 # métricas, importâncias, gráficos
├─ src/
│  ├─ __init__.py
│  ├─ data_loading.py
│  ├─ feature_engineering.py
│  └─ modeling.py
├─ scripts/
│  └─ train.py
├─ api/
│  └─ main.py               # FastAPI
├─ dashboard/
│  └─ app.py                # Streamlit
├─ requirements.txt
└─ README.md
```

## Setup (Windows/PowerShell)

```powershell
cd C:\Users\Site5\Health-Project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Coloque o dataset em `data/diabetic_data.csv`.

## Treino do Modelo

```powershell
python scripts/train.py --data_path data/diabetic_data.csv --test_size 0.2 --random_state 42
```

Saídas:
- `models/best_model.joblib`: pipeline completo (pré-processamento + classificador)
- `models/metadata.json`: metadados (modelo vencedor, colunas de entrada, versão)
- `reports/metrics.json`: métricas no conjunto de teste
- `reports/feature_importance.csv`: importâncias (permutation ou nativas)

### Baixar automaticamente do Kaggle

Este projeto suporta baixar `diabetic_data.csv` automaticamente, se não existir, via:
- Kaggle CLI com slug do dataset: `--kaggle_dataset OWNER/DATASET`
- `opendatasets` com URL do dataset: `--kaggle_url https://www.kaggle.com/datasets/OWNER/DATASET`

Exemplos:
```powershell
python scripts/train.py --kaggle_dataset OWNER/DATASET
# ou
python scripts/train.py --kaggle_url https://www.kaggle.com/datasets/OWNER/DATASET
```

Pré-requisito: configurar o token do Kaggle em `%USERPROFILE%\.kaggle\kaggle.json` (Windows) ou `~/.kaggle/kaggle.json`.

## API (FastAPI)

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health`
- `POST /predict` (JSON com chaves = colunas de entrada). Exemplo mínimo:

```json
{
  "age": "[60-70)",
  "time_in_hospital": 5,
  "num_lab_procedures": 40,
  "num_medications": 12,
  "number_outpatient": 1,
  "number_emergency": 0,
  "number_inpatient": 0,
  "insulin": "No",
  "metformin": "No",
  "glyburide": "No",
  "diag_1": "250.13",
  "diag_2": "401.9",
  "diag_3": "V45.81"
}
```

## Dashboard (Streamlit)

```powershell
streamlit run dashboard/app.py
```

Recursos:
- Visualizações rápidas (distribuição de alvo se dataset estiver em `data/`)
- Formulário para predição
- Exibição de métricas e importâncias se disponíveis em `reports/`

No dashboard, use o expansor "Baixar do Kaggle" na barra lateral para baixar o CSV (informe slug ou URL do dataset).

## Notas de Modelagem
- Alvo binário: `readmit_30 = 1` se `readmitted` == `<30`, caso contrário `0`.
- Engenharia de atributos:
  - `medication_ratio = num_medications / max(1, time_in_hospital)`
  - `prior_visits = number_outpatient + number_emergency + number_inpatient`
  - Mapeamento de diagnósticos (ICD) para grupos clínicos amplos
  - Conversão de `age` para ponto médio e faixas `[0–30], [30–60], [60+]`
- Modelos testados: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
- Métrica principal: ROC-AUC (com F1 e matriz de confusão auxiliares)

## Fonte do Dataset
- Kaggle: `Prediction on Hospital Readmission` (arquivo `diabetic_data.csv`)
- Origem UCI: Diabetes 130-US hospitals for years 1999–2008

## Boas Práticas
- Código modular (carregamento/limpeza, features, modelagem separados)
- Pipelines (`sklearn`) para garantir reprodutibilidade no deploy
- Importâncias por `permutation_importance` para modelo-agnóstico


