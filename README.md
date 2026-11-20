# Projeto: Previsão de Readmissão Hospitalar (Diabetes)

Este projeto constrói um pipeline de Machine Learning para estimar a probabilidade de um paciente ser readmitido em menos de 30 dias após a alta, a partir do dataset `diabetic_data.csv` (Kaggle/UCI).

## Estrutura

```
Health-Project/
├─ data/                    # coloque aqui o diabetic_data.csv
├─ models/                  # artefatos treinados (.joblib, metadata)
├─ reports/                 # métricas e importâncias
├─ src/
│  ├─ data_loading.py       # leitura do CSV e montagem da coorte
│  ├─ feature_engineering.py# contrato de entrada e engenharia de atributos
│  ├─ modeling.py           # divisão, validação cruzada, limiar, treino e artefatos
│  ├─ artifacts.py          # carga do modelo salvo e predição de um registro
│  └─ kaggle_utils.py       # download opcional do dataset
├─ scripts/
│  └─ train.py
├─ api/
│  └─ main.py               # FastAPI
├─ dashboard/
│  └─ app.py                # Streamlit
├─ notebooks/
│  ├─ 01_eda.ipynb          # exploração dos dados
│  └─ 02_modelo_shap.ipynb  # calibração, limiar e SHAP
├─ tests/                   # pytest
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

## Testes

```powershell
python -m pytest -q
```

Os testes usam dados sintéticos e treinam um modelo pequeno em diretório temporário; não dependem de `data/` nem de `models/`.

## Treino do Modelo

```powershell
python scripts/train.py --data_path data/diabetic_data.csv --test_size 0.2 --random_state 42
```

Saídas:
- `models/best_model.joblib`: pipeline completo (engenharia de atributos + pré-processamento + classificador). Recebe um DataFrame com as colunas brutas de `RAW_INPUT_COLUMNS`.
- `models/metadata.json`: modelo vencedor, colunas de entrada, limiar de decisão, versão do scikit-learn, contagens de linhas/pacientes e data do treino.
- `reports/metrics.json`: métricas de validação cruzada por candidato e métricas no conjunto de teste.
- `reports/feature_importance.csv`: importância por permutação (queda de ROC-AUC) de cada coluna de entrada, medida no teste.

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
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict`: JSON com as colunas brutas de `RAW_INPUT_COLUMNS` (em `src/feature_engineering.py`). Todos os campos são opcionais; campos ausentes são imputados pelo pipeline. Contagens negativas e chaves desconhecidas são rejeitadas com 422. A decisão usa o limiar salvo em `models/metadata.json`.

Exemplo:

```json
{
  "age": "[60-70)",
  "gender": "Female",
  "race": "Caucasian",
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "admission_source_id": 7,
  "time_in_hospital": 5,
  "num_lab_procedures": 40,
  "num_procedures": 1,
  "num_medications": 12,
  "number_outpatient": 1,
  "number_emergency": 0,
  "number_inpatient": 0,
  "number_diagnoses": 9,
  "A1Cresult": "None",
  "max_glu_serum": "None",
  "insulin": "Steady",
  "metformin": "No",
  "glyburide": "No",
  "glipizide": "No",
  "pioglitazone": "No",
  "rosiglitazone": "No",
  "change": "Ch",
  "diabetesMed": "Yes",
  "diag_1": "428",
  "diag_2": "250.01",
  "diag_3": "401",
  "medical_specialty": "InternalMedicine"
}
```

Resposta: `model`, `probability_readmit_30`, `threshold` e `prediction` (1 se a probabilidade for maior ou igual ao limiar).

## Dashboard (Streamlit)

```powershell
streamlit run dashboard/app.py
```

Recursos:
- Formulário de predição com todas as colunas de entrada do modelo, decidindo pelo limiar salvo
- ROC-AUC, PR-AUC, Brier e limiar do conjunto de teste, e importâncias por permutação
- EDA rápida se o dataset estiver em `data/`
- Expansor "Baixar do Kaggle" na barra lateral

## Notas de Modelagem
- Alvo binário: `readmit_30 = 1` se `readmitted == "<30"`, caso contrário `0`.
- Coorte: atendimentos com alta por óbito ou para hospice (`discharge_disposition_id` em 11, 13, 14, 19, 20, 21) são removidos, pois não podem gerar readmissão. Todos os demais atendimentos de cada paciente são mantidos.
- Valores ausentes: apenas `?` e `Unknown/Invalid` são tratados como ausentes. Em `A1Cresult` e `max_glu_serum`, `"None"` significa "exame não realizado" e é mantido como categoria própria. Categóricas ausentes viram a categoria `"missing"`; numéricas ausentes recebem a mediana.
- Engenharia de atributos (`engineer_features`, primeiro passo do pipeline salvo, então treino e inferência usam o mesmo código):
  - `age_mid`: ponto médio da faixa etária
  - `medication_ratio = num_medications / max(1, time_in_hospital)`
  - `prior_visits = number_outpatient + number_emergency + number_inpatient`
  - `diag_1`, `diag_2`, `diag_3` mapeados para grupos de ICD-9 (Strack et al., 2014)
  - `admission_type_id`, `discharge_disposition_id` e `admission_source_id` tratados como categorias
  - categorias com menos de 0,1% das linhas são agrupadas como infrequentes no one-hot
- Divisão: 20% dos pacientes vão para o teste (`StratifiedGroupKFold` por `patient_nbr`); nenhum paciente aparece em treino e teste.
- Modelos testados: Logistic Regression, Random Forest e XGBoost, sem pesos de classe para manter as probabilidades interpretáveis.
- Seleção: validação cruzada de 5 folds agrupada por paciente (`StratifiedGroupKFold`) no treino; vence a maior ROC-AUC média. A PR-AUC é registrada para cada candidato.
- Limiar: o que maximiza o F1 nas probabilidades out-of-fold do vencedor. O vencedor é reajustado no treino completo e avaliado uma única vez no teste.

## Resultados

Treino de 2026-11-20 com `python scripts/train.py` (padrões: `test_size=0.2`, `random_state=42`), cerca de 5 minutos em CPU de 12 threads.

Coorte: 99.343 atendimentos de 69.990 pacientes, após remover 2.423 altas por óbito/hospice. Divisão por paciente: treino com 79.625 atendimentos (55.952 pacientes) e teste com 19.718 atendimentos (14.038 pacientes), sem paciente em comum.

Validação cruzada agrupada no treino (5 folds, média ± desvio):

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0,662 ± 0,008 | 0,216 ± 0,010 |
| Random Forest | 0,671 ± 0,007 | 0,223 ± 0,011 |
| XGBoost (vencedor) | 0,675 ± 0,008 | 0,233 ± 0,012 |

Teste (XGBoost, avaliado uma vez):

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,680 |
| PR-AUC | 0,238 (prevalência 0,111) |
| Brier | 0,093 |
| Limiar (máx. F1 out-of-fold) | 0,137 |
| Precisão / recall / F1 no limiar | 0,203 / 0,472 / 0,284 |
| Matriz de confusão `[[VN, FP], [FN, VP]]` | `[[13486, 4044], [1155, 1033]]` |

As colunas mais importantes por permutação são `number_inpatient` (queda de 0,074 na ROC-AUC) e `discharge_disposition_id` (0,050); as demais ficam abaixo de 0,008.

ROC-AUC entre 0,64 e 0,68 é o típico para este dataset. O modelo ordena risco melhor que o acaso, mas não separa bem os casos: no limiar escolhido, cerca de 1 em cada 5 alertas é uma readmissão real. Um ROC-AUC bem acima de 0,72 aqui deve ser tratado como suspeita de vazamento, por exemplo o mesmo paciente em treino e teste.

## Notebooks

Os notebooks estão commitados com as saídas, então dá para ler sem executar.

- `notebooks/01_eda.ipynb` explora a coorte: alvo, valores ausentes, pacientes repetidos e taxa de readmissão por variável.
- `notebooks/02_modelo_shap.ipynb` avalia o modelo salvo no conjunto de teste: curva de calibração, efeito do limiar e explicação por SHAP. Rode o treino antes.

Para executar de novo:

```powershell
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_modelo_shap.ipynb
```

O que eles mostram:

- O histórico de internações e o destino da alta concentram o sinal. A taxa de readmissão vai de 8,6% sem internações no ano anterior a 37,1% com cinco ou mais, e de 9,3% na alta para casa a 27,7% na alta para reabilitação.
- As probabilidades estão bem calibradas: em dez faixas do teste, o previsto fica a menos de 2 pontos percentuais do observado.
- No SHAP, `number_inpatient` (0,231) e `discharge_disposition_id` (0,226) têm o dobro da média de |SHAP| do terceiro atributo, o que concorda com a importância por permutação.

## Fonte do Dataset
- Kaggle: `Prediction on Hospital Readmission` (arquivo `diabetic_data.csv`)
- Origem UCI: Diabetes 130-US hospitals for years 1999–2008
- Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International.
