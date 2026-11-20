from __future__ import annotations

from typing import Mapping

import pandas as pd

MISSING_LABEL = "Não informado"

# Códigos do arquivo IDs_mapping.csv que acompanha o dataset. Os de óbito e hospice
# não aparecem porque saem da coorte.
DISCHARGE_LABELS: Mapping[int, str] = {
    1: "Casa",
    2: "Outro hospital de curta permanência",
    3: "Enfermagem especializada (SNF)",
    4: "Cuidados intermediários (ICF)",
    5: "Outra instituição de internação",
    6: "Casa com assistência domiciliar",
    7: "Saída contra recomendação médica",
    8: "Casa com infusão domiciliar",
    9: "Internado neste hospital",
    10: "Neonato para outro hospital",
    12: "Segue como paciente",
    15: "Leito de transição",
    16: "Ambulatório de outra instituição",
    17: "Ambulatório desta instituição",
    18: MISSING_LABEL,
    22: "Reabilitação",
    23: "Hospital de longa permanência",
    24: "Casa de repouso (Medicaid)",
    25: "Não mapeado",
    26: "Desconhecido",
    27: "Unidade federal de saúde",
    28: "Hospital psiquiátrico",
}

ADMISSION_TYPE_LABELS: Mapping[int, str] = {
    1: "Emergência",
    2: "Urgência",
    3: "Eletiva",
    4: "Recém-nascido",
    5: "Não disponível",
    6: MISSING_LABEL,
    7: "Centro de trauma",
    8: "Não mapeado",
}

_MEDICATION_DOSE = {"No": "Não usa", "Steady": "Dose mantida", "Up": "Dose aumentada", "Down": "Dose reduzida"}
_LAB_RESULT = {"None": "Não medido", "Norm": "Normal"}

# As 72 especialidades que aparecem no dataset. "DCPTEAM" é uma sigla sem
# significado documentado e fica como está.
MEDICAL_SPECIALTY_LABELS: Mapping[str, str] = {
    "AllergyandImmunology": "Alergia e imunologia",
    "Anesthesiology": "Anestesiologia",
    "Anesthesiology-Pediatric": "Anestesiologia pediátrica",
    "Cardiology": "Cardiologia",
    "Cardiology-Pediatric": "Cardiologia pediátrica",
    "DCPTEAM": "DCPTEAM",
    "Dentistry": "Odontologia",
    "Dermatology": "Dermatologia",
    "Emergency/Trauma": "Emergência e trauma",
    "Endocrinology": "Endocrinologia",
    "Endocrinology-Metabolism": "Endocrinologia e metabolismo",
    "Family/GeneralPractice": "Medicina de família e clínica geral",
    "Gastroenterology": "Gastroenterologia",
    "Gynecology": "Ginecologia",
    "Hematology": "Hematologia",
    "Hematology/Oncology": "Hematologia e oncologia",
    "Hospitalist": "Medicina hospitalar",
    "InfectiousDiseases": "Infectologia",
    "InternalMedicine": "Clínica médica",
    "Nephrology": "Nefrologia",
    "Neurology": "Neurologia",
    "Neurophysiology": "Neurofisiologia",
    "Obsterics&Gynecology-GynecologicOnco": "Ginecologia oncológica",
    "Obstetrics": "Obstetrícia",
    "ObstetricsandGynecology": "Ginecologia e obstetrícia",
    "Oncology": "Oncologia",
    "Ophthalmology": "Oftalmologia",
    "Orthopedics": "Ortopedia",
    "Orthopedics-Reconstructive": "Ortopedia reconstrutiva",
    "Osteopath": "Osteopatia",
    "Otolaryngology": "Otorrinolaringologia",
    "OutreachServices": "Serviços de extensão comunitária",
    "Pathology": "Patologia",
    "Pediatrics": "Pediatria",
    "Pediatrics-AllergyandImmunology": "Alergia e imunologia pediátrica",
    "Pediatrics-CriticalCare": "Terapia intensiva pediátrica",
    "Pediatrics-EmergencyMedicine": "Emergência pediátrica",
    "Pediatrics-Endocrinology": "Endocrinologia pediátrica",
    "Pediatrics-Hematology-Oncology": "Hematologia e oncologia pediátrica",
    "Pediatrics-InfectiousDiseases": "Infectologia pediátrica",
    "Pediatrics-Neurology": "Neurologia pediátrica",
    "Pediatrics-Pulmonology": "Pneumologia pediátrica",
    "Perinatology": "Perinatologia",
    "PhysicalMedicineandRehabilitation": "Medicina física e reabilitação",
    "PhysicianNotFound": "Médico não encontrado",
    "Podiatry": "Podologia",
    "Proctology": "Proctologia",
    "Psychiatry": "Psiquiatria",
    "Psychiatry-Addictive": "Psiquiatria de dependência química",
    "Psychiatry-Child/Adolescent": "Psiquiatria da infância e adolescência",
    "Psychology": "Psicologia",
    "Pulmonology": "Pneumologia",
    "Radiologist": "Radiologista",
    "Radiology": "Radiologia",
    "Resident": "Residente",
    "Rheumatology": "Reumatologia",
    "Speech": "Fonoaudiologia",
    "SportsMedicine": "Medicina esportiva",
    "Surgeon": "Cirurgião",
    "Surgery-Cardiovascular": "Cirurgia cardiovascular",
    "Surgery-Cardiovascular/Thoracic": "Cirurgia cardiovascular e torácica",
    "Surgery-Colon&Rectal": "Cirurgia colorretal",
    "Surgery-General": "Cirurgia geral",
    "Surgery-Maxillofacial": "Cirurgia bucomaxilofacial",
    "Surgery-Neuro": "Neurocirurgia",
    "Surgery-Pediatric": "Cirurgia pediátrica",
    "Surgery-Plastic": "Cirurgia plástica",
    "Surgery-PlasticwithinHeadandNeck": "Cirurgia plástica de cabeça e pescoço",
    "Surgery-Thoracic": "Cirurgia torácica",
    "Surgery-Vascular": "Cirurgia vascular",
    "SurgicalSpecialty": "Especialidade cirúrgica",
    "Urology": "Urologia",
}

VALUE_LABELS: Mapping[str, Mapping[str, str]] = {
    "medical_specialty": MEDICAL_SPECIALTY_LABELS,
    "gender": {"Female": "Feminino", "Male": "Masculino"},
    "race": {
        "Caucasian": "Branca",
        "AfricanAmerican": "Negra",
        "Hispanic": "Hispânica",
        "Asian": "Asiática",
        "Other": "Outra",
    },
    "A1Cresult": _LAB_RESULT,
    "max_glu_serum": _LAB_RESULT,
    "insulin": _MEDICATION_DOSE,
    "metformin": _MEDICATION_DOSE,
    "glyburide": _MEDICATION_DOSE,
    "glipizide": _MEDICATION_DOSE,
    "pioglitazone": _MEDICATION_DOSE,
    "rosiglitazone": _MEDICATION_DOSE,
    "change": {"Ch": "Houve mudança", "No": "Sem mudança"},
    "diabetesMed": {"Yes": "Sim", "No": "Não"},
}

DIAGNOSIS_GROUP_LABELS: Mapping[str, str] = {
    "circulatory": "Circulatório",
    "respiratory": "Respiratório",
    "digestive": "Digestivo",
    "diabetes": "Diabetes",
    "injury": "Lesão ou envenenamento",
    "musculoskeletal": "Musculoesquelético",
    "genitourinary": "Geniturinário",
    "neoplasms": "Neoplasias",
    "other": "Outros",
    "missing": MISSING_LABEL,
}

FEATURE_LABELS: Mapping[str, str] = {
    "age": "Faixa etária",
    "age_mid": "Idade",
    "gender": "Gênero",
    "race": "Raça",
    "admission_type_id": "Tipo de admissão",
    "discharge_disposition_id": "Destino da alta",
    "admission_source_id": "Origem da admissão",
    "time_in_hospital": "Dias de internação",
    "num_lab_procedures": "Exames laboratoriais",
    "num_procedures": "Procedimentos",
    "num_medications": "Medicamentos",
    "number_outpatient": "Consultas ambulatoriais no ano anterior",
    "number_emergency": "Emergências no ano anterior",
    "number_inpatient": "Internações no ano anterior",
    "number_diagnoses": "Número de diagnósticos",
    "A1Cresult": "Resultado de A1C",
    "max_glu_serum": "Glicose sérica",
    "insulin": "Insulina",
    "metformin": "Metformina",
    "glyburide": "Gliburida",
    "glipizide": "Glipizida",
    "pioglitazone": "Pioglitazona",
    "rosiglitazone": "Rosiglitazona",
    "change": "Mudança de medicação",
    "diabetesMed": "Medicação para diabetes",
    "diag_1": "Diagnóstico principal",
    "diag_2": "Diagnóstico secundário",
    "diag_3": "Diagnóstico adicional",
    "diag_1_group": "Diagnóstico principal",
    "diag_2_group": "Diagnóstico secundário",
    "diag_3_group": "Diagnóstico adicional",
    "medical_specialty": "Especialidade médica",
    "medication_ratio": "Medicamentos por dia de internação",
    "prior_visits": "Visitas no ano anterior (total)",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def value_label(column: str, value: object) -> str:
    """Rótulo legível de um valor bruto; valores sem tradução aparecem como estão."""
    if value is None or pd.isna(value):
        return MISSING_LABEL
    return VALUE_LABELS.get(column, {}).get(str(value), str(value))


def label_values(series: pd.Series, column: str) -> pd.Series:
    return series.map(lambda value: value_label(column, value))


def label_codes(series: pd.Series, labels: Mapping[int, str]) -> pd.Series:
    """Troca códigos numéricos pelo nome; código fora da tabela vira "Código N"."""
    return series.map(lambda code: MISSING_LABEL if pd.isna(code) else labels.get(int(code), f"Código {int(code)}"))
