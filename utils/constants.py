PAD_CHAR = "**PAD**"
EMBEDDING_SIZE = 100
MAX_LENGTH = 2500

import os

# Detect environment (Colab vs local)
if os.path.exists('/content'):
    # Google Colab paths
    BASE_DIR = '/content/MIMIC_ICD_bmif204/MIMIC-IV-ICD-data-processing'
    MODEL_DIR = f'{BASE_DIR}/models/'
    DATA_DIR = f'{BASE_DIR}/mimicdata/mimic4_icd10'
    MIMIC_3_DIR = f'{BASE_DIR}/mimicdata/mimic3'
    MIMIC_2_DIR = f'{BASE_DIR}/mimicdata/mimic2'
    MIMIC_4_DIR = f'{BASE_DIR}/mimicdata/mimic4_icd10'
else:
    # Local paths (relative)
    MODEL_DIR = './models/'
    DATA_DIR = './mimicdata/mimic4_icd10'
    MIMIC_3_DIR = './mimicdata/mimic3'
    MIMIC_2_DIR = './mimicdata/mimic2'
    MIMIC_4_DIR = './mimicdata/mimic4_icd10'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)