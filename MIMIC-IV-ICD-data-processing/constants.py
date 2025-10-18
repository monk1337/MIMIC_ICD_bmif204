PAD_CHAR = "**PAD**"
EMBEDDING_SIZE = 100
MAX_LENGTH = 2500

# Update these paths
MODEL_DIR = './models/'
DATA_DIR = './mimicdata/mimic4_icd10'
MIMIC_3_DIR = './mimicdata/mimic3'
MIMIC_2_DIR = './mimicdata/mimic2'
MIMIC_4_DIR = './mimicdata/mimic4_icd10'

import os
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)