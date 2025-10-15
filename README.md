# MIMIC-IV ICD-10 Data Processing

Process MIMIC-IV data for ICD-10 code prediction from discharge summaries.

## Prerequisites

- PhysioNet credentialed access to MIMIC-IV and MIMIC-IV-Note
- Python 3.6+ with pandas, numpy, nltk, tqdm, scipy
```bash
pip install pandas numpy nltk tqdm scipy
python -c "import nltk; nltk.download('punkt')"
```

## Quick Start

### 1. Download Data

Download these files from PhysioNet:

**From MIMIC-IV v2.2:**
- `hosp/diagnoses_icd.csv.gz`
- `hosp/procedures_icd.csv.gz`

**From MIMIC-IV-Note v2.2:**
- `note/discharge.csv.gz`

### 2. Setup Directory
```bash
# Extract files
gunzip *.csv.gz

# Organize structure
mimicdata/
└── physionet.org/files/mimiciv/2.2/
    ├── hosp/
    │   ├── diagnoses_icd.csv
    │   └── procedures_icd.csv
    └── note/
        └── discharge.csv
```

### 3. Run Processing
```bash
jupyter notebook notebooks/dataproc_mimic_IV_exploration_icd10.ipynb
# Run all cells (Cell → Run All)
# Takes ~30-60 minutes
```

### 4. Clean Up

Keep only final files:
```python
import os
import shutil

keep = {'train_full.csv', 'dev_full.csv', 'test_full.csv', 
        'train_50.csv', 'dev_50.csv', 'test_50.csv',
        'vocab.csv', 'TOP_50_CODES.csv', 'top50_icd10_code_list.txt'}

for f in os.listdir('./mimicdata/mimic4_icd10'):
    if f not in keep:
        os.remove(f'./mimicdata/mimic4_icd10/{f}')
```

### 5. Organize
```bash
cd mimicdata/mimic4_icd10
mkdir -p full_code top_50

mv train_full.csv dev_full.csv test_full.csv full_code/
mv train_50.csv dev_50.csv test_50.csv TOP_50_CODES.csv top50_icd10_code_list.txt top_50/
```

## Final Structure
```
mimic4_icd10/
├── full_code/
│   ├── train_full.csv  (110,442 samples, 26,788 codes)
│   ├── dev_full.csv    (4,017 samples)
│   └── test_full.csv   (7,851 samples)
├── top_50/
│   ├── train_50.csv    (~52k samples, 50 codes)
│   ├── dev_50.csv      (~1.7k samples)
│   ├── test_50.csv     (~3.3k samples)
│   ├── TOP_50_CODES.csv
│   └── top50_icd10_code_list.txt
└── vocab.csv           (69,972 terms)
```

## File Format

Each CSV contains:
- `subject_id`: Patient ID
- `hadm_id`: Admission ID  
- `text`: Preprocessed discharge summary
- `labels`: Semicolon-separated ICD-10 codes
- `length`: Token count

## Troubleshooting

**"discharge.csv not found"**: Download from MIMIC-IV-Note (separate dataset)
**Memory errors**: Need 16GB+ RAM or increase swap space
**"Error" during splitting**: Expected for unmapped HADM_IDs, ignore

# Preprocessed data
https://hu-my.sharepoint.com/:f:/g/personal/ankit_pal_fas_harvard_edu/EmA0M9hp1p5ArHXWMtc7slABml3BdFs0eBV4EfDgEPu0kg?e=Wxjbh0