#!/usr/bin/env python3
"""
Enrich ICD prediction evaluation data with clinical metadata from MIMIC-IV

This script joins your evaluation CSV with MIMIC tables to add:
1. Comorbidity burden (number of ICD codes per admission)
2. Admission context (emergency vs elective, documentation quality proxy)
3. Demographics (age, gender, race - for fairness analysis)

Output: Enriched CSV ready for clinical subgroup analysis in dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Path to your evaluation CSV (UPDATE THIS with actual path on server)
EVAL_CSV_PATH = "llm_eval_250_sample.csv"  # Change to actual path

# Path to extracted MIMIC data on server
MIMIC_BASE = "extracted_data/files/mimiciv/3.1"
HOSP_DIR = f"{MIMIC_BASE}/hosp"

# Output path
OUTPUT_PATH = "llm_eval_250_enriched.csv"

print("="*80)
print("MIMIC-IV Clinical Data Enrichment Script")
print("="*80)
print(f"Input: {EVAL_CSV_PATH}")
print(f"MIMIC Data: {MIMIC_BASE}")
print(f"Output: {OUTPUT_PATH}")
print()

# ============================================================================
# Step 1: Load Evaluation Data
# ============================================================================

print("Step 1: Loading evaluation data...")
df_eval = pd.read_csv(EVAL_CSV_PATH)
print(f"  ✓ Loaded {len(df_eval)} evaluation samples")
print(f"  Columns: {list(df_eval.columns)}")
print(f"  Unique patients (subject_id): {df_eval['subject_id'].nunique()}")
print(f"  Unique admissions (hadm_id): {df_eval['hadm_id'].nunique()}")
print()

# ============================================================================
# Step 2: Load MIMIC Core Tables
# ============================================================================

print("Step 2: Loading MIMIC tables...")

# Patients table (demographics)
print("  Loading patients.csv...")
patients = pd.read_csv(f"{HOSP_DIR}/patients.csv")
print(f"    ✓ {len(patients)} patients")

# Admissions table (admission details)
print("  Loading admissions.csv...")
admissions = pd.read_csv(f"{HOSP_DIR}/admissions.csv")
print(f"    ✓ {len(admissions)} admissions")

# Diagnoses table (for comorbidity calculation)
print("  Loading diagnoses_icd.csv...")
diagnoses = pd.read_csv(f"{HOSP_DIR}/diagnoses_icd.csv")
print(f"    ✓ {len(diagnoses)} diagnosis records")

print()

# ============================================================================
# Step 3: Calculate Comorbidity Burden
# ============================================================================

print("Step 3: Calculating comorbidity burden...")

# Count number of ICD codes per admission
comorbidity_counts = diagnoses.groupby('hadm_id').size().reset_index(name='num_comorbidities')

# Create tiers based on clinically meaningful thresholds
comorbidity_counts['comorbidity_tier'] = pd.cut(
    comorbidity_counts['num_comorbidities'],
    bins=[0, 5, 10, 999],
    labels=['low', 'medium', 'high'],
    include_lowest=True
)

print(f"  ✓ Calculated comorbidity burden for {len(comorbidity_counts)} admissions")
print(f"  Distribution:")
tier_dist = comorbidity_counts['comorbidity_tier'].value_counts()
for tier, count in tier_dist.items():
    pct = 100 * count / len(comorbidity_counts)
    print(f"    {tier}: {count} ({pct:.1f}%)")
print()

# ============================================================================
# Step 4: Process Demographics
# ============================================================================

print("Step 4: Processing demographics...")

# Calculate age at admission
# MIMIC uses anchor_age and anchor_year for privacy
# We'll use anchor_age as proxy (good enough for subgroups)
patients['age_group'] = pd.cut(
    patients['anchor_age'],
    bins=[0, 40, 65, 80, 150],
    labels=['<40', '40-65', '66-80', '>80'],
    include_lowest=True
)

print(f"  ✓ Created age groups")

# Gender is already in patients table
print(f"  ✓ Gender available: {patients['gender'].value_counts().to_dict()}")
print()

# ============================================================================
# Step 5: Process Admission Characteristics
# ============================================================================

print("Step 5: Processing admission characteristics...")

# Clean race categories
def standardize_race(race_str):
    """Standardize race categories into main groups"""
    if pd.isna(race_str):
        return 'Unknown'
    race_upper = str(race_str).upper()
    
    if 'WHITE' in race_upper:
        return 'White'
    elif 'BLACK' in race_upper or 'AFRICAN' in race_upper:
        return 'Black'
    elif 'ASIAN' in race_upper:
        return 'Asian'
    elif 'HISPANIC' in race_upper or 'LATINO' in race_upper:
        return 'Hispanic'
    else:
        return 'Other'

admissions['race_group'] = admissions['race'].apply(standardize_race)

# Admission type
admissions['is_emergency'] = admissions['admission_type'] == 'EMERGENCY'
admissions['admission_context'] = admissions['admission_type'].apply(
    lambda x: 'Emergency' if x == 'EMERGENCY' else 'Elective/Urgent'
)

# Insurance type (for fairness analysis, though we decided it's optional)
def categorize_insurance(ins_str):
    """Categorize insurance into main types"""
    if pd.isna(ins_str):
        return 'Unknown'
    ins_upper = str(ins_str).upper()
    
    if 'MEDICARE' in ins_upper:
        return 'Medicare'
    elif 'MEDICAID' in ins_upper:
        return 'Medicaid'
    elif 'OTHER' in ins_upper or 'PRIVATE' in ins_upper:
        return 'Private'
    else:
        return 'Other'

admissions['insurance_group'] = admissions['insurance'].apply(categorize_insurance)

# Length of stay (los is already in admissions as a column, but we'll create groups)
# Note: some versions of MIMIC need to calculate this from admit/discharge times
if 'los' not in admissions.columns:
    # Calculate LOS if not present
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    admissions['los'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / (24 * 3600)

# Create LOS groups
admissions['los_group'] = pd.cut(
    admissions['los'],
    bins=[0, 3, 7, 999],
    labels=['short', 'medium', 'long'],
    include_lowest=True
)

# Mortality flag
admissions['mortality'] = admissions['hospital_expire_flag'] == 1

print(f"  ✓ Race groups:")
for race, count in admissions['race_group'].value_counts().items():
    print(f"    {race}: {count}")
print(f"  ✓ Admission context:")
for ctx, count in admissions['admission_context'].value_counts().items():
    print(f"    {ctx}: {count}")
print()

# ============================================================================
# Step 6: Join Everything Together
# ============================================================================

print("Step 6: Joining data...")

# Start with evaluation data
df_enriched = df_eval.copy()

# Join demographics (patients)
df_enriched = df_enriched.merge(
    patients[['subject_id', 'gender', 'anchor_age', 'age_group']],
    on='subject_id',
    how='left'
)

# Join admission characteristics
df_enriched = df_enriched.merge(
    admissions[[
        'hadm_id', 
        'race_group', 
        'admission_context', 
        'is_emergency',
        'insurance_group', 
        'los', 
        'los_group',
        'mortality',
        'admission_type',
        'admission_location',
        'discharge_location'
    ]],
    on='hadm_id',
    how='left'
)

# Join comorbidity burden
df_enriched = df_enriched.merge(
    comorbidity_counts[['hadm_id', 'num_comorbidities', 'comorbidity_tier']],
    on='hadm_id',
    how='left'
)

print(f"  ✓ Enriched data shape: {df_enriched.shape}")
print(f"  ✓ New columns added: {len(df_enriched.columns) - len(df_eval.columns)}")
print()

# ============================================================================
# Step 7: Data Quality Checks
# ============================================================================

print("Step 7: Data quality checks...")

# Check for missing values in key fields
key_fields = ['gender', 'age_group', 'race_group', 'admission_context', 'comorbidity_tier']
missing_summary = []

for field in key_fields:
    missing = df_enriched[field].isna().sum()
    pct = 100 * missing / len(df_enriched)
    missing_summary.append(f"  {field}: {missing} missing ({pct:.1f}%)")
    if missing > 0:
        print(f"  ⚠️  {field}: {missing} missing ({pct:.1f}%)")
    else:
        print(f"  ✓ {field}: no missing values")

print()

# ============================================================================
# Step 8: Summary Statistics
# ============================================================================

print("Step 8: Summary statistics for clinical subgroups...")
print()

print("PRIORITY 1: Comorbidity Burden (affects task difficulty)")
print("-" * 60)
comorbidity_summary = df_enriched.groupby('comorbidity_tier').agg({
    'hadm_id': 'count',
    'num_comorbidities': ['mean', 'median', 'std']
}).round(2)
print(comorbidity_summary)
print()

print("PRIORITY 2: Admission Context (documentation quality proxy)")
print("-" * 60)
admission_summary = df_enriched['admission_context'].value_counts()
for ctx, count in admission_summary.items():
    pct = 100 * count / len(df_enriched)
    print(f"  {ctx}: {count} ({pct:.1f}%)")
print()

print("PRIORITY 3: Race/Ethnicity (health equity analysis)")
print("-" * 60)
race_summary = df_enriched['race_group'].value_counts()
for race, count in race_summary.items():
    pct = 100 * count / len(df_enriched)
    print(f"  {race}: {count} ({pct:.1f}%)")
print()

print("Additional Demographics (for reference)")
print("-" * 60)
print("Age Groups:")
age_summary = df_enriched['age_group'].value_counts().sort_index()
for age, count in age_summary.items():
    pct = 100 * count / len(df_enriched)
    print(f"  {age}: {count} ({pct:.1f}%)")
print()
print("Gender:")
gender_summary = df_enriched['gender'].value_counts()
for gender, count in gender_summary.items():
    pct = 100 * count / len(df_enriched)
    print(f"  {gender}: {count} ({pct:.1f}%)")
print()

# ============================================================================
# Step 9: Save Enriched Data
# ============================================================================

print("Step 9: Saving enriched data...")

# Reorder columns for better readability
# Put IDs and predictions first, then demographics, then clinical factors
column_order = [
    'sample_id', 'subject_id', 'hadm_id',
    # Original columns (if present)
    'text', 'labels', 
    # Demographics
    'gender', 'anchor_age', 'age_group', 'race_group',
    # Admission characteristics
    'admission_context', 'is_emergency', 'admission_type',
    # Comorbidity
    'num_comorbidities', 'comorbidity_tier',
    # Additional fields
    'insurance_group', 'los', 'los_group', 'mortality',
    # Original stratification (if present)
    'length', 'length_tier', 'code_frequency_tier', 'num_codes', 
    'stratification_group', 'dataset'
]

# Only keep columns that exist
final_columns = [col for col in column_order if col in df_enriched.columns]
# Add any remaining columns not in the order list
remaining_cols = [col for col in df_enriched.columns if col not in final_columns]
final_columns.extend(remaining_cols)

df_enriched = df_enriched[final_columns]

# Save to CSV
df_enriched.to_csv(OUTPUT_PATH, index=False)
print(f"  ✓ Saved enriched data to: {OUTPUT_PATH}")
print(f"  ✓ Rows: {len(df_enriched)}")
print(f"  ✓ Columns: {len(df_enriched.columns)}")
print()

# ============================================================================
# Step 10: Generate Data Dictionary
# ============================================================================

print("Step 10: Generating data dictionary...")

data_dict = {
    'Column': [],
    'Description': [],
    'Type': [],
    'Example Values': []
}

column_descriptions = {
    'sample_id': ('Sample identifier', 'ID', 'Integer'),
    'subject_id': ('MIMIC patient identifier', 'ID', 'Integer'),
    'hadm_id': ('MIMIC hospital admission identifier', 'ID', 'Integer'),
    'gender': ('Patient gender', 'Demographic', 'M/F'),
    'anchor_age': ('Patient age at anchor year', 'Demographic', '18-90+'),
    'age_group': ('Age category', 'Demographic', '<40, 40-65, 66-80, >80'),
    'race_group': ('Race/ethnicity category', 'Demographic', 'White, Black, Asian, Hispanic, Other'),
    'admission_context': ('Admission urgency level', 'Clinical', 'Emergency, Elective/Urgent'),
    'is_emergency': ('Boolean flag for emergency admission', 'Clinical', 'True/False'),
    'admission_type': ('Original MIMIC admission type', 'Clinical', 'EMERGENCY, URGENT, ELECTIVE'),
    'num_comorbidities': ('Number of ICD diagnosis codes', 'Clinical', '1-50+'),
    'comorbidity_tier': ('Comorbidity burden category', 'Clinical', 'low (1-5), medium (6-10), high (11+)'),
    'insurance_group': ('Insurance category', 'Socioeconomic', 'Medicare, Medicaid, Private, Other'),
    'los': ('Length of stay in days', 'Outcome', '0.5-60+'),
    'los_group': ('Length of stay category', 'Outcome', 'short (<3d), medium (3-7d), long (>7d)'),
    'mortality': ('Hospital mortality flag', 'Outcome', 'True/False'),
    'text': ('Discharge summary text', 'Input', 'Free text'),
    'labels': ('True ICD codes', 'Output', 'Semicolon-separated ICD codes'),
    'length': ('Text length in tokens/chars', 'Feature', 'Integer'),
    'length_tier': ('Text length category', 'Feature', 'short, medium, long'),
    'code_frequency_tier': ('Code rarity category', 'Feature', 'common, medium, rare'),
    'num_codes': ('Number of ICD codes', 'Feature', 'Integer')
}

for col in df_enriched.columns:
    if col in column_descriptions:
        desc, cat, example = column_descriptions[col]
        data_dict['Column'].append(col)
        data_dict['Description'].append(desc)
        data_dict['Type'].append(cat)
        data_dict['Example Values'].append(example)

df_dict = pd.DataFrame(data_dict)
dict_path = OUTPUT_PATH.replace('.csv', '_data_dictionary.csv')
df_dict.to_csv(dict_path, index=False)
print(f"  ✓ Saved data dictionary to: {dict_path}")
print()

# ============================================================================
# Summary and Next Steps
# ============================================================================

print("="*80)
print("ENRICHMENT COMPLETE!")
print("="*80)
print()
print("📊 KEY CLINICAL SUBGROUPS FOR ANALYSIS:")
print()
print("1️⃣  COMORBIDITY BURDEN (Task Difficulty)")
print("   - Low (1-5 codes): Simpler cases, expect high performance")
print("   - Medium (6-10 codes): Moderate complexity")
print("   - High (11+ codes): Complex patients, expect lower performance")
print()
print("2️⃣  ADMISSION CONTEXT (Documentation Quality)")
print("   - Emergency: Less complete notes, harder prediction")
print("   - Elective/Urgent: Better documentation, easier prediction")
print()
print("3️⃣  RACE/ETHNICITY (Health Equity)")
print("   - White, Black, Asian, Hispanic, Other")
print("   - Test for systemic bias in model performance")
print()
print("="*80)
print("NEXT STEPS:")
print("="*80)
print()
print("1. Load enriched data in your evaluation script:")
print(f"   df = pd.read_csv('{OUTPUT_PATH}')")
print()
print("2. Calculate model performance per subgroup:")
print("   for tier in ['low', 'medium', 'high']:")
print("       subset = df[df['comorbidity_tier'] == tier]")
print("       # Calculate P@10, R@10, F1, etc.")
print()
print("3. Create JSON files for dashboard:")
print("   results = {")
print("       'stratified': {")
print("           'comorbidity': {...},")
print("           'admission_context': {...},")
print("           'race': {...}")
print("       }")
print("   }")
print()
print("4. Update dashboard to show clinical subgroups!")
print()
print("✅ Ready for clinically meaningful analysis!")
print()
