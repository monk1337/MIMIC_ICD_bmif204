#!/usr/bin/env python3
"""
Calculate model performance metrics stratified by clinical subgroups

This script:
1. Loads enriched evaluation data (from enrich_evaluation_data.py)
2. Calculates performance metrics for each clinical subgroup
3. Generates JSON files for the dashboard
4. Creates summary visualizations

Focus: 3 clinically meaningful factors for ICD coding
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Input: Enriched evaluation CSV
ENRICHED_CSV = "llm_eval_250_enriched.csv"

# Input: Model predictions (UPDATE with your actual prediction files)
# Assuming you have predictions in a format like:
# sample_id, predicted_codes, confidence_scores
PREDICTIONS_DIR = "model_predictions"  # Directory with prediction CSVs

# Output directory for JSON files (for dashboard)
OUTPUT_DIR = "dashboard_data_subgroups"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Models to evaluate
MODELS = [
    "GPT-4o-mini",
    "OpenBioLLM-70B", 
    "Mistral-7B",
    "Trained Att ConvNet",
    "Pretrained ConvNet"
]

print("="*80)
print("Clinical Subgroup Performance Analysis")
print("="*80)
print()

# ============================================================================
# Helper Functions
# ============================================================================

def parse_icd_codes(code_str):
    """Parse semicolon-separated ICD codes"""
    if pd.isna(code_str) or code_str == '':
        return []
    return [c.strip() for c in str(code_str).split(';') if c.strip()]

def calculate_precision_at_k(y_true, y_pred, k=10):
    """Calculate Precision@K"""
    if len(y_pred) == 0:
        return 0.0
    
    top_k = y_pred[:k]
    hits = len(set(top_k) & set(y_true))
    return hits / min(k, len(top_k))

def calculate_recall_at_k(y_true, y_pred, k=10):
    """Calculate Recall@K"""
    if len(y_true) == 0:
        return 0.0
    
    top_k = y_pred[:k]
    hits = len(set(top_k) & set(y_true))
    return hits / len(y_true)

def calculate_f1_at_k(y_true, y_pred, k=10):
    """Calculate F1@K"""
    prec = calculate_precision_at_k(y_true, y_pred, k)
    rec = calculate_recall_at_k(y_true, y_pred, k)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def calculate_multilabel_metrics(y_true_list, y_pred_list, threshold=0.5):
    """Calculate micro/macro F1, Precision, Recall for multi-label"""
    # This is a simplified version - adjust based on your prediction format
    # For multi-label, you typically use binary indicators for each code
    
    # For now, using set-based metrics
    all_metrics = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        if len(y_true) == 0:
            continue
            
        # Simple set-based metrics
        true_set = set(y_true)
        pred_set = set(y_pred[:15])  # Consider top 15 predictions
        
        if len(pred_set) > 0:
            prec = len(true_set & pred_set) / len(pred_set)
        else:
            prec = 0.0
            
        if len(true_set) > 0:
            rec = len(true_set & pred_set) / len(true_set)
        else:
            rec = 0.0
            
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
            
        all_metrics['precision'].append(prec)
        all_metrics['recall'].append(rec)
        all_metrics['f1'].append(f1)
    
    return {
        'f1_micro': np.mean(all_metrics['f1']),
        'prec_micro': np.mean(all_metrics['precision']),
        'rec_micro': np.mean(all_metrics['recall'])
    }

# ============================================================================
# Load Data
# ============================================================================

print("Loading enriched evaluation data...")
df = pd.read_csv(ENRICHED_CSV)
print(f"✓ Loaded {len(df)} samples")
print()

# Parse true labels
df['true_codes'] = df['labels'].apply(parse_icd_codes)
df['num_true_codes'] = df['true_codes'].apply(len)

# ============================================================================
# Mock Predictions (REPLACE WITH YOUR ACTUAL PREDICTIONS)
# ============================================================================

print("⚠️  NOTE: Using mock predictions for demonstration")
print("   Replace this section with your actual model predictions!")
print()

# For demonstration, create mock predictions
# In reality, load from your prediction files
np.random.seed(42)

for model in MODELS:
    # Mock: randomly select from true codes + some noise
    predictions = []
    for _, row in df.iterrows():
        true_codes = row['true_codes']
        # Mock: correct 50-70% of codes, add some random ones
        correct_ratio = np.random.uniform(0.5, 0.7)
        n_correct = int(len(true_codes) * correct_ratio)
        
        pred = true_codes[:n_correct].copy()
        # Add some random wrong codes
        n_wrong = max(0, 10 - len(pred))
        wrong_codes = [f"WRONG_{i}" for i in range(n_wrong)]
        pred.extend(wrong_codes)
        
        predictions.append(pred)
    
    df[f'{model}_pred'] = predictions

# ============================================================================
# Calculate Subgroup Performance
# ============================================================================

print("Calculating performance by clinical subgroups...")
print()

results = {}

for model in MODELS:
    print(f"Analyzing {model}...")
    
    model_results = {
        'model_name': model,
        'overall': {},
        'stratified': {
            'comorbidity': {},
            'admission_context': {},
            'race': {}
        }
    }
    
    # Overall performance
    y_true_all = df['true_codes'].tolist()
    y_pred_all = df[f'{model}_pred'].tolist()
    
    overall_p5 = np.mean([calculate_precision_at_k(yt, yp, 5) for yt, yp in zip(y_true_all, y_pred_all)])
    overall_p10 = np.mean([calculate_precision_at_k(yt, yp, 10) for yt, yp in zip(y_true_all, y_pred_all)])
    overall_r5 = np.mean([calculate_recall_at_k(yt, yp, 5) for yt, yp in zip(y_true_all, y_pred_all)])
    overall_r10 = np.mean([calculate_recall_at_k(yt, yp, 10) for yt, yp in zip(y_true_all, y_pred_all)])
    overall_f10 = np.mean([calculate_f1_at_k(yt, yp, 10) for yt, yp in zip(y_true_all, y_pred_all)])
    
    multilabel_metrics = calculate_multilabel_metrics(y_true_all, y_pred_all)
    
    model_results['overall'] = {
        'prec_at_5': overall_p5,
        'prec_at_10': overall_p10,
        'rec_at_5': overall_r5,
        'rec_at_10': overall_r10,
        'f1_at_10': overall_f10,
        **multilabel_metrics
    }
    
    # ========================================================================
    # Comorbidity Burden Stratification
    # ========================================================================
    
    for tier in ['low', 'medium', 'high']:
        subset = df[df['comorbidity_tier'] == tier]
        if len(subset) == 0:
            continue
        
        y_true = subset['true_codes'].tolist()
        y_pred = subset[f'{model}_pred'].tolist()
        
        p10 = np.mean([calculate_precision_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        r10 = np.mean([calculate_recall_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        f10 = np.mean([calculate_f1_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        
        model_results['stratified']['comorbidity'][tier] = {
            'f1_micro': f10,  # Using F1@10 as proxy
            'prec_at_10': p10,
            'rec_at_10': r10,
            'n_samples': len(subset)
        }
    
    # ========================================================================
    # Admission Context Stratification
    # ========================================================================
    
    for context in df['admission_context'].dropna().unique():
        subset = df[df['admission_context'] == context]
        if len(subset) == 0:
            continue
        
        y_true = subset['true_codes'].tolist()
        y_pred = subset[f'{model}_pred'].tolist()
        
        p10 = np.mean([calculate_precision_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        r10 = np.mean([calculate_recall_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        f10 = np.mean([calculate_f1_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        
        context_key = context.lower().replace('/', '_').replace(' ', '_')
        model_results['stratified']['admission_context'][context_key] = {
            'f1_micro': f10,
            'prec_at_10': p10,
            'rec_at_10': r10,
            'n_samples': len(subset)
        }
    
    # ========================================================================
    # Race/Ethnicity Stratification
    # ========================================================================
    
    for race in df['race_group'].dropna().unique():
        subset = df[df['race_group'] == race]
        if len(subset) < 5:  # Skip if too few samples
            continue
        
        y_true = subset['true_codes'].tolist()
        y_pred = subset[f'{model}_pred'].tolist()
        
        p10 = np.mean([calculate_precision_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        r10 = np.mean([calculate_recall_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        f10 = np.mean([calculate_f1_at_k(yt, yp, 10) for yt, yp in zip(y_true, y_pred)])
        
        race_key = race.lower().replace(' ', '_')
        model_results['stratified']['race'][race_key] = {
            'f1_micro': f10,
            'prec_at_10': p10,
            'rec_at_10': r10,
            'n_samples': len(subset)
        }
    
    results[model] = model_results
    
    print(f"  ✓ Overall P@10: {overall_p10:.3f}, R@10: {overall_r10:.3f}")
    print(f"  ✓ Comorbidity tiers: {list(model_results['stratified']['comorbidity'].keys())}")
    print(f"  ✓ Admission contexts: {list(model_results['stratified']['admission_context'].keys())}")
    print(f"  ✓ Race groups: {list(model_results['stratified']['race'].keys())}")
    print()

# ============================================================================
# Save Results to JSON
# ============================================================================

print("Saving results to JSON files...")

for model, model_results in results.items():
    # Create filename similar to existing format
    filename = f"eval_results_{model.lower().replace(' ', '_').replace('-', '_')}_subgroups.json"
    filepath = Path(OUTPUT_DIR) / filename
    
    with open(filepath, 'w') as f:
        json.dump(model_results, f, indent=2)
    
    print(f"  ✓ Saved {filename}")

print()

# ============================================================================
# Create Summary Report
# ============================================================================

print("="*80)
print("CLINICAL SUBGROUP ANALYSIS SUMMARY")
print("="*80)
print()

# Comorbidity Analysis
print("1️⃣  COMORBIDITY BURDEN (Task Difficulty)")
print("-" * 60)
print(f"{'Model':<25} {'Low (1-5)':<12} {'Med (6-10)':<12} {'High (11+)':<12}")
print("-" * 60)

for model in MODELS:
    model_res = results[model]['stratified']['comorbidity']
    low_f1 = model_res.get('low', {}).get('f1_micro', 0)
    med_f1 = model_res.get('medium', {}).get('f1_micro', 0)
    high_f1 = model_res.get('high', {}).get('f1_micro', 0)
    
    print(f"{model:<25} {low_f1:.3f}        {med_f1:.3f}        {high_f1:.3f}")

print()
print("Interpretation:")
print("  • Performance should DECREASE from Low → Medium → High")
print("  • This is EXPECTED: more comorbidities = harder task")
print("  • Graceful degradation indicates robust model")
print()

# Admission Context Analysis
print("2️⃣  ADMISSION CONTEXT (Documentation Quality)")
print("-" * 60)
print(f"{'Model':<25} {'Elective/Urgent':<20} {'Emergency':<20}")
print("-" * 60)

for model in MODELS:
    model_res = results[model]['stratified']['admission_context']
    elective_f1 = model_res.get('elective_urgent', {}).get('f1_micro', 0)
    emergency_f1 = model_res.get('emergency', {}).get('f1_micro', 0)
    
    gap = elective_f1 - emergency_f1
    gap_str = f"(Δ {gap:+.3f})"
    
    print(f"{model:<25} {elective_f1:.3f}              {emergency_f1:.3f}        {gap_str}")

print()
print("Interpretation:")
print("  • Elective admissions should perform BETTER (better documentation)")
print("  • Large gap suggests model relies on documentation completeness")
print("  • Small gap suggests model is robust to documentation quality")
print()

# Race/Ethnicity Analysis
print("3️⃣  RACE/ETHNICITY (Health Equity)")
print("-" * 60)
race_groups = ['white', 'black', 'asian', 'hispanic', 'other']
print(f"{'Model':<25} ", end='')
for race in race_groups:
    print(f"{race.capitalize():<10}", end=' ')
print()
print("-" * 60)

for model in MODELS:
    model_res = results[model]['stratified']['race']
    print(f"{model:<25} ", end='')
    
    for race in race_groups:
        f1 = model_res.get(race, {}).get('f1_micro', 0)
        if f1 > 0:
            print(f"{f1:.3f}     ", end=' ')
        else:
            print(f"{'N/A':<10}", end=' ')
    print()

print()
print("Interpretation:")
print("  • Differences SHOULD be minimal (ICD coding doesn't depend on race)")
print("  • Large disparities suggest SYSTEMIC BIAS in documentation")
print("  • NOT a model failure, but evidence of healthcare inequity")
print()

# ============================================================================
# Action Items
# ============================================================================

print("="*80)
print("NEXT STEPS FOR DASHBOARD")
print("="*80)
print()
print("1. Copy JSON files to dashboard public/data/ directory:")
print(f"   cp {OUTPUT_DIR}/*.json /path/to/dashboard/public/data/")
print()
print("2. Update Dashboard.js to add new views:")
print("   - 'Performance by Comorbidity Burden'")
print("   - 'Performance by Admission Context'")
print("   - 'Health Equity Analysis (Race)'")
print()
print("3. Update ChartView.js to handle new stratification keys:")
print("   - 'stratified-comorbidity'")
print("   - 'stratified-admission'")
print("   - 'stratified-race'")
print()
print("4. For presentation, focus on:")
print("   - Comorbidity: Shows clinical thinking (task difficulty)")
print("   - Admission: Shows understanding of real-world constraints")
print("   - Race: Shows commitment to health equity")
print()
print("✅ Clinical subgroup analysis complete!")
print()
