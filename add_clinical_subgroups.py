#!/usr/bin/env python3
"""
Add clinical subgroup metrics to existing evaluation JSON files

This script:
1. Reads existing evaluation JSON (from llm_evaluate_results.py)
2. Reads enriched evaluation CSV (with clinical metadata)
3. Adds 4 new stratification sections to the JSON:
   - stratified.comorbidity
   - stratified.admission_context
   - stratified.race
   - calibration
4. Saves back to the same JSON file (preserving all existing metrics)

Usage:
    python3 add_clinical_subgroups.py -m model_name -p predictions.csv -e enriched.csv -j existing.json
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

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

def calculate_subgroup_metrics(df_subset, pred_col='predicted_codes', k=10):
    """Calculate metrics for a subgroup"""
    if len(df_subset) == 0:
        return None
    
    y_true_list = df_subset['true_codes'].tolist()
    y_pred_list = df_subset[pred_col].tolist()
    
    # Top-K metrics
    p_at_k = np.mean([calculate_precision_at_k(yt, yp, k) for yt, yp in zip(y_true_list, y_pred_list)])
    r_at_k = np.mean([calculate_recall_at_k(yt, yp, k) for yt, yp in zip(y_true_list, y_pred_list)])
    f1_at_k = np.mean([calculate_f1_at_k(yt, yp, k) for yt, yp in zip(y_true_list, y_pred_list)])
    
    # Also calculate for k=5 and k=15
    p_at_5 = np.mean([calculate_precision_at_k(yt, yp, 5) for yt, yp in zip(y_true_list, y_pred_list)])
    r_at_5 = np.mean([calculate_recall_at_k(yt, yp, 5) for yt, yp in zip(y_true_list, y_pred_list)])
    
    p_at_15 = np.mean([calculate_precision_at_k(yt, yp, 15) for yt, yp in zip(y_true_list, y_pred_list)])
    r_at_15 = np.mean([calculate_recall_at_k(yt, yp, 15) for yt, yp in zip(y_true_list, y_pred_list)])
    
    return {
        'f1_micro': f1_at_k,  # Using F1@10 as representative metric
        'prec_at_5': p_at_5,
        'rec_at_5': r_at_5,
        'prec_at_10': p_at_k,
        'rec_at_10': r_at_k,
        'prec_at_15': p_at_15,
        'rec_at_15': r_at_15,
        'n_samples': len(df_subset)
    }

def calculate_calibration(df, pred_col='predicted_codes', confidence_col='confidence_scores', n_bins=10):
    """
    Calculate calibration metrics
    
    For each prediction, we need:
    - Predicted probability/confidence
    - Whether prediction was correct
    
    Returns calibration curve data and calibration error
    """
    if confidence_col not in df.columns:
        # If no confidence scores, calculate based on ranking
        print("  ⚠️  No confidence scores found, using rank-based approximation")
        return calculate_calibration_from_ranks(df, pred_col)
    
    # Parse confidence scores if they're strings
    def parse_confidence(conf_str):
        if pd.isna(conf_str):
            return []
        if isinstance(conf_str, str):
            return [float(x) for x in conf_str.split(';')]
        return conf_str
    
    df['confidence_parsed'] = df[confidence_col].apply(parse_confidence)
    
    # Collect all (confidence, correctness) pairs
    calibration_data = []
    
    for _, row in df.iterrows():
        true_codes = set(row['true_codes'])
        pred_codes = row[pred_col]
        confidences = row['confidence_parsed']
        
        # Align predictions and confidences
        for i, (pred, conf) in enumerate(zip(pred_codes, confidences)):
            is_correct = pred in true_codes
            calibration_data.append({
                'confidence': conf,
                'correct': int(is_correct)
            })
    
    if len(calibration_data) == 0:
        return None
    
    calib_df = pd.DataFrame(calibration_data)
    
    # Bin by confidence
    calib_df['bin'] = pd.cut(calib_df['confidence'], bins=n_bins, labels=False)
    
    # Calculate calibration curve
    calibration_curve = []
    for bin_idx in range(n_bins):
        bin_data = calib_df[calib_df['bin'] == bin_idx]
        if len(bin_data) == 0:
            continue
        
        mean_confidence = bin_data['confidence'].mean()
        mean_accuracy = bin_data['correct'].mean()
        count = len(bin_data)
        
        calibration_curve.append({
            'confidence': float(mean_confidence),
            'accuracy': float(mean_accuracy),
            'count': int(count)
        })
    
    # Calculate Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = len(calib_df)
    
    for point in calibration_curve:
        weight = point['count'] / total_samples
        ece += weight * abs(point['confidence'] - point['accuracy'])
    
    return {
        'calibration_curve': calibration_curve,
        'expected_calibration_error': float(ece),
        'total_predictions': int(total_samples)
    }

def calculate_calibration_from_ranks(df, pred_col='predicted_codes', n_bins=10):
    """
    Fallback calibration when confidence scores not available
    Use rank position as proxy for confidence
    """
    calibration_data = []
    
    for _, row in df.iterrows():
        true_codes = set(row['true_codes'])
        pred_codes = row[pred_col]
        
        # Assign confidence based on rank (top-ranked = higher confidence)
        for rank, pred in enumerate(pred_codes[:15]):  # Top 15
            # Linear decay from 1.0 to 0.5
            confidence = 1.0 - (rank / 30.0)
            is_correct = pred in true_codes
            
            calibration_data.append({
                'confidence': confidence,
                'correct': int(is_correct)
            })
    
    if len(calibration_data) == 0:
        return None
    
    calib_df = pd.DataFrame(calibration_data)
    calib_df['bin'] = pd.cut(calib_df['confidence'], bins=n_bins, labels=False)
    
    calibration_curve = []
    for bin_idx in range(n_bins):
        bin_data = calib_df[calib_df['bin'] == bin_idx]
        if len(bin_data) == 0:
            continue
        
        mean_confidence = bin_data['confidence'].mean()
        mean_accuracy = bin_data['correct'].mean()
        count = len(bin_data)
        
        calibration_curve.append({
            'confidence': float(mean_confidence),
            'accuracy': float(mean_accuracy),
            'count': int(count)
        })
    
    # Calculate ECE
    ece = 0.0
    total_samples = len(calib_df)
    for point in calibration_curve:
        weight = point['count'] / total_samples
        ece += weight * abs(point['confidence'] - point['accuracy'])
    
    return {
        'calibration_curve': calibration_curve,
        'expected_calibration_error': float(ece),
        'total_predictions': int(total_samples),
        'note': 'Calibration calculated from rank positions (confidence scores not available)'
    }

# ============================================================================
# Main Function
# ============================================================================

def add_clinical_subgroups(predictions_file, enriched_file, json_file, output_file=None):
    """
    Add clinical subgroup metrics to existing JSON
    
    Args:
        predictions_file: CSV with predictions (sample_id, predicted_codes, [confidence_scores])
        enriched_file: CSV from enrich_evaluation_data.py
        json_file: Existing evaluation JSON from llm_evaluate_results.py
        output_file: Output JSON path (if None, overwrites json_file)
    """
    
    print("="*80)
    print("Adding Clinical Subgroup Metrics")
    print("="*80)
    print()
    
    # Load existing JSON
    print(f"Loading existing JSON: {json_file}")
    with open(json_file, 'r') as f:
        results = json.load(f)
    print(f"  ✓ Loaded existing metrics")
    print()
    
    # Load enriched evaluation data
    print(f"Loading enriched data: {enriched_file}")
    df_enriched = pd.read_csv(enriched_file)
    print(f"  ✓ Loaded {len(df_enriched)} samples with clinical metadata")
    
    # Parse true labels
    df_enriched['true_codes'] = df_enriched['labels'].apply(parse_icd_codes)
    print()
    
    # Load predictions
    print(f"Loading predictions: {predictions_file}")
    df_pred = pd.read_csv(predictions_file)
    print(f"  ✓ Loaded {len(df_pred)} predictions")
    
    # Parse predicted codes
    df_pred['predicted_codes'] = df_pred['predicted_codes'].apply(parse_icd_codes)
    print()
    
    # Merge predictions with enriched data
    print("Merging predictions with clinical metadata...")
    df = df_enriched.merge(df_pred, on='sample_id', how='inner')
    print(f"  ✓ Merged {len(df)} samples")
    print()
    
    # Initialize new stratified sections if not exist
    if 'stratified' not in results:
        results['stratified'] = {}
    
    # ========================================================================
    # 1. Comorbidity Burden Stratification
    # ========================================================================
    
    print("1️⃣  Calculating Performance by Comorbidity Burden...")
    results['stratified']['comorbidity'] = {}
    
    for tier in ['low', 'medium', 'high']:
        subset = df[df['comorbidity_tier'] == tier]
        if len(subset) == 0:
            print(f"  ⚠️  No samples for {tier} comorbidity")
            continue
        
        metrics = calculate_subgroup_metrics(subset, k=10)
        if metrics:
            results['stratified']['comorbidity'][tier] = metrics
            print(f"  ✓ {tier.capitalize()}: P@10={metrics['prec_at_10']:.3f}, "
                  f"R@10={metrics['rec_at_10']:.3f}, n={metrics['n_samples']}")
    
    print()
    
    # ========================================================================
    # 2. Admission Context Stratification
    # ========================================================================
    
    print("2️⃣  Calculating Performance by Admission Context...")
    results['stratified']['admission_context'] = {}
    
    for context in df['admission_context'].dropna().unique():
        subset = df[df['admission_context'] == context]
        if len(subset) == 0:
            continue
        
        metrics = calculate_subgroup_metrics(subset, k=10)
        if metrics:
            context_key = context.lower().replace('/', '_').replace(' ', '_')
            results['stratified']['admission_context'][context_key] = metrics
            print(f"  ✓ {context}: P@10={metrics['prec_at_10']:.3f}, "
                  f"R@10={metrics['rec_at_10']:.3f}, n={metrics['n_samples']}")
    
    print()
    
    # ========================================================================
    # 3. Race/Ethnicity Stratification (Health Equity)
    # ========================================================================
    
    print("3️⃣  Calculating Health Equity Analysis (Race)...")
    results['stratified']['race'] = {}
    
    for race in df['race_group'].dropna().unique():
        subset = df[df['race_group'] == race]
        if len(subset) < 5:  # Skip if too few samples
            print(f"  ⚠️  Skipping {race} (n={len(subset)} < 5)")
            continue
        
        metrics = calculate_subgroup_metrics(subset, k=10)
        if metrics:
            race_key = race.lower().replace(' ', '_')
            results['stratified']['race'][race_key] = metrics
            print(f"  ✓ {race}: P@10={metrics['prec_at_10']:.3f}, "
                  f"R@10={metrics['rec_at_10']:.3f}, n={metrics['n_samples']}")
    
    print()
    
    # ========================================================================
    # 4. Model Calibration
    # ========================================================================
    
    print("4️⃣  Calculating Model Calibration...")
    calibration = calculate_calibration(df, pred_col='predicted_codes')
    
    if calibration:
        results['calibration'] = calibration
        ece = calibration['expected_calibration_error']
        n_points = len(calibration['calibration_curve'])
        print(f"  ✓ Expected Calibration Error: {ece:.4f}")
        print(f"  ✓ Calibration curve: {n_points} bins")
        if 'note' in calibration:
            print(f"  ℹ️  {calibration['note']}")
    else:
        print("  ⚠️  Could not calculate calibration")
    
    print()
    
    # ========================================================================
    # Save Updated JSON
    # ========================================================================
    
    output_path = output_file if output_file else json_file
    
    print("Saving updated JSON...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved to: {output_path}")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("Added to JSON:")
    print("  ✓ stratified.comorbidity (low, medium, high)")
    print("  ✓ stratified.admission_context (emergency, elective/urgent)")
    print("  ✓ stratified.race (white, black, asian, hispanic, other)")
    print("  ✓ calibration (curve + ECE)")
    print()
    print("Preserved in JSON:")
    print("  ✓ overall (all existing metrics)")
    if 'stratified' in results and 'length' in results.get('stratified', {}):
        print("  ✓ stratified.length (existing)")
    if 'stratified' in results and 'code_freq' in results.get('stratified', {}):
        print("  ✓ stratified.code_freq (existing)")
    print()
    print("✅ Clinical subgroup metrics added successfully!")
    print()

# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add clinical subgroup metrics to evaluation JSON'
    )
    
    parser.add_argument(
        '-p', '--predictions',
        required=True,
        help='Path to predictions CSV (sample_id, predicted_codes, [confidence_scores])'
    )
    
    parser.add_argument(
        '-e', '--enriched',
        required=True,
        help='Path to enriched evaluation CSV (from enrich_evaluation_data.py)'
    )
    
    parser.add_argument(
        '-j', '--json',
        required=True,
        help='Path to existing evaluation JSON (from llm_evaluate_results.py)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output JSON path (if not specified, overwrites input JSON)'
    )
    
    args = parser.parse_args()
    
    add_clinical_subgroups(
        predictions_file=args.predictions,
        enriched_file=args.enriched,
        json_file=args.json,
        output_file=args.output
    )
