#!/usr/bin/env python3
"""
Comprehensive Metrics Calculator for ICD Prediction Models

Calculates ALL metrics for dashboard in a single run:
- Overall metrics (F1 Micro/Macro, Precision, Recall, AUC, P@K, R@K, F1@K)
- Stratified by code frequency (common, medium, rare)
- Stratified by document length (short, medium, long)
- Stratified by comorbidity burden (low, medium, high)
- Stratified by admission context (emergency, elective)
- Stratified by race (white, black, asian, hispanic, other)
- Calibration metrics

Input: {model}_results.json (per-sample predictions)
Output: eval_results_{model}.json (dashboard-compatible format)
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Helper Functions
# ============================================================================

def normalize_icd_code(code):
    """Normalize ICD code by removing dots and standardizing format"""
    if pd.isna(code):
        return ""
    return str(code).replace('.', '').replace(' ', '').upper().strip()

def parse_icd_codes(code_list):
    """Parse list of ICD codes and normalize them"""
    if isinstance(code_list, str):
        # If it's a string, try to parse as list
        try:
            code_list = json.loads(code_list)
        except:
            code_list = [c.strip() for c in code_list.split(';') if c.strip()]
    
    if not isinstance(code_list, list):
        return []
    
    return [normalize_icd_code(c) for c in code_list if c]

def calculate_precision_at_k(y_true, y_pred, k):
    """Calculate Precision@K"""
    if len(y_pred) == 0:
        return 0.0
    
    top_k = y_pred[:k]
    true_set = set(y_true)
    hits = len([p for p in top_k if p in true_set])
    return hits / min(k, len(top_k))

def calculate_recall_at_k(y_true, y_pred, k):
    """Calculate Recall@K"""
    if len(y_true) == 0:
        return 0.0
    
    top_k = y_pred[:k]
    true_set = set(y_true)
    hits = len([p for p in top_k if p in true_set])
    return hits / len(y_true)

def calculate_f1_at_k(y_true, y_pred, k):
    """Calculate F1@K"""
    prec = calculate_precision_at_k(y_true, y_pred, k)
    rec = calculate_recall_at_k(y_true, y_pred, k)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def calculate_multilabel_metrics(df, all_codes):
    """
    Calculate multilabel classification metrics (micro/macro)
    
    Args:
        df: DataFrame with 'true_codes' and 'predicted_codes' columns
        all_codes: Set of all possible ICD codes
    
    Returns:
        dict with f1_micro, f1_macro, prec_micro, prec_macro, rec_micro, rec_macro,
        auc_micro, auc_macro
    """
    # Create binary label matrices
    n_samples = len(df)
    n_codes = len(all_codes)
    code_to_idx = {code: idx for idx, code in enumerate(sorted(all_codes))}
    
    y_true = np.zeros((n_samples, n_codes), dtype=int)
    y_pred = np.zeros((n_samples, n_codes), dtype=int)
    y_score = np.zeros((n_samples, n_codes), dtype=float)
    
    for sample_idx, (_, row) in enumerate(df.iterrows()):
        true_codes = row['true_codes']
        pred_codes = row['predicted_codes']
        
        # True labels
        for code in true_codes:
            if code in code_to_idx:
                y_true[sample_idx, code_to_idx[code]] = 1
        
        # Predicted labels (binary: top 15 predictions)
        for rank, code in enumerate(pred_codes[:15]):
            if code in code_to_idx:
                y_pred[sample_idx, code_to_idx[code]] = 1
                # Score based on rank (higher rank = lower score)
                y_score[sample_idx, code_to_idx[code]] = 1.0 - (rank / 30.0)
    
    # Calculate metrics
    metrics = {}
    
    # F1, Precision, Recall (Micro/Macro)
    try:
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['prec_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['prec_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['rec_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['rec_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"  ⚠️  Error calculating F1/Precision/Recall: {e}")
        metrics.update({
            'f1_micro': 0.0, 'f1_macro': 0.0,
            'prec_micro': 0.0, 'prec_macro': 0.0,
            'rec_micro': 0.0, 'rec_macro': 0.0
        })
    
    # AUC (Micro/Macro) - requires scores
    try:
        # Only calculate AUC for codes that appear in true labels
        valid_cols = y_true.sum(axis=0) > 0
        if valid_cols.sum() > 0:
            y_true_valid = y_true[:, valid_cols]
            y_score_valid = y_score[:, valid_cols]
            
            # Micro AUC
            metrics['auc_micro'] = roc_auc_score(y_true_valid.ravel(), y_score_valid.ravel())
            
            # Macro AUC (average of per-class AUCs)
            auc_scores = []
            for col_idx in range(y_true_valid.shape[1]):
                if len(np.unique(y_true_valid[:, col_idx])) > 1:  # Need both classes
                    try:
                        auc_col = roc_auc_score(y_true_valid[:, col_idx], y_score_valid[:, col_idx])
                        auc_scores.append(auc_col)
                    except:
                        pass
            
            metrics['auc_macro'] = np.mean(auc_scores) if auc_scores else 0.0
        else:
            metrics['auc_micro'] = 0.0
            metrics['auc_macro'] = 0.0
    except Exception as e:
        print(f"  ⚠️  Error calculating AUC: {e}")
        metrics['auc_micro'] = 0.0
        metrics['auc_macro'] = 0.0
    
    return metrics

def calculate_topk_metrics(df, k_values=[5, 8, 15]):
    """Calculate Precision@K, Recall@K, F1@K for multiple K values"""
    metrics = {}
    
    for k in k_values:
        prec_list = []
        rec_list = []
        f1_list = []
        
        for _, row in df.iterrows():
            true_codes = row['true_codes']
            pred_codes = row['predicted_codes']
            
            prec_list.append(calculate_precision_at_k(true_codes, pred_codes, k))
            rec_list.append(calculate_recall_at_k(true_codes, pred_codes, k))
            f1_list.append(calculate_f1_at_k(true_codes, pred_codes, k))
        
        metrics[f'prec_at_{k}'] = np.mean(prec_list)
        metrics[f'rec_at_{k}'] = np.mean(rec_list)
        metrics[f'f1_at_{k}'] = np.mean(f1_list)
    
    return metrics

def calculate_stratified_metrics(df, stratify_col, all_codes):
    """Calculate metrics for each stratum"""
    stratified = {}
    
    for stratum_value in df[stratify_col].dropna().unique():
        subset = df[df[stratify_col] == stratum_value]
        
        if len(subset) < 5:  # Skip if too few samples
            continue
        
        # Overall multilabel metrics
        ml_metrics = calculate_multilabel_metrics(subset, all_codes)
        
        # Top-K metrics
        topk_metrics = calculate_topk_metrics(subset, k_values=[5, 8, 15])
        
        stratified[str(stratum_value)] = {
            **ml_metrics,
            **topk_metrics,
            'n_samples': len(subset)
        }
    
    return stratified

def calculate_calibration(df, n_bins=10):
    """Calculate calibration metrics"""
    calibration_data = []
    
    for _, row in df.iterrows():
        true_codes = set(row['true_codes'])
        pred_codes = row['predicted_codes']
        
        # Assign confidence based on rank (since we may not have explicit scores)
        for rank, pred in enumerate(pred_codes[:15]):
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
        'total_predictions': int(total_samples),
        'note': 'Calibration calculated from rank positions (confidence scores not available)'
    }

# ============================================================================
# Main Calculator
# ============================================================================

def calculate_all_metrics(results_json_path, enriched_csv_path, output_json_path, model_name=None):
    """
    Calculate all metrics and save to dashboard-compatible JSON
    
    Args:
        results_json_path: Path to {model}_results.json (per-sample predictions)
        enriched_csv_path: Path to enriched evaluation CSV
        output_json_path: Path to save eval_results_{model}.json
        model_name: Model name (extracted from filename if not provided)
    """
    
    print("="*80)
    print("COMPREHENSIVE METRICS CALCULATOR")
    print("="*80)
    print()
    
    # Determine model name
    if model_name is None:
        model_name = Path(results_json_path).stem.replace('_results', '')
    
    print(f"Model: {model_name}")
    print(f"Results: {results_json_path}")
    print(f"Enriched data: {enriched_csv_path}")
    print(f"Output: {output_json_path}")
    print()
    
    # Load results JSON (per-sample predictions)
    print("Loading predictions...")
    with open(results_json_path, 'r') as f:
        results_list = json.load(f)
    
    df_results = pd.DataFrame(results_list)
    print(f"  ✓ Loaded {len(df_results)} predictions")
    
    # Normalize codes
    df_results['true_codes'] = df_results['actual_codes'].apply(parse_icd_codes)
    df_results['predicted_codes'] = df_results['predicted_codes'].apply(parse_icd_codes)
    
    # Load enriched CSV
    print("Loading enriched data...")
    df_enriched = pd.read_csv(enriched_csv_path)
    print(f"  ✓ Loaded {len(df_enriched)} enriched samples")
    
    # Merge
    print("Merging data...")
    df = df_results.merge(df_enriched, on='sample_id', how='inner', suffixes=('', '_enriched'))
    print(f"  ✓ Merged {len(df)} samples")
    print()
    
    # Get all unique codes
    all_codes = set()
    for codes in df['true_codes']:
        all_codes.update(codes)
    print(f"Total unique ICD codes: {len(all_codes)}")
    print()
    
    # Initialize results structure
    eval_results = {
        'model_name': model_name,
        'overall': {},
        'stratified': {},
        'calibration': {}
    }
    
    # ========================================================================
    # 1. OVERALL METRICS
    # ========================================================================
    
    print("1️⃣  Calculating overall metrics...")
    
    # Multilabel metrics (F1, Precision, Recall, AUC - Micro/Macro)
    ml_metrics = calculate_multilabel_metrics(df, all_codes)
    eval_results['overall'].update(ml_metrics)
    
    # Top-K metrics (P@5/8/15, R@5/8/15, F1@5/8/15)
    topk_metrics = calculate_topk_metrics(df, k_values=[5, 8, 15])
    eval_results['overall'].update(topk_metrics)
    
    print(f"  ✓ F1 Micro: {ml_metrics['f1_micro']:.4f}")
    print(f"  ✓ P@5: {topk_metrics['prec_at_5']:.4f}, R@5: {topk_metrics['rec_at_5']:.4f}")
    print(f"  ✓ P@8: {topk_metrics['prec_at_8']:.4f}, R@8: {topk_metrics['rec_at_8']:.4f}")
    print(f"  ✓ P@15: {topk_metrics['prec_at_15']:.4f}, R@15: {topk_metrics['rec_at_15']:.4f}")
    print()
    
    # ========================================================================
    # 2. STRATIFIED BY CODE FREQUENCY (EXISTING)
    # ========================================================================
    
    print("2️⃣  Stratifying by code frequency...")
    if 'code_frequency_tier' in df.columns:
        eval_results['stratified']['code_freq'] = calculate_stratified_metrics(
            df, 'code_frequency_tier', all_codes
        )
        for tier, metrics in eval_results['stratified']['code_freq'].items():
            print(f"  ✓ {tier}: F1={metrics['f1_micro']:.4f}, n={metrics['n_samples']}")
    else:
        print("  ⚠️  code_frequency_tier not found in data")
    print()
    
    # ========================================================================
    # 3. STRATIFIED BY DOCUMENT LENGTH (EXISTING)
    # ========================================================================
    
    print("3️⃣  Stratifying by document length...")
    if 'length_tier' in df.columns:
        eval_results['stratified']['length'] = calculate_stratified_metrics(
            df, 'length_tier', all_codes
        )
        for tier, metrics in eval_results['stratified']['length'].items():
            print(f"  ✓ {tier}: F1={metrics['f1_micro']:.4f}, n={metrics['n_samples']}")
    else:
        print("  ⚠️  length_tier not found in data")
    print()
    
    # ========================================================================
    # 4. STRATIFIED BY COMORBIDITY BURDEN (NEW)
    # ========================================================================
    
    print("4️⃣  Stratifying by comorbidity burden...")
    if 'comorbidity_tier' in df.columns:
        eval_results['stratified']['comorbidity'] = calculate_stratified_metrics(
            df, 'comorbidity_tier', all_codes
        )
        for tier, metrics in eval_results['stratified']['comorbidity'].items():
            print(f"  ✓ {tier}: P@10={metrics['prec_at_8']:.4f}, R@10={metrics['rec_at_8']:.4f}, n={metrics['n_samples']}")
    else:
        print("  ⚠️  comorbidity_tier not found in enriched data")
    print()
    
    # ========================================================================
    # 5. STRATIFIED BY ADMISSION CONTEXT (NEW)
    # ========================================================================
    
    print("5️⃣  Stratifying by admission context...")
    if 'admission_context' in df.columns:
        eval_results['stratified']['admission_context'] = calculate_stratified_metrics(
            df, 'admission_context', all_codes
        )
        for context, metrics in eval_results['stratified']['admission_context'].items():
            print(f"  ✓ {context}: P@10={metrics['prec_at_8']:.4f}, R@10={metrics['rec_at_8']:.4f}, n={metrics['n_samples']}")
    else:
        print("  ⚠️  admission_context not found in enriched data")
    print()
    
    # ========================================================================
    # 6. STRATIFIED BY RACE (NEW - HEALTH EQUITY)
    # ========================================================================
    
    print("6️⃣  Stratifying by race (health equity)...")
    if 'race_group' in df.columns:
        eval_results['stratified']['race'] = calculate_stratified_metrics(
            df, 'race_group', all_codes
        )
        for race, metrics in eval_results['stratified']['race'].items():
            print(f"  ✓ {race}: P@10={metrics['prec_at_8']:.4f}, R@10={metrics['rec_at_8']:.4f}, n={metrics['n_samples']}")
    else:
        print("  ⚠️  race_group not found in enriched data")
    print()
    
    # ========================================================================
    # 7. CALIBRATION METRICS (NEW)
    # ========================================================================
    
    print("7️⃣  Calculating calibration...")
    calibration = calculate_calibration(df, n_bins=10)
    if calibration:
        eval_results['calibration'] = calibration
        print(f"  ✓ Expected Calibration Error: {calibration['expected_calibration_error']:.4f}")
        print(f"  ✓ Calibration curve: {len(calibration['calibration_curve'])} bins")
    else:
        print("  ⚠️  Could not calculate calibration")
    print()
    
    # ========================================================================
    # 8. SAVE RESULTS
    # ========================================================================
    
    print("8️⃣  Saving results...")
    with open(output_json_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"  ✓ Saved to: {output_json_path}")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("="*80)
    print("METRICS SUMMARY")
    print("="*80)
    print()
    print(f"Model: {model_name}")
    print(f"Samples: {len(df)}")
    print()
    print("Overall Performance:")
    print(f"  F1 Micro: {eval_results['overall']['f1_micro']:.4f}")
    print(f"  F1 Macro: {eval_results['overall']['f1_macro']:.4f}")
    print(f"  P@5: {eval_results['overall']['prec_at_5']:.4f}")
    print(f"  P@8: {eval_results['overall']['prec_at_8']:.4f}")
    print(f"  R@8: {eval_results['overall']['rec_at_8']:.4f}")
    print()
    print("Stratifications:")
    for strat_name in eval_results['stratified']:
        n_groups = len(eval_results['stratified'][strat_name])
        print(f"  ✓ {strat_name}: {n_groups} groups")
    print()
    print("✅ All metrics calculated and saved!")
    print()

# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate comprehensive metrics for ICD prediction dashboard'
    )
    
    parser.add_argument(
        '-r', '--results',
        required=True,
        help='Path to {model}_results.json (per-sample predictions)'
    )
    
    parser.add_argument(
        '-e', '--enriched',
        required=True,
        help='Path to enriched evaluation CSV'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output path for eval_results_{model}.json'
    )
    
    parser.add_argument(
        '-m', '--model',
        default=None,
        help='Model name (auto-detected from filename if not specified)'
    )
    
    args = parser.parse_args()
    
    calculate_all_metrics(
        results_json_path=args.results,
        enriched_csv_path=args.enriched,
        output_json_path=args.output,
        model_name=args.model
    )
