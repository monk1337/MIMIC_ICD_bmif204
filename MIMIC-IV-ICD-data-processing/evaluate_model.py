#!/usr/bin/env python
"""
Evaluate a trained model and display comprehensive metrics

Usage:
    python evaluate_model.py --model-path ./models/conv_attn_Nov_10_15:30:45/model_best_f1_micro.pth
    python evaluate_model.py --compare-models ./models/model1/ ./models/model2/
"""

import sys
import os
import json
import argparse
from tabulate import tabulate

def load_metrics(model_dir):
    """Load metrics from a model directory"""
    metrics_file = os.path.join(model_dir, 'metrics.json')
    params_file = os.path.join(model_dir, 'params.json')
    
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found at {metrics_file}")
        return None, None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    params = None
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
    
    return metrics, params

def display_metrics(metrics, params=None, model_name="Model"):
    """Display metrics in a formatted table"""
    print("\n" + "="*80)
    print(f"{model_name.upper()} - EVALUATION RESULTS")
    print("="*80)
    
    if params:
        print("\nModel Configuration:")
        print(f"  Label Set: {params.get('Y', 'N/A')}")
        print(f"  Filter Size: {params.get('filter_size', 'N/A')}")
        print(f"  Dropout: {params.get('dropout', 'N/A')}")
        print(f"  Learning Rate: {params.get('lr', 'N/A')}")
        print(f"  Lambda: {params.get('lmbda', 'N/A')}")
    
    # Prepare metrics table
    print("\n" + "-"*80)
    print("PERFORMANCE METRICS")
    print("-"*80)
    
    # Get the most recent epoch metrics (last values in lists)
    def get_final_metric(metrics, key):
        if key in metrics and metrics[key]:
            val = metrics[key]
            if isinstance(val, list):
                return val[-1] if val else 0
            return val
        return 0
    
    # Macro metrics
    macro_table = [
        ["Accuracy", f"{get_final_metric(metrics, 'acc_macro'):.4f}"],
        ["Precision", f"{get_final_metric(metrics, 'prec_macro'):.4f}"],
        ["Recall", f"{get_final_metric(metrics, 'rec_macro'):.4f}"],
        ["F1-Score", f"{get_final_metric(metrics, 'f1_macro'):.4f}"],
        ["AUC", f"{get_final_metric(metrics, 'auc_macro'):.4f}"],
    ]
    
    print("\nMACRO Metrics (averaged across all labels):")
    print(tabulate(macro_table, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Micro metrics
    micro_table = [
        ["Accuracy", f"{get_final_metric(metrics, 'acc_micro'):.4f}"],
        ["Precision", f"{get_final_metric(metrics, 'prec_micro'):.4f}"],
        ["Recall", f"{get_final_metric(metrics, 'rec_micro'):.4f}"],
        ["F1-Score", f"{get_final_metric(metrics, 'f1_micro'):.4f}"],
        ["AUC", f"{get_final_metric(metrics, 'auc_micro'):.4f}"],
    ]
    
    print("\nMICRO Metrics (averaged across all predictions):")
    print(tabulate(micro_table, headers=["Metric", "Value"], tablefmt="grid"))
    
    # @k metrics
    k_metrics = []
    for k in [5, 8, 15]:
        prec_k = get_final_metric(metrics, f'prec_at_{k}')
        rec_k = get_final_metric(metrics, f'rec_at_{k}')
        f1_k = get_final_metric(metrics, f'f1_at_{k}')
        if prec_k > 0 or rec_k > 0:
            k_metrics.append([f"@{k}", f"{prec_k:.4f}", f"{rec_k:.4f}", f"{f1_k:.4f}"])
    
    if k_metrics:
        print("\nMetrics @k (top-k predictions):")
        print(tabulate(k_metrics, headers=["k", "Precision", "Recall", "F1"], tablefmt="grid"))
    
    # Loss metrics
    loss_table = []
    for split in ['tr', 'te']:
        loss_key = f'loss_{split}'
        if loss_key in metrics:
            loss_val = get_final_metric(metrics, loss_key)
            split_name = "Train" if split == 'tr' else "Test"
            loss_table.append([split_name, f"{loss_val:.6f}"])
    
    if loss_table:
        print("\nLoss:")
        print(tabulate(loss_table, headers=["Split", "Loss"], tablefmt="grid"))
    
    print("="*80 + "\n")

def compare_models(model_dirs):
    """Compare metrics across multiple models"""
    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100)
    
    all_metrics = []
    model_names = []
    
    for model_dir in model_dirs:
        metrics, params = load_metrics(model_dir)
        if metrics:
            model_name = os.path.basename(model_dir.rstrip('/'))
            model_names.append(model_name)
            all_metrics.append((metrics, params, model_name))
    
    if len(all_metrics) < 2:
        print("Need at least 2 valid models to compare")
        return
    
    # Create comparison table
    def get_final_metric(metrics, key):
        if key in metrics and metrics[key]:
            val = metrics[key]
            if isinstance(val, list):
                return val[-1] if val else 0
            return val
        return 0
    
    comparison_data = []
    metric_keys = [
        ('Macro F1', 'f1_macro'),
        ('Micro F1', 'f1_micro'),
        ('Macro Prec', 'prec_macro'),
        ('Micro Prec', 'prec_micro'),
        ('Macro Rec', 'rec_macro'),
        ('Micro Rec', 'rec_micro'),
        ('Macro AUC', 'auc_macro'),
        ('Micro AUC', 'auc_micro'),
        ('Prec@8', 'prec_at_8'),
        ('Rec@8', 'rec_at_8'),
        ('F1@8', 'f1_at_8'),
    ]
    
    for metric_name, metric_key in metric_keys:
        row = [metric_name]
        for metrics, params, name in all_metrics:
            val = get_final_metric(metrics, metric_key)
            row.append(f"{val:.4f}")
        comparison_data.append(row)
    
    headers = ["Metric"] + model_names
    print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
    
    # Configuration comparison
    print("\n" + "-"*100)
    print("MODEL CONFIGURATIONS")
    print("-"*100)
    
    config_data = []
    config_keys = ['Y', 'dropout', 'lr', 'filter_size', 'num_filter_maps', 'lmbda']
    
    for key in config_keys:
        row = [key]
        for metrics, params, name in all_metrics:
            if params and key in params:
                row.append(str(params[key]))
            else:
                row.append("N/A")
        config_data.append(row)
    
    print(tabulate(config_data, headers=headers, tablefmt="grid"))
    print("="*100 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models and display metrics")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-dir", type=str, dest="model_dir",
                       help="Path to model directory containing metrics.json")
    group.add_argument("--model-path", type=str, dest="model_path",
                       help="Path to model .pth file (will use parent directory)")
    group.add_argument("--compare-models", type=str, nargs='+', dest="compare_models",
                       help="Paths to multiple model directories to compare")
    
    args = parser.parse_args()
    
    if args.compare_models:
        compare_models(args.compare_models)
    else:
        if args.model_path:
            model_dir = os.path.dirname(args.model_path)
        else:
            model_dir = args.model_dir
        
        metrics, params = load_metrics(model_dir)
        
        if metrics:
            model_name = os.path.basename(model_dir.rstrip('/'))
            display_metrics(metrics, params, model_name)
        else:
            print("Failed to load metrics")
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("\nNote: 'tabulate' package not found. Installing for better formatting...")
        print("Run: pip install tabulate")
        print("\nShowing basic output instead:\n")
        # Fallback to simple printing
        main()
