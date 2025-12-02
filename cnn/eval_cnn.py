#!/usr/bin/env python3
"""
Evaluate Trained ConvNet Model on 900 Samples and Calculate All Metrics

This unified script:
1. Loads trained ConvNet model
2. Runs inference on llm_eval_900_enriched.csv (or llm_eval_900_sample.csv)
3. Saves predictions in LLM-compatible JSON format
4. Calls calculate_all_metrics_sklearn.py to compute all metrics
5. Outputs eval_results_convnet.json for dashboard

Usage:
    python eval_convnet_900.py --model-path ./models/conv_attn_best/model_best_f1_micro.pth --gpu
    python eval_convnet_900.py --model-path ./models/conv_attn_best/model_best_f1_micro.pth
"""

import sys
import os
import argparse
import csv
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path to import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import *
from utils import datasets
from utils import evaluation
from learn import models
from learn import tools

# Import the metrics calculator
from utils.calculate_all_metrics import calculate_all_metrics

# ============================================================================
# Model Loading (from eval_250.py)
# ============================================================================

def get_device(use_gpu=False):
    """Determine the best available device"""
    if not use_gpu:
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device('mps')
    else:
        print("GPU requested but not available, using CPU")
        return torch.device('cpu')

def load_trained_model(model_path, dicts, gpu=False):
    """Load a trained model from .pth file"""
    print(f"Loading model from: {model_path}")
    
    # Determine device
    device = get_device(gpu)
    
    # Load model state
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load params to know model architecture
    model_dir = os.path.dirname(model_path)
    params_file = os.path.join(model_dir, 'params.json')
    
    if not os.path.exists(params_file):
        raise ValueError(f"params.json not found in {model_dir}")
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    print(f"Model configuration:")
    print(f"  Architecture: {params.get('model', 'unknown')}")
    print(f"  Label set: {params.get('Y', 'unknown')}")
    
    # Create args object from params
    class Args:
        pass
    
    args = Args()
    
    # Infer model type
    if 'model' in params:
        args.model = params['model']
    else:
        dir_name = os.path.basename(model_dir)
        if 'conv_attn' in dir_name:
            args.model = 'conv_attn'
        elif 'cnn_vanilla' in dir_name:
            args.model = 'cnn_vanilla'
        elif 'rnn' in dir_name:
            args.model = 'rnn'
        else:
            args.model = 'conv_attn'
    
    args.Y = params.get('Y', 'full')
    args.filter_size = params.get('filter_size', 10)
    args.num_filter_maps = params.get('num_filter_maps', 50)
    args.dropout = params.get('dropout', 0.2)
    args.gpu = gpu
    args.embed_file = params.get('embed_file', None)
    args.lmbda = params.get('lmbda', 0)
    args.command = 'test'
    args.version = params.get('version', 'mimic4')
    args.embed_size = params.get('embed_size', 100)
    args.code_emb = params.get('code_emb', None)
    args.test_model = None
    args.rnn_dim = params.get('rnn_dim', 128)
    args.cell_type = params.get('cell_type', 'lstm')
    args.rnn_layers = params.get('rnn_layers', 1)
    args.bidirectional = params.get('bidirectional', False)
    args.pool = params.get('pool', 'max')
    
    # Build model architecture
    model = tools.pick_model(args, dicts)
    
    # Load weights
    model.load_state_dict(checkpoint)
    
    if gpu:
        model.to(device)
    
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model, params, device

# ============================================================================
# Inference and Prediction Generation
# ============================================================================

def run_inference_and_save_predictions(model, data_file, dicts, device, output_json, batch_size=16):
    """
    Run model inference and save predictions in LLM-compatible JSON format
    
    Output format matches gemini_results.json:
    [
        {
            "sample_id": 1,
            "hadm_id": 12345,
            "actual_codes": ["A001", "B002"],
            "predicted_codes": ["A001", "C003", ...],  # Top 15, ordered by score
            "code_frequency_tier": "common",
            "length_tier": "short"
        },
        ...
    ]
    """
    print(f"\n{'='*80}")
    print(f"RUNNING INFERENCE ON 900 SAMPLES")
    print(f"{'='*80}\n")
    
    # Set CSV field size limit
    csv.field_size_limit(sys.maxsize)
    
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    
    # Reverse mapping: index to code
    ind2code = {idx: code for code, idx in c2ind.items()}
    num_labels = len(c2ind)
    
    all_docs = []
    all_labels = []
    all_sample_data = []
    
    # Load data
    print("Loading data...")
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Detect CSV format
        # Check if we have all required columns
        header_lower = [h.lower() for h in header]
        
        for row in reader:
            try:
                # Find indices dynamically
                sample_id = int(row[header_lower.index('sample_id')])
                hadm_id = int(row[header_lower.index('hadm_id')])
                text = row[header_lower.index('text')]
                codes_str = row[header_lower.index('labels')]
                
                # Try to get enrichment columns (may not exist)
                try:
                    length_tier = row[header_lower.index('length_tier')]
                except:
                    length_tier = 'unknown'
                
                try:
                    code_frequency_tier = row[header_lower.index('code_frequency_tier')]
                except:
                    code_frequency_tier = 'unknown'
                
                # Convert text to indices
                text_tokens = [int(w2ind.get(w, len(w2ind)+1)) for w in text.split()]
                
                # Truncate to max length
                if len(text_tokens) > MAX_LENGTH:
                    text_tokens = text_tokens[:MAX_LENGTH]
                
                # Convert codes to multi-hot vector
                labels_idx = np.zeros(num_labels)
                actual_codes = []
                for code in codes_str.split(';'):
                    code = code.strip()
                    if code and code in c2ind:
                        labels_idx[c2ind[code]] = 1
                        actual_codes.append(code)
                
                all_docs.append(text_tokens)
                all_labels.append(labels_idx)
                all_sample_data.append({
                    'sample_id': sample_id,
                    'hadm_id': hadm_id,
                    'actual_codes': actual_codes,
                    'code_frequency_tier': code_frequency_tier,
                    'length_tier': length_tier
                })
            except Exception as e:
                print(f"  ⚠️  Error processing row: {e}")
                continue
    
    print(f"✓ Loaded {len(all_docs)} samples")
    
    # Run inference
    print("\nRunning inference...")
    num_samples = len(all_docs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_scores = []
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            
            # Get batch
            batch_docs = all_docs[start_idx:end_idx]
            batch_labels = all_labels[start_idx:end_idx]
            
            # Pad documents
            max_len = max(len(doc) for doc in batch_docs)
            padded_docs = []
            for doc in batch_docs:
                if len(doc) < max_len:
                    doc = doc + [0] * (max_len - len(doc))
                padded_docs.append(doc)
            
            # Convert to tensors
            docs_tensor = torch.LongTensor(padded_docs).to(device)
            labels_tensor = torch.FloatTensor(np.array(batch_labels)).to(device)
            
            # Forward pass
            outputs = model(docs_tensor, labels_tensor)
            
            # Extract just outputs (may be tuple)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get scores and move to CPU immediately
            scores = outputs.cpu().numpy()
            all_scores.append(scores)
            
            # Clean up GPU memory
            del docs_tensor, labels_tensor, outputs
            if str(device).startswith('mps') or str(device).startswith('cuda'):
                torch.mps.empty_cache() if str(device).startswith('mps') else torch.cuda.empty_cache()
    
    # Concatenate all scores
    y_scores = np.vstack(all_scores)
    
    print(f"✓ Inference complete. Score matrix shape: {y_scores.shape}")
    
    # Convert scores to top-15 ranked predictions
    print("\nGenerating top-15 predictions...")
    results_list = []
    
    for i in range(num_samples):
        sample_scores = y_scores[i]
        
        # Get top 15 indices by score
        top_indices = np.argsort(sample_scores)[::-1][:15]
        
        # Convert indices to ICD codes
        predicted_codes = []
        for idx in top_indices:
            if idx in ind2code:
                predicted_codes.append(ind2code[idx])
        
        # Create result entry
        result = {
            'sample_id': all_sample_data[i]['sample_id'],
            'hadm_id': all_sample_data[i]['hadm_id'],
            'actual_codes': all_sample_data[i]['actual_codes'],
            'predicted_codes': predicted_codes,
            'code_frequency_tier': all_sample_data[i]['code_frequency_tier'],
            'length_tier': all_sample_data[i]['length_tier'],
            'model': 'ConvNet',
            'success': True
        }
        
        results_list.append(result)
    
    # Save to JSON
    print(f"\nSaving predictions to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"✓ Saved {len(results_list)} predictions")
    print()
    
    return output_json

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ConvNet model on 900 samples and calculate all metrics'
    )
    
    parser.add_argument('--model-path', required=True,
                        help='Path to trained model .pth file')
    parser.add_argument('--data-file', default='llm_eval_900_enriched.csv',
                        help='Path to 900-sample CSV file (default: llm_eval_900_enriched.csv)')
    parser.add_argument('--output-dir', default='llm_eval_results_900',
                        help='Output directory for results (default: llm_eval_results_900)')
    parser.add_argument('--model-name', default='Trained ConvNet',
                        help='Model name for output files')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CONVNET MODEL EVALUATION ON 900 SAMPLES")
    print("="*80)
    print()
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_file}")
    print(f"GPU: {args.gpu}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dictionaries (using load_lookups like eval_250.py)
    print("Loading vocabularies...")
    
    # Create args object for loading lookups
    class LoadArgs:
        pass
    
    load_args = LoadArgs()
    load_args.Y = 'full'
    load_args.version = 'mimic4'
    load_args.vocab = os.path.join(MIMIC_4_DIR, 'vocab.csv')
    load_args.data_path = os.path.join(MIMIC_4_DIR, 'full_code', 'train_full.csv')
    load_args.public_model = False
    load_args.model = 'conv_attn'
    load_args.embed_file = None
    load_args.embed_size = 100
    
    dicts = datasets.load_lookups(load_args, desc_embed=False)
    
    print(f"✓ Vocabulary size: {len(dicts['ind2w'])}")
    print(f"✓ Number of ICD codes: {len(dicts['ind2c'])}")
    print()
    
    # Load model
    device = get_device(args.gpu)
    model, params, device = load_trained_model(args.model_path, dicts, args.gpu)
    print()
    
    # Run inference and save predictions
    predictions_json = output_dir / 'convnet_predictions_900.json'
    run_inference_and_save_predictions(
        model=model,
        data_file=args.data_file,
        dicts=dicts,
        device=device,
        output_json=predictions_json,
        batch_size=args.batch_size
    )
    
    # Calculate all metrics using sklearn
    print("="*80)
    print("CALCULATING ALL METRICS (sklearn-based)")
    print("="*80)
    print()
    
    output_metrics_json = output_dir / 'eval_results_trained_convnet.json'
    
    calculate_all_metrics(
        results_json_path=str(predictions_json),
        enriched_csv_path=args.data_file,
        output_json_path=str(output_metrics_json),
        model_name=args.model_name
    )
    
    print()
    print("="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print()
    print(f"Predictions saved to: {predictions_json}")
    print(f"Metrics saved to: {output_metrics_json}")
    print()
    print("You can now copy the metrics JSON to your dashboard:")
    print(f"  cp {output_metrics_json} model-dashboard/public/data/")
    print()

if __name__ == '__main__':
    main()
