#!/usr/bin/env python
"""
Train models on the full MIMIC-IV ICD-10 code set (~27k codes)
This script trains on all codes and outputs comprehensive metrics including:
- Macro/Micro: accuracy, precision, recall, F1
- AUC (macro and micro)
- Precision@8, Precision@15
- Recall@8, Recall@15
- F1@8, F1@15

Usage:
    python train_full_codes.py --model conv_attn --gpu
    python train_full_codes.py --model cnn_vanilla --gpu --batch-size 8
    python train_full_codes.py --model logreg --gpu
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import MIMIC_4_DIR
from learn.training import main

def train_full_codes(args):
    """
    Configure and run training on full code set
    """
    # Set data paths for full code training
    args.data_path = os.path.join(MIMIC_4_DIR, 'full_code', 'train_full.csv')
    args.vocab = os.path.join(MIMIC_4_DIR, 'vocab.csv')
    args.Y = 'full'  # Use full code set instead of top 50
    args.version = 'mimic4'
    
    # Verify files exist
    if not os.path.exists(args.data_path):
        print(f"ERROR: Training data not found at {args.data_path}")
        print(f"Please ensure you have run the data processing notebook first.")
        return
    
    if not os.path.exists(args.vocab):
        print(f"ERROR: Vocabulary file not found at {args.vocab}")
        return
    
    print("="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Data path: {args.data_path}")
    print(f"Vocab: {args.vocab}")
    print(f"Label set: {args.Y} (all codes)")
    print(f"Epochs: {args.n_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"GPU: {args.gpu}")
    print(f"Dropout: {args.dropout}")
    print("="*80)
    print()
    
    # Run training
    main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train neural network on MIMIC-IV with full ICD-10 code set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train CNN with attention (CAML model) - recommended
  python train_full_codes.py --model conv_attn --gpu --n-epochs 50
  
  # Train vanilla CNN
  python train_full_codes.py --model cnn_vanilla --gpu --n-epochs 30
  
  # Train logistic regression baseline
  python train_full_codes.py --model logreg --gpu --n-epochs 20
  
  # Train RNN with LSTM
  python train_full_codes.py --model rnn --cell-type lstm --gpu --n-epochs 30
  
  # Use smaller batch size for large models
  python train_full_codes.py --model conv_attn --gpu --batch-size 8 --n-epochs 50
  
  # Train with pre-trained embeddings (if available)
  python train_full_codes.py --model conv_attn --gpu --embed-file ./mimicdata/mimic4_icd10/embeddings.embed
        """
    )
    
    # Model selection
    parser.add_argument("--model", type=str, required=True,
                        choices=["conv_attn", "cnn_vanilla", "rnn", "logreg"],
                        help="Model architecture to use")
    
    # Training parameters
    parser.add_argument("--n-epochs", type=int, default=50, dest="n_epochs",
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size",
                        help="Training batch size (default: 16, use 8 for large models)")
    parser.add_argument("--lr", type=float, default=1e-3, dest="lr",
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.5, dest="dropout",
                        help="Dropout rate (default: 0.5)")
    parser.add_argument("--weight-decay", type=float, default=0, dest="weight_decay",
                        help="L2 regularization weight decay (default: 0)")
    
    # Model architecture parameters
    parser.add_argument("--embed-size", type=int, default=100, dest="embed_size",
                        help="Embedding dimension size (default: 100)")
    parser.add_argument("--filter-size", type=str, default="4", dest="filter_size",
                        help="Convolution filter size (default: 4)")
    parser.add_argument("--num-filter-maps", type=int, default=50, dest="num_filter_maps",
                        help="Number of filters (default: 50)")
    
    # RNN-specific parameters
    parser.add_argument("--cell-type", type=str, choices=["lstm", "gru"], default="gru",
                        dest="cell_type", help="RNN cell type (default: gru)")
    parser.add_argument("--rnn-dim", type=int, default=128, dest="rnn_dim",
                        help="RNN hidden dimension (default: 128)")
    parser.add_argument("--rnn-layers", type=int, default=1, dest="rnn_layers",
                        help="Number of RNN layers (default: 1)")
    parser.add_argument("--bidirectional", action="store_const", const=True,
                        dest="bidirectional", help="Use bidirectional RNN")
    
    # Logistic regression parameters
    parser.add_argument("--pool", type=str, choices=["max", "avg"], default="max",
                        dest="pool", help="Pooling type for logreg (default: max)")
    
    # Advanced options
    parser.add_argument("--embed-file", type=str, dest="embed_file",
                        help="Path to pre-trained embeddings file")
    parser.add_argument("--code-emb", type=str, dest="code_emb",
                        help="Path to code embeddings for initialization")
    parser.add_argument("--lmbda", type=float, default=0, dest="lmbda",
                        help="DR-CAML description regularization weight (default: 0)")
    
    # Training control
    parser.add_argument("--criterion", type=str, default="f1_micro", dest="criterion",
                        help="Early stopping metric (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=3, dest="patience",
                        help="Early stopping patience in epochs (default: 3)")
    
    # Hardware and output
    parser.add_argument("--gpu", action="store_const", const=True, dest="gpu",
                        help="Use GPU if available (CUDA or MPS)")
    parser.add_argument("--samples", action="store_const", const=True, dest="samples",
                        help="Save attention samples for interpretation")
    parser.add_argument("--quiet", action="store_const", const=True, dest="quiet",
                        help="Reduce training output verbosity")
    
    # Testing
    parser.add_argument("--test-model", type=str, dest="test_model",
                        help="Path to saved model for evaluation only")
    parser.add_argument("--public-model", action="store_const", const=True,
                        dest="public_model", help="Flag for public pre-trained models")
    
    args = parser.parse_args()
    
    # Store command for reproducibility
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    
    # Run training
    train_full_codes(args)
