#!/bin/bash
# Quick start script for training on full MIMIC-IV ICD-10 code set
# This script provides easy commands to train different models

echo "=========================================="
echo "MIMIC-IV Full Code Training Script"
echo "=========================================="
echo ""

# Check if data exists
if [ ! -f "./mimicdata/mimic4_icd10/full_code/train_full.csv" ]; then
    echo "ERROR: Training data not found!"
    echo "Please run the data processing notebook first:"
    echo "  jupyter notebook notebooks/dataproc_mimic_IV_exploration_icd10.ipynb"
    exit 1
fi

# Default model
MODEL="${1:-conv_attn}"
BATCH_SIZE="${2:-16}"
EPOCHS="${3:-50}"

echo "Training Configuration:"
echo "  Model: $MODEL"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo ""
echo "To use different settings, run:"
echo "  ./run_full_training.sh MODEL BATCH_SIZE EPOCHS"
echo "  Example: ./run_full_training.sh cnn_vanilla 8 30"
echo ""
echo "=========================================="
echo ""

# Train the model
python train_full_codes.py \
    --model $MODEL \
    --n-epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gpu \
    --samples \
    --dropout 0.5 \
    --lr 0.001 \
    --patience 5 \
    --criterion f1_micro

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Results saved in: ./models/"
echo ""
echo "To evaluate the model, check the metrics.json file in the model directory"
