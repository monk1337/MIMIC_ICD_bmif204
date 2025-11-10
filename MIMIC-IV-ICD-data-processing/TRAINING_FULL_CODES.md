# Training on Full MIMIC-IV ICD-10 Code Set

This guide explains how to train models on **all ~27,000 ICD-10 codes** instead of just the top 50.

## Quick Start

### Method 1: Using the Training Script (Recommended)

```bash
# Make the script executable
chmod +x run_full_training.sh

# Train with default settings (CNN with attention, 50 epochs)
./run_full_training.sh

# Train with specific model
./run_full_training.sh conv_attn 16 50

# Train vanilla CNN with smaller batch
./run_full_training.sh cnn_vanilla 8 30
```

### Method 2: Using Python Script Directly

```bash
# Train CNN with attention (CAML) - Best performance
python train_full_codes.py --model conv_attn --gpu --n-epochs 50

# Train vanilla CNN
python train_full_codes.py --model cnn_vanilla --gpu --n-epochs 30 --batch-size 8

# Train logistic regression (fastest, baseline)
python train_full_codes.py --model logreg --gpu --n-epochs 20

# Train RNN with LSTM
python train_full_codes.py --model rnn --cell-type lstm --gpu --n-epochs 30
```

## Model Options

| Model | Description | Recommended Settings | Training Time |
|-------|-------------|---------------------|---------------|
| `conv_attn` | CNN with attention (CAML) | `--batch-size 16 --n-epochs 50` | ~4-6 hours |
| `cnn_vanilla` | Standard CNN | `--batch-size 8 --n-epochs 30` | ~2-3 hours |
| `logreg` | Logistic regression baseline | `--batch-size 32 --n-epochs 20` | ~1-2 hours |
| `rnn` | RNN/LSTM/GRU | `--batch-size 16 --n-epochs 30` | ~5-8 hours |

## Output Metrics

The training script provides comprehensive evaluation metrics:

### Macro Metrics (averaged across all labels)
- **Accuracy**: Overall correctness
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Micro Metrics (averaged across all predictions)
- Same metrics as macro, but calculated per-instance

### Metrics @k (top-k predictions)
- **Precision@8**: Precision considering top 8 predictions
- **Recall@8**: Recall considering top 8 predictions
- **F1@8**: F1-score for top 8 predictions
- **Precision@15**: Precision considering top 15 predictions
- **Recall@15**: Recall considering top 15 predictions
- **F1@15**: F1-score for top 15 predictions

## Expected Performance

Based on similar studies, expected metrics on full code set:

| Metric | Expected Range | Top-50 Comparison |
|--------|----------------|-------------------|
| Macro F1 | 0.35 - 0.45 | 0.55 - 0.65 |
| Micro F1 | 0.50 - 0.60 | 0.65 - 0.75 |
| Precision@8 | 0.65 - 0.75 | 0.75 - 0.85 |
| Recall@8 | 0.40 - 0.50 | 0.55 - 0.65 |
| AUC Micro | 0.85 - 0.90 | 0.90 - 0.95 |

*Note: Performance on full code set is naturally lower due to many rare codes*

## Training Output Example

```
==================================================
TRAINING CONFIGURATION
==================================================
Model: conv_attn
Data path: ./mimicdata/mimic4_icd10/full_code/train_full.csv
Vocab: ./mimicdata/mimic4_icd10/vocab.csv
Label set: full (all codes)
Epochs: 50
Batch size: 16
Learning rate: 0.001
GPU: True
==================================================

loading lookups...
Building model...

EPOCH 0
Train epoch: 0 [batch #0, batch_size 16, seq length 2500]	Loss: 0.0234
Train epoch: 0 [batch #25, batch_size 16, seq length 2500]	Loss: 0.0198
...

file for evaluation: ./mimicdata/mimic4_icd10/full_code/dev_full.csv
[MACRO] accuracy, precision, recall, f-measure, AUC
0.5234, 0.4123, 0.3987, 0.4053, 0.8734

[MICRO] accuracy, precision, recall, f-measure, AUC
0.6789, 0.5678, 0.5234, 0.5445, 0.8912

prec_at_8: 0.7123
rec_at_8: 0.4567
f1_at_8: 0.5589

prec_at_15: 0.6234
rec_at_15: 0.5123
f1_at_15: 0.5623
```

## Evaluating Trained Models

After training completes, evaluate the model:

```bash
# Evaluate a single model
python evaluate_model.py --model-dir ./models/conv_attn_Nov_10_15:30:45/

# Or using the model file path
python evaluate_model.py --model-path ./models/conv_attn_Nov_10_15:30:45/model_best_f1_micro.pth

# Compare multiple models
python evaluate_model.py --compare-models \
    ./models/conv_attn_Nov_10_15:30:45/ \
    ./models/cnn_vanilla_Nov_10_16:45:12/ \
    ./models/logreg_Nov_10_14:20:30/
```

## Advanced Training Options

### Using Pre-trained Embeddings

If you have Word2Vec embeddings:

```bash
python train_full_codes.py \
    --model conv_attn \
    --embed-file ./mimicdata/mimic4_icd10/embeddings.embed \
    --gpu
```

### Label Description Regularization (DR-CAML)

```bash
python train_full_codes.py \
    --model conv_attn \
    --lmbda 0.1 \
    --gpu
```

### Adjusting Training Parameters

```bash
python train_full_codes.py \
    --model conv_attn \
    --gpu \
    --n-epochs 100 \
    --batch-size 8 \
    --lr 0.0005 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --patience 10 \
    --criterion f1_micro
```

### Saving Attention Samples

```bash
python train_full_codes.py \
    --model conv_attn \
    --gpu \
    --samples
```

This saves examples of where the model focuses attention in:
- `tp_dev_examples_*.txt` - True positive examples
- `fp_dev_examples_*.txt` - False positive examples

## Memory Requirements

Training on full code set requires more memory:

| Model | Minimum RAM | Recommended RAM | GPU VRAM |
|-------|-------------|-----------------|----------|
| logreg | 8 GB | 16 GB | 4 GB |
| cnn_vanilla | 12 GB | 24 GB | 6 GB |
| conv_attn | 16 GB | 32 GB | 8 GB |
| rnn | 20 GB | 48 GB | 10 GB |

**Tips for limited memory:**
- Reduce `--batch-size` (e.g., 8 or 4)
- Use `logreg` or `cnn_vanilla` instead of `conv_attn`
- Close other applications
- Use smaller `--num-filter-maps` (e.g., 25 instead of 50)

## Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
python train_full_codes.py --model conv_attn --gpu --batch-size 4

# Or use CPU (slower but no memory limit)
python train_full_codes.py --model conv_attn --batch-size 8
```

### Training Too Slow

```bash
# Use simpler model
python train_full_codes.py --model cnn_vanilla --gpu

# Reduce filter maps
python train_full_codes.py --model conv_attn --gpu --num-filter-maps 25
```

### Model Not Improving

```bash
# Adjust learning rate
python train_full_codes.py --model conv_attn --gpu --lr 0.0001

# Increase patience
python train_full_codes.py --model conv_attn --gpu --patience 10

# Add regularization
python train_full_codes.py --model conv_attn --gpu --weight-decay 0.0001 --dropout 0.3
```

## Results Location

Training results are saved in:
```
./models/[model]_[timestamp]/
├── model_best_f1_micro.pth    # Best model checkpoint
├── metrics.json                # All metrics over epochs
├── params.json                 # Training parameters
├── preds_dev.psv              # Dev set predictions
├── preds_test.psv             # Test set predictions
├── pred_100_scores_dev.json   # Top 100 prediction scores (dev)
└── pred_100_scores_test.json  # Top 100 prediction scores (test)
```

## Comparison: Top-50 vs Full Codes

To train on Top-50 codes for comparison:

```bash
cd learn
python training.py \
    ../mimicdata/mimic4_icd10/top_50/train_50.csv \
    ../mimicdata/mimic4_icd10/vocab.csv \
    50 \
    conv_attn \
    50 \
    --gpu \
    --dataset mimic4
```

Then compare:
```bash
python evaluate_model.py --compare-models \
    ./models/full_code_model/ \
    ./models/top_50_model/
```

## Next Steps

After training:

1. **Evaluate on test set** - Results automatically saved during training
2. **Analyze predictions** - Check `preds_test.psv` for predicted codes
3. **Examine attention** - Review `tp_*_examples.txt` and `fp_*_examples.txt`
4. **Compare models** - Use `evaluate_model.py --compare-models`
5. **Fine-tune** - Adjust hyperparameters based on results

## Citation

If you use this code, please cite the original CAML/DR-CAML paper:

```
@article{mullenbach2018explainable,
  title={Explainable prediction of medical codes from clinical text},
  author={Mullenbach, James and Wiegreffe, Sarah and Duke, Jon and Sun, Jimeng and Eisenstein, Jacob},
  journal={NAACL-HLT},
  year={2018}
}
```
