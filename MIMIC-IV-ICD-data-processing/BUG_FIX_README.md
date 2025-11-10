# Bug Fix Summary

## Issue Encountered

Your training **failed** during the first epoch evaluation phase with the following error:

```
TypeError: save_samples() got multiple values for argument 'dicts'
```

## What Happened

1. ✅ **Training started successfully** - Completed all ~6,903 batches of epoch 0
2. ✅ **Training phase completed** - Average loss: 0.00717
3. ❌ **Evaluation phase failed** - Error occurred when trying to save attention samples

## Root Cause

There were **2 bugs** in the code:

### Bug 1: Incorrect Function Call (Line 280)
The `save_samples()` function was being called with an extra `epoch` parameter that doesn't exist in the function signature.

**Before (BROKEN):**
```python
interpret.save_samples(data, output, target_data, alpha, window_size, epoch, tp_file, fp_file, dicts=dicts)
```

**After (FIXED):**
```python
interpret.save_samples(data, output, target_data, alpha, window_size, tp_file, fp_file, dicts=dicts)
```

### Bug 2: Deprecated PyTorch Code (Line 255)
Using deprecated `volatile=True` parameter (removed in PyTorch 1.0+).

**Before (DEPRECATED):**
```python
data, target = Variable(torch.LongTensor(data), volatile=True), Variable(torch.FloatTensor(target))
```

**After (MODERN):**
```python
with torch.no_grad():
    data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
```

## Fixes Applied

✅ Both bugs have been fixed in `/learn/training.py`

## Next Steps

You can now **resume training** with the fixed code:

### Option 1: Start Fresh (Recommended)
```bash
python train_full_codes.py --model conv_attn --gpu --n-epochs 50 --samples
```

### Option 2: Without Saving Samples (Faster)
If you don't need attention examples, skip the `--samples` flag:
```bash
python train_full_codes.py --model conv_attn --gpu --n-epochs 50
```

This will avoid the code path that caused the error entirely.

### Option 3: Quick Test First
Test with fewer epochs to ensure everything works:
```bash
python train_full_codes.py --model conv_attn --gpu --n-epochs 5
```

## What You'll See When It Works

After epoch 0 completes, you should see:

```
epoch loss: 0.0071689...
file for evaluation: .../dev_full.csv
4017it [XX:XX, XX.XXit/s]

[MACRO] accuracy, precision, recall, f-measure, AUC
0.XXXX, 0.XXXX, 0.XXXX, 0.XXXX, 0.XXXX

[MICRO] accuracy, precision, recall, f-measure, AUC
0.XXXX, 0.XXXX, 0.XXXX, 0.XXXX, 0.XXXX

prec_at_8: 0.XXXX
rec_at_8: 0.XXXX
f1_at_8: 0.XXXX

prec_at_15: 0.XXXX
rec_at_15: 0.XXXX
f1_at_15: 0.XXXX

saved metrics, params, model to directory ./models/conv_attn_XXX
```

Then it will continue to epoch 1, 2, 3, etc.

## Training Progress from Your Log

Your training was progressing well before the crash:
- **Batches processed**: 6,903
- **Training time**: ~13 minutes for epoch 0
- **Processing speed**: ~7 iterations/second
- **Final training loss**: 0.00717

This suggests the model was learning (loss decreased from higher values initially).

## Expected Total Training Time

For 50 epochs with full code set:
- **~13 minutes per epoch** × 50 epochs = **~11 hours**
- With early stopping (patience=5), likely **5-15 epochs** = **1-3 hours**

## Files Modified

- ✅ `/learn/training.py` - Fixed both bugs

All other files remain unchanged and working correctly.
