#!/usr/bin/env python
"""
Create a stratified sample of 900 records for LLM evaluation (GPT, Claude, etc.)

Stratification Strategy:
- 100 samples in EACH scenario (9 total scenarios)
- Code Frequency: common (top 0.1%), medium (0.1-1%), rare (bottom 99%)
- Text Length: short, medium, long (33rd/67th percentiles)

Distribution:
                Short    Medium    Long    Total
Common          100      100      100      300
Medium          100      100      100      300  
Rare            100      100      100      300
Total           300      300      300      900

Frequency Thresholds (99.9th/99th percentiles):
- Ultra-aggressive to ensure sufficient rare/medium samples
- Balances distribution for meaningful stratified evaluation

Usage:
    python create_llm_eval_sample.py
    python create_llm_eval_sample.py --seed 42
    python create_llm_eval_sample.py --input test_full.csv --output llm_eval_900.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import Counter
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import MIMIC_4_DIR


def load_code_frequencies(train_file):
    """
    Calculate code frequencies from training data
    """
    print(f"📊 Loading training data to calculate code frequencies...")
    print(f"   File: {train_file}")
    
    if not os.path.exists(train_file):
        print(f"   ❌ ERROR: Training file not found at {train_file}")
        print(f"   Please ensure train_full.csv exists")
        return None
    
    print(f"   Loading... (this may take a minute)")
    df_train = pd.read_csv(train_file)
    
    # Count all codes across training set
    all_codes = []
    for labels_str in df_train['labels']:
        codes = labels_str.split(';')
        all_codes.extend(codes)
    
    code_counts = Counter(all_codes)
    print(f"   ✓ Found {len(code_counts)} unique codes in training set")
    print(f"   ✓ Total code occurrences: {len(all_codes):,}")
    
    return code_counts


def classify_code_frequency(codes, code_counts, percentile_67, percentile_33):
    """
    Classify a record based on its codes' frequencies
    Uses the AVERAGE frequency of all codes in the record
    
    Returns: 'common', 'medium', or 'rare'
    """
    if code_counts is None:
        # Fallback if no training data
        num_codes = len(codes)
        if num_codes <= 3:
            return 'common'
        elif num_codes <= 6:
            return 'medium'
        else:
            return 'rare'
    
    # Get average frequency of all codes in this record
    frequencies = [code_counts.get(code, 0) for code in codes]
    avg_freq = np.mean(frequencies) if frequencies else 0
    
    if avg_freq >= percentile_67:
        return 'common'
    elif avg_freq >= percentile_33:
        return 'medium'
    else:
        return 'rare'


def classify_length(length):
    """
    Classify text length into short/medium/long
    Based on percentiles of the test set
    """
    return length  # Will classify after loading full dataset


def stratified_sample(df, target_distribution, seed=42, target_total=500):
    """
    Sample records according to target distribution
    Will fill any shortfall proportionally from other strata
    
    target_distribution: dict with keys like ('common', 'short') -> count
    target_total: desired total samples (default 500)
    """
    np.random.seed(seed)
    
    sampled_records = []
    shortfall = 0
    remaining_by_stratum = {}
    
    # First pass: sample what we can from each stratum
    for (freq_tier, length_tier), target_count in target_distribution.items():
        # Filter records matching this stratum
        mask = (df['code_frequency_tier'] == freq_tier) & (df['length_tier'] == length_tier)
        stratum_df = df[mask]
        
        available = len(stratum_df)
        
        if available == 0:
            print(f"   ⚠️  No records for {freq_tier}/{length_tier}, skipping...")
            shortfall += target_count
            continue
        elif available < target_count:
            print(f"   ⚠️  Only {available} records for {freq_tier}/{length_tier}, need {target_count}")
            sample = stratum_df
            shortfall += (target_count - available)
        else:
            # Random sample
            sample = stratum_df.sample(n=target_count, random_state=seed)
            # Keep track of remaining samples in this stratum
            remaining = stratum_df.drop(sample.index)
            if len(remaining) > 0:
                remaining_by_stratum[(freq_tier, length_tier)] = remaining
        
        sampled_records.append(sample)
        print(f"   ✓ Sampled {len(sample):3d}/{target_count:3d} for {freq_tier:6s} × {length_tier:6s}")
    
    # Second pass: fill shortfall if needed
    if shortfall > 0 and remaining_by_stratum:
        print(f"\n   📊 Shortfall: {shortfall} samples to reach {target_total}")
        print(f"   🔄 Filling proportionally from strata with extra samples...")
        
        # Calculate how much to take from each remaining stratum
        total_available = sum(len(df) for df in remaining_by_stratum.values())
        
        for (freq_tier, length_tier), remaining_df in remaining_by_stratum.items():
            # Proportional allocation
            proportion = len(remaining_df) / total_available
            additional_needed = int(shortfall * proportion)
            additional_needed = min(additional_needed, len(remaining_df))  # Don't exceed available
            
            if additional_needed > 0:
                additional_sample = remaining_df.sample(n=additional_needed, random_state=seed+1)
                sampled_records.append(additional_sample)
                print(f"   + Added {additional_needed:3d} from {freq_tier:6s} × {length_tier:6s}")
    
    final_df = pd.concat(sampled_records, ignore_index=True)
    
    # If still short, sample randomly from all remaining
    current_total = len(final_df)
    if current_total < target_total:
        still_needed = target_total - current_total
        print(f"\n   ⚠️  Still need {still_needed} more samples")
        print(f"   🔄 Adding random samples to reach {target_total}...")
        
        # Get all samples not yet selected
        already_selected_ids = set(final_df['hadm_id'])
        remaining_all = df[~df['hadm_id'].isin(already_selected_ids)]
        
        if len(remaining_all) >= still_needed:
            extra_samples = remaining_all.sample(n=still_needed, random_state=seed+2)
            final_df = pd.concat([final_df, extra_samples], ignore_index=True)
            print(f"   ✓ Added {still_needed} samples to reach exactly {target_total}")
        else:
            print(f"   ⚠️  Warning: Only {len(remaining_all)} samples available, target not reached")
    
    return final_df


def main(args):
    print("="*80)
    print("STRATIFIED SAMPLING FOR LLM EVALUATION")
    print("="*80)
    print()
    
    # Define paths - need to handle both relative and absolute
    if args.input:
        test_file = args.input
    else:
        test_file = os.path.join(MIMIC_4_DIR, 'full_code', 'test_full.csv')
        # Also try absolute path from script location
        if not os.path.exists(test_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            test_file = os.path.join(script_dir, 'mimicdata', 'mimic4_icd10', 'full_code', 'test_full.csv')
    
    train_file = os.path.join(MIMIC_4_DIR, 'full_code', 'train_full.csv')
    if not os.path.exists(train_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(script_dir, 'mimicdata', 'mimic4_icd10', 'full_code', 'train_full.csv')
    
    output_file = args.output
    
    # Load test data
    print(f"📂 Loading test data: {test_file}")
    if not os.path.exists(test_file):
        print(f"❌ ERROR: Test file not found: {test_file}")
        return
    
    df = pd.read_csv(test_file)
    print(f"   ✓ Loaded {len(df):,} records")
    print()
    
    # Parse labels into list
    df['codes_list'] = df['labels'].apply(lambda x: x.split(';'))
    df['num_codes'] = df['codes_list'].apply(len)
    
    # Get code frequencies from training set
    code_counts = load_code_frequencies(train_file)
    print()
    
    # Calculate percentiles for code frequency tiers
    # Use ULTRA aggressive percentiles to get balanced distribution with enough rare/medium
    if code_counts:
        freq_values = sorted(code_counts.values(), reverse=True)
        # Use 99.9th and 99th percentiles - only top 0.1% are "common"
        # This will push most records into rare/medium categories
        percentile_999 = np.percentile(freq_values, 99.9)  # Top 0.1% = common
        percentile_99 = np.percentile(freq_values, 99)     # Top 1% = medium
        
        # Also show some statistics
        max_freq = max(freq_values)
        min_freq = min(freq_values)
        median_freq = np.median(freq_values)
        
        print(f"📈 Code frequency statistics:")
        print(f"   Most frequent code:  {max_freq:,.0f} occurrences")
        print(f"   Median frequency:    {median_freq:.0f} occurrences")
        print(f"   Least frequent:      {min_freq:.0f} occurrence(s)")
        print()
        print(f"📊 Stratification thresholds (using avg code freq per record):")
        print(f"   Common (top 0.1%):   Avg ≥ {percentile_999:.0f} occurrences per code")
        print(f"   Medium (0.1-1%):     Avg {percentile_99:.0f} - {percentile_999:.0f} occurrences")
        print(f"   Rare (bottom 99%):   Avg < {percentile_99:.0f} occurrences")
        
        percentile_67 = percentile_999
        percentile_33 = percentile_99
    else:
        percentile_67 = percentile_33 = None
    print()
    
    # Classify each record by code frequency
    print("🏷️  Classifying records by code frequency...")
    df['code_frequency_tier'] = df['codes_list'].apply(
        lambda codes: classify_code_frequency(codes, code_counts, percentile_67, percentile_33)
    )
    
    # Calculate length percentiles
    length_33 = df['length'].quantile(0.33)
    length_67 = df['length'].quantile(0.67)
    print(f"📏 Text length thresholds:")
    print(f"   Short:  < {length_33:.0f} tokens")
    print(f"   Medium: {length_33:.0f} - {length_67:.0f} tokens")
    print(f"   Long:   > {length_67:.0f} tokens")
    print()
    
    # Classify by length
    print("🏷️  Classifying records by text length...")
    def classify_len(length):
        if length < length_33:
            return 'short'
        elif length < length_67:
            return 'medium'
        else:
            return 'long'
    
    df['length_tier'] = df['length'].apply(classify_len)
    
    # Show distribution
    print()
    print("📊 Distribution in test set:")
    dist_table = pd.crosstab(df['code_frequency_tier'], df['length_tier'], margins=True)
    print(dist_table)
    print()
    
    # Define target distribution for 900 samples (100 in EACH scenario)
    target_distribution = {
        ('common', 'short'):  100,
        ('common', 'medium'): 100,
        ('common', 'long'):   100,
        ('medium', 'short'):  100,
        ('medium', 'medium'): 100,
        ('medium', 'long'):   100,
        ('rare', 'short'):    100,
        ('rare', 'medium'):   100,
        ('rare', 'long'):     100,
    }
    
    print("🎯 Target distribution (900 samples - 100 in EACH scenario):")
    print("                Short    Medium    Long    Total")
    print(f"Common          {target_distribution[('common', 'short')]:3d}      {target_distribution[('common', 'medium')]:3d}      {target_distribution[('common', 'long')]:3d}      300")
    print(f"Medium          {target_distribution[('medium', 'short')]:3d}      {target_distribution[('medium', 'medium')]:3d}      {target_distribution[('medium', 'long')]:3d}      300")
    print(f"Rare            {target_distribution[('rare', 'short')]:3d}      {target_distribution[('rare', 'medium')]:3d}      {target_distribution[('rare', 'long')]:3d}      300")
    print(f"Total           {sum([v for k,v in target_distribution.items() if k[1]=='short']):3d}      {sum([v for k,v in target_distribution.items() if k[1]=='medium']):3d}      {sum([v for k,v in target_distribution.items() if k[1]=='long']):3d}      900")
    print()
    
    # Perform stratified sampling
    print("🎲 Performing stratified sampling...")
    df_sample = stratified_sample(df, target_distribution, seed=args.seed, target_total=900)
    print()
    
    # Verify final distribution
    print("✅ Final sample distribution:")
    final_dist = pd.crosstab(df_sample['code_frequency_tier'], df_sample['length_tier'], margins=True)
    print(final_dist)
    print()
    
    # Add metadata columns
    df_sample['sample_id'] = range(1, len(df_sample) + 1)
    df_sample['dataset'] = 'test'
    df_sample['stratification_group'] = df_sample['code_frequency_tier'] + '_' + df_sample['length_tier']
    
    # Reorder columns for clarity
    column_order = [
        'sample_id',
        'subject_id',
        'hadm_id',
        'text',
        'labels',
        'length',
        'length_tier',
        'code_frequency_tier',
        'num_codes',
        'stratification_group',
        'dataset'
    ]
    
    df_sample = df_sample[column_order]
    
    # Save to CSV
    df_sample.to_csv(output_file, index=False)
    print(f"💾 Saved {len(df_sample)} samples to: {output_file}")
    print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total samples:              {len(df_sample)}")
    print(f"Average text length:        {df_sample['length'].mean():.1f} tokens")
    print(f"Average codes per record:   {df_sample['num_codes'].mean():.1f}")
    print(f"Length range:               {df_sample['length'].min()} - {df_sample['length'].max()} tokens")
    print(f"Codes per record range:     {df_sample['num_codes'].min()} - {df_sample['num_codes'].max()}")
    print()
    
    # Estimated token costs (rough estimates)
    avg_tokens = df_sample['length'].mean()
    total_tokens = df_sample['length'].sum()
    
    print("💰 ESTIMATED LLM API COSTS (Input tokens only):")
    print(f"Total input tokens:         {total_tokens:,.0f}")
    print()
    print("Per model estimates:")
    print(f"  GPT-4o:      ${total_tokens * 2.5 / 1_000_000:.2f}  ($2.50 per 1M input tokens)")
    print(f"  GPT-4:       ${total_tokens * 30 / 1_000_000:.2f}  ($30.00 per 1M input tokens)")
    print(f"  Claude 3.5:  ${total_tokens * 3 / 1_000_000:.2f}  ($3.00 per 1M input tokens)")
    print(f"  Claude 3:    ${total_tokens * 15 / 1_000_000:.2f}  ($15.00 per 1M input tokens)")
    print()
    print("Note: Add output token costs (~50-200 codes per response)")
    print("      Multiply by number of prompts if doing multiple passes")
    print()
    
    print("="*80)
    print("✨ DONE! Ready for LLM evaluation")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create stratified sample for LLM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", type=str,
                        help="Input test CSV file (default: mimicdata/mimic4_icd10/full_code/test_full.csv)")
    parser.add_argument("--output", type=str, default="llm_eval_900_sample.csv",
                        help="Output CSV file (default: llm_eval_900_sample.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    main(args)
