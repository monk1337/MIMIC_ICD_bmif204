#!/usr/bin/env python3
"""
Create Ensemble Dataset - Merge CNN + GPT-4o predictions with discharge summaries
Enrich candidate codes with ICD-10 knowledge graph hierarchy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from utils import icd10 as cm

def normalize_code(code):
    """Normalize ICD-10 code by removing dots and standardizing"""
    if not code:
        return ""
    return str(code).replace('.', '').replace(' ', '').upper().strip()

def normalize_codes_list(codes):
    """Normalize a list of codes"""
    return [normalize_code(c) for c in codes if c]

def is_valid_icd10_code(code):
    """
    Check if a code exists in the ICD-10 knowledge graph
    Returns True if valid, False otherwise
    """
    if not code or not isinstance(code, str):
        return False
    
    try:
        # First try the built-in validation
        is_valid = cm.is_valid_item(code)
        if not is_valid:
            return False
        
        # Double-check by trying to get the data
        # If this raises an exception, the code doesn't exist
        _ = cm.get_full_data(code, search_in_ancestors=False, prioritize_blocks=False)
        return True
    except Exception as e:
        # Any exception means the code is invalid
        return False

def filter_valid_codes(codes_list):
    """
    Filter a list of codes to only include valid ICD-10 codes
    Returns tuple: (valid_codes, invalid_codes)
    """
    valid = []
    invalid = []
    
    for code in codes_list:
        if code and is_valid_icd10_code(code):
            valid.append(code)
        elif code:
            invalid.append(code)
    
    return valid, invalid

def enrich_code_with_kg(code):
    """
    Enrich a single ICD-10 code with knowledge graph data
    Returns dict with code hierarchy, description, parents, children
    NOTE: This assumes the code has already been validated
    """
    # Get full data from ICD-10 knowledge graph
    kg_data = cm.get_full_data(code, search_in_ancestors=True, prioritize_blocks=False)
    
    # Extract key information for LLM context
    enriched = {
        "code": kg_data.get("code", code),
        "description": kg_data.get("description", ""),
        "type": kg_data.get("type", ""),
        "parent": kg_data.get("parent", ""),
        "children": kg_data.get("children", []),
        "parent_chain": []
    }
    
    # Build simplified parent chain (just codes and descriptions)
    if "parentChain" in kg_data:
        for parent_code, parent_data in kg_data["parentChain"].items():
            enriched["parent_chain"].append({
                "code": parent_code,
                "description": parent_data.get("description", ""),
                "type": parent_data.get("type", "")
            })
    
    return enriched

def enrich_codes_with_kg(codes_list):
    """
    Enrich a list of ICD-10 codes with knowledge graph data
    Returns dict mapping code -> enriched data
    NOTE: Only valid codes should be passed to this function
    """
    enriched_codes = {}
    for code in codes_list:
        if code:  # Skip empty codes
            try:
                enriched_codes[code] = enrich_code_with_kg(code)
            except Exception as e:
                # This shouldn't happen if validation worked properly
                # But add as safeguard
                print(f"    ⚠️  Warning: Failed to enrich code '{code}': {str(e)}")
                continue
    return enriched_codes

def main():
    print("="*80)
    print("Creating Ensemble Dataset")
    print("="*80)
    print()
    
    # ========================================================================
    # 1. Load discharge summaries and metadata
    # ========================================================================
    print("1️⃣  Loading discharge summaries and metadata...")
    df_enriched = pd.read_csv('llm_eval_900_enriched.csv')
    print(f"  ✓ Loaded {len(df_enriched)} samples from enriched CSV")
    print(f"  ✓ Columns: {list(df_enriched.columns)[:10]}...")
    print()
    
    # ========================================================================
    # 2. Load CNN predictions
    # ========================================================================
    print("2️⃣  Loading CNN predictions...")
    with open('llm_eval_results_900/convnet_predictions_900.json', 'r') as f:
        cnn_data = json.load(f)
    
    df_cnn = pd.DataFrame(cnn_data)
    df_cnn = df_cnn[['sample_id', 'predicted_codes', 'actual_codes']]
    df_cnn.rename(columns={
        'predicted_codes': 'cnn_predicted_codes',
        'actual_codes': 'cnn_actual_codes'
    }, inplace=True)
    print(f"  ✓ Loaded {len(df_cnn)} CNN predictions")
    print()
    
    # ========================================================================
    # 3. Load GPT-4o predictions
    # ========================================================================
    print("3️⃣  Loading GPT-4o predictions...")
    with open('llm_eval_results_900/gemini_qween_deepseek_claude4/gpt-4o_intermediate.json', 'r') as f:
        gpt_data = json.load(f)
    
    df_gpt = pd.DataFrame(gpt_data)
    df_gpt = df_gpt[['sample_id', 'predicted_codes', 'reasoning_content', 'codes_with_reasoning']]
    df_gpt.rename(columns={
        'predicted_codes': 'gpt4o_predicted_codes',
        'reasoning_content': 'gpt4o_reasoning',
        'codes_with_reasoning': 'gpt4o_codes_with_reasoning'
    }, inplace=True)
    print(f"  ✓ Loaded {len(df_gpt)} GPT-4o predictions")
    print()
    
    # ========================================================================
    # 4. Merge all datasets
    # ========================================================================
    print("4️⃣  Merging datasets...")
    df_merged = df_enriched.merge(df_cnn, on='sample_id', how='left')
    df_merged = df_merged.merge(df_gpt, on='sample_id', how='left')
    print(f"  ✓ Merged {len(df_merged)} samples")
    print()
    
    # ========================================================================
    # 5. Create union of candidate codes and validate against ICD-10
    # ========================================================================
    print("5️⃣  Creating union of CNN + GPT-4o codes and validating...")
    
    def create_candidate_union_validated(row):
        """Create union of CNN and GPT-4o predictions, keeping only valid ICD-10 codes"""
        cnn_codes = row['cnn_predicted_codes'] if isinstance(row['cnn_predicted_codes'], list) else []
        gpt_codes = row['gpt4o_predicted_codes'] if isinstance(row['gpt4o_predicted_codes'], list) else []
        
        # Normalize all codes
        cnn_normalized = normalize_codes_list(cnn_codes)
        gpt_normalized = normalize_codes_list(gpt_codes)
        
        # Create union (remove duplicates)
        union = list(set(cnn_normalized + gpt_normalized))
        
        # Validate against ICD-10 knowledge graph
        valid_codes, invalid_codes = filter_valid_codes(union)
        
        return {
            'valid': valid_codes,
            'invalid': invalid_codes,
            'total_before': len(union),
            'valid_count': len(valid_codes),
            'invalid_count': len(invalid_codes)
        }
    
    df_merged['candidate_validation'] = df_merged.apply(create_candidate_union_validated, axis=1)
    
    # Extract valid codes as the final union
    df_merged['candidate_codes_union'] = df_merged['candidate_validation'].apply(lambda x: x['valid'])
    df_merged['invalid_codes'] = df_merged['candidate_validation'].apply(lambda x: x['invalid'])
    
    # Count statistics
    df_merged['num_cnn_codes'] = df_merged['cnn_predicted_codes'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df_merged['num_gpt4o_codes'] = df_merged['gpt4o_predicted_codes'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df_merged['num_candidate_codes'] = df_merged['candidate_codes_union'].apply(len)
    
    # Calculate validation statistics
    total_before = df_merged['candidate_validation'].apply(lambda x: x['total_before']).sum()
    total_valid = df_merged['candidate_validation'].apply(lambda x: x['valid_count']).sum()
    total_invalid = df_merged['candidate_validation'].apply(lambda x: x['invalid_count']).sum()
    
    print(f"  ✓ Average CNN codes per sample: {df_merged['num_cnn_codes'].mean():.2f}")
    print(f"  ✓ Average GPT-4o codes per sample: {df_merged['num_gpt4o_codes'].mean():.2f}")
    print(f"  ✓ Average candidate codes before validation: {total_before / len(df_merged):.2f}")
    print(f"  ✓ Average valid codes (in ICD-10): {df_merged['num_candidate_codes'].mean():.2f}")
    print(f"  ✓ Total codes validated: {total_before}")
    print(f"  ✓ Valid codes: {total_valid} ({100*total_valid/total_before:.1f}%)")
    print(f"  ✓ Invalid codes filtered out: {total_invalid} ({100*total_invalid/total_before:.1f}%)")
    
    # Show some examples of invalid codes
    all_invalid = []
    for invalid_list in df_merged['invalid_codes']:
        all_invalid.extend(invalid_list)
    
    if all_invalid:
        from collections import Counter
        invalid_counter = Counter(all_invalid)
        print(f"  ⚠️  Top 10 most common invalid codes:")
        for code, count in invalid_counter.most_common(10):
            print(f"     - {code}: {count} occurrences")
    
    print()
    
    # ========================================================================
    # 5b. Enrich candidate codes with ICD-10 Knowledge Graph
    # ========================================================================
    print("5️⃣b Enriching candidate codes with ICD-10 hierarchy...")
    
    def enrich_row_codes(row):
        """Enrich candidate codes with KG data for each row"""
        candidate_codes = row['candidate_codes_union']
        if not candidate_codes:
            return {}
        return enrich_codes_with_kg(candidate_codes)
    
    # This may take a few minutes for 900 samples
    from tqdm import tqdm
    tqdm.pandas(desc="Enriching codes")
    
    df_merged['kg_enriched_codes'] = df_merged.progress_apply(enrich_row_codes, axis=1)
    
    # Count how many codes were successfully enriched
    total_enriched = sum(len(codes) for codes in df_merged['kg_enriched_codes'])
    print(f"  ✓ Enriched {total_enriched} unique codes with KG hierarchy data")
    print()
    
    # ========================================================================
    # 6. Save ensemble dataset
    # ========================================================================
    print("6️⃣  Saving ensemble dataset...")
    
    output_path = 'ensemble_dataset_900.csv'
    df_merged.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ Shape: {df_merged.shape}")
    print()
    
    # ========================================================================
    # 7. Create a JSON version with structured data
    # ========================================================================
    print("7️⃣  Creating JSON version...")
    
    ensemble_data = []
    for _, row in df_merged.iterrows():
        # Parse labels/actual codes safely
        try:
            if isinstance(row['labels'], str):
                # Try to parse as JSON
                if row['labels'].strip():
                    actual_codes = json.loads(row['labels'])
                else:
                    actual_codes = []
            elif isinstance(row['labels'], list):
                actual_codes = row['labels']
            else:
                # Use from CNN or GPT if available
                actual_codes = row['cnn_actual_codes'] if isinstance(row['cnn_actual_codes'], list) else []
        except (json.JSONDecodeError, ValueError):
            # Fallback: try splitting by semicolon (format: "S62512B;S62611A;...")
            if isinstance(row['labels'], str):
                actual_codes = [c.strip() for c in row['labels'].split(';') if c.strip()]
            else:
                actual_codes = row['cnn_actual_codes'] if isinstance(row['cnn_actual_codes'], list) else []
        
        sample = {
            'sample_id': int(row['sample_id']),
            'hadm_id': int(row['hadm_id']),
            'discharge_summary': row['text'],
            'actual_codes': actual_codes,
            'cnn_predicted_codes': row['cnn_predicted_codes'] if isinstance(row['cnn_predicted_codes'], list) else [],
            'gpt4o_predicted_codes': row['gpt4o_predicted_codes'] if isinstance(row['gpt4o_predicted_codes'], list) else [],
            'candidate_codes_union': row['candidate_codes_union'],
            'invalid_codes_filtered': row['invalid_codes'] if isinstance(row['invalid_codes'], list) else [],
            'kg_enriched_codes': row['kg_enriched_codes'] if isinstance(row['kg_enriched_codes'], dict) else {},
            'gpt4o_reasoning': row['gpt4o_reasoning'] if pd.notna(row['gpt4o_reasoning']) else "",
            'validation_stats': {
                'total_before_validation': row['candidate_validation']['total_before'] if isinstance(row['candidate_validation'], dict) else 0,
                'valid_count': row['candidate_validation']['valid_count'] if isinstance(row['candidate_validation'], dict) else 0,
                'invalid_count': row['candidate_validation']['invalid_count'] if isinstance(row['candidate_validation'], dict) else 0
            },
            'metadata': {
                'gender': row['gender'] if pd.notna(row['gender']) else "",
                'age_group': row['age_group'] if pd.notna(row['age_group']) else "",
                'race_group': row['race_group'] if pd.notna(row['race_group']) else "",
                'code_frequency_tier': row['code_frequency_tier'] if pd.notna(row['code_frequency_tier']) else "",
                'length_tier': row['length_tier'] if pd.notna(row['length_tier']) else "",
                'comorbidity_tier': row['comorbidity_tier'] if pd.notna(row['comorbidity_tier']) else "",
                'num_codes': int(row['num_codes']) if pd.notna(row['num_codes']) else 0
            }
        }
        ensemble_data.append(sample)
    
    output_json_path = 'ensemble_dataset_900.json'
    with open(output_json_path, 'w') as f:
        json.dump(ensemble_data, f, indent=2)
    
    print(f"  ✓ Saved to: {output_json_path}")
    print()
    
    # ========================================================================
    # 8. Summary statistics
    # ========================================================================
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Total Samples: {len(df_merged)}")
    print()
    print("Columns in CSV:")
    for col in df_merged.columns:
        print(f"  - {col}")
    print()
    print("✅ Ensemble dataset with ICD-10 KG enrichment ready for final LLM arbiter!")
    print()
    print("Next Steps:")
    print("  1. Use ensemble_dataset_900.json as input")
    print("  2. For each sample, send to final LLM arbiter:")
    print("     - Discharge summary")
    print("     - Candidate codes with full hierarchy")
    print("     - CNN + GPT-4o reasoning context")
    print("  3. LLM selects final codes based on clinical evidence + hierarchy")
    print()

if __name__ == '__main__':
    main()
