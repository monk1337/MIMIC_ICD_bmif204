#!/usr/bin/env python3
"""
LLM Results Evaluator - Step 2 of Evaluation
Load saved API responses and calculate metrics matching eval_250.py format
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import logging
import argparse
import re
import random
import string

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Metrics - use the same comprehensive evaluation as training
from utils import evaluation

# Simple sklearn metrics for per-sample analysis
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score


class LLMResultsEvaluator:
    """Evaluate saved LLM responses"""
    
    def __init__(self, results_folder: str):
        """Initialize evaluator with results folder"""
        self.results_folder = Path(results_folder)
        
        if not self.results_folder.exists():
            raise ValueError(f"Results folder not found: {results_folder}")
        
        self.setup_logging()
        self.load_results()
        self.load_metadata()
        
        logging.info("="*80)
        logging.info(f"EVALUATING RESULTS FROM: {self.results_folder.name}")
        logging.info("="*80)
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    def load_results(self):
        """Load saved API responses"""
        # Try final results first, then intermediate
        responses_file = self.results_folder / "responses_final.json"
        if not responses_file.exists():
            responses_file = self.results_folder / "responses_intermediate.json"
        
        if not responses_file.exists():
            raise ValueError(f"No responses file found in {self.results_folder}")
        
        logging.info(f"Loading responses from: {responses_file.name}")
        
        with open(responses_file, 'r') as f:
            self.results = json.load(f)
        
        logging.info(f"Loaded {len(self.results)} results")
    
    def load_metadata(self):
        """Load metadata"""
        metadata_file = self.results_folder / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            logging.info(f"Model: {self.metadata.get('model_path', 'Unknown')}")
        else:
            self.metadata = {}
            logging.warning("No metadata file found")
    
    def normalize_icd_code(self, code: str) -> str:
        """Normalize ICD-10 code by removing dots/periods"""
        return code.replace('.', '').replace(' ', '').upper().strip()
    
    def parse_llm_response(self, content: str) -> Dict:
        """Parse LLM response to extract codes and reasoning"""
        try:
            # Try to parse as JSON
            data = json.loads(content)
            
            # Extract codes and reasoning
            codes_with_reasoning = []
            codes_only = []
            
            if 'codes' in data:
                for item in data['codes']:
                    if isinstance(item, dict):
                        code = item.get('code', '').strip()
                        reasoning = item.get('reasoning', '').strip()
                        if code:
                            # Normalize code (remove dots)
                            normalized = self.normalize_icd_code(code)
                            codes_only.append(normalized)
                            codes_with_reasoning.append({
                                'code': normalized,
                                'original_code': code,  # Keep original for display
                                'reasoning': reasoning
                            })
                    elif isinstance(item, str):
                        normalized = self.normalize_icd_code(item.strip())
                        codes_only.append(normalized)
                        codes_with_reasoning.append({
                            'code': normalized,
                            'original_code': item.strip(),
                            'reasoning': ''
                        })
            
            return {
                'parsed': True,
                'codes': set(codes_only),
                'codes_with_reasoning': codes_with_reasoning
            }
            
        except json.JSONDecodeError:
            logging.debug("Failed to parse JSON response, using fallback extraction")
            # Fallback: extract ICD-10 codes using regex
            pattern = r'\b[A-Z]\d{1,3}(?:\.\d{1,4})?\b'
            codes = set(self.normalize_icd_code(c) for c in re.findall(pattern, content))
            
            return {
                'parsed': False,
                'codes': codes,
                'codes_with_reasoning': [{'code': c, 'original_code': c, 'reasoning': ''} for c in codes]
            }
    
    def calculate_metrics(self, predicted: Set, actual: Set) -> Dict:
        """Calculate evaluation metrics"""
        all_codes = sorted(predicted | actual)
        
        if not all_codes:
            return {
                'exact_match': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'jaccard_similarity': 0.0,
                'codes_predicted': 0,
                'codes_actual': len(actual),
                'codes_correct': 0,
                'codes_missed': len(actual),
                'codes_extra': 0
            }
        
        # Binary vectors for sklearn
        pred_vector = [1 if code in predicted else 0 for code in all_codes]
        actual_vector = [1 if code in actual else 0 for code in all_codes]
        
        # Calculate metrics
        exact_match = 1 if predicted == actual else 0
        precision = precision_score(actual_vector, pred_vector, zero_division=0)
        recall = recall_score(actual_vector, pred_vector, zero_division=0)
        f1 = f1_score(actual_vector, pred_vector, zero_division=0)
        jaccard = jaccard_score(actual_vector, pred_vector, zero_division=0)
        
        # Detailed counts
        correct = predicted & actual
        missed = actual - predicted
        extra = predicted - actual
        
        return {
            'exact_match': exact_match,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'jaccard_similarity': float(jaccard),
            'codes_predicted': len(predicted),
            'codes_actual': len(actual),
            'codes_correct': len(correct),
            'codes_missed': len(missed),
            'codes_extra': len(extra),
            'correct_codes': sorted(correct),
            'missed_codes': sorted(missed),
            'extra_codes': sorted(extra)
        }
    
    def evaluate_result(self, result: Dict) -> Dict:
        """Evaluate a single result"""
        llm_response = result['llm_response']
        
        if not llm_response['success']:
            return {
                'sample_id': result['sample_id'],
                'success': False,
                'error': llm_response.get('error', 'Unknown error')
            }
        
        # Parse response
        content = llm_response['content']
        parsed = self.parse_llm_response(content)
        
        # Get actual codes and normalize them too
        actual_codes = set(self.normalize_icd_code(c) for c in result['actual_codes'])
        predicted_codes = parsed['codes']
        
        # Calculate metrics
        metrics = self.calculate_metrics(predicted_codes, actual_codes)
        
        # Build evaluation result
        evaluation = {
            'sample_id': result['sample_id'],
            'hadm_id': result['hadm_id'],
            'subject_id': result['subject_id'],
            'success': True,
            'stratification_group': result['stratification_group'],
            'code_frequency_tier': result['code_frequency_tier'],
            'length_tier': result['length_tier'],
            'text_length': result['text_length'],
            'response_parsed': parsed['parsed'],
            'actual_codes': sorted(actual_codes),
            'predicted_codes': sorted(predicted_codes),
            'codes_with_reasoning': parsed['codes_with_reasoning'],
            'reasoning_content': llm_response.get('reasoning_content'),
            **metrics,
            'usage': llm_response.get('usage', {}),
            'response_time': llm_response.get('response_time', 0)
        }
        
        return evaluation
    
    def evaluate_all(self):
        """Evaluate all results"""
        logging.info(f"\nEvaluating {len(self.results)} results...")
        
        evaluations = []
        
        for result in self.results:
            evaluation = self.evaluate_result(result)
            evaluations.append(evaluation)
        
        self.evaluations = evaluations
        self.successful = [e for e in evaluations if e['success']]
        
        logging.info(f"Evaluation complete!")
        logging.info(f"  Successful: {len(self.successful)}")
        logging.info(f"  Failed: {len(evaluations) - len(self.successful)}")
        
        return evaluations
    
    def calculate_comprehensive_metrics(self) -> Dict:
        """Calculate comprehensive metrics matching neural network evaluation"""
        if not self.successful:
            return {}
        
        # Get all unique codes across all samples
        all_codes = set()
        for e in self.successful:
            all_codes.update(e['actual_codes'])
            all_codes.update(e['predicted_codes'])
        
        all_codes_sorted = sorted(all_codes)
        code_to_idx = {code: idx for idx, code in enumerate(all_codes_sorted)}
        num_labels = len(all_codes_sorted)
        
        logging.info(f"Building multi-hot vectors for {num_labels} unique codes...")
        
        # Build multi-hot vectors
        y_true = np.zeros((len(self.successful), num_labels))
        y_pred = np.zeros((len(self.successful), num_labels))
        y_scores = np.zeros((len(self.successful), num_labels))
        
        for i, e in enumerate(self.successful):
            # True labels
            for code in e['actual_codes']:
                if code in code_to_idx:
                    y_true[i, code_to_idx[code]] = 1
            
            # Predicted labels (binary - LLMs don't give scores)
            for code in e['predicted_codes']:
                if code in code_to_idx:
                    y_pred[i, code_to_idx[code]] = 1
                    y_scores[i, code_to_idx[code]] = 1.0  # Binary confidence
        
        logging.info("Calculating comprehensive metrics...")
        
        # Use the same evaluation function as trained models
        metrics = evaluation.all_metrics(
            y_pred, y_true, 
            k=[5, 8, 15],  # Standard @k metrics
            yhat_raw=y_scores, 
            calc_auc=True
        )
        
        return metrics
    
    def calculate_stratified_metrics(self, y_pred, y_true, y_scores, metadata_list):
        """Calculate metrics stratified by length_tier and code_frequency_tier"""
        from collections import defaultdict
        
        stratified = {
            'by_length': {},
            'by_code_freq': {},
            'cross_tab': {},
            'sample_counts': {}
        }
        
        # Group indices by strata
        length_indices = defaultdict(list)
        freq_indices = defaultdict(list)
        cross_indices = defaultdict(list)
        
        for idx, meta in enumerate(metadata_list):
            length_tier = meta['length_tier']
            freq_tier = meta['code_frequency_tier']
            
            length_indices[length_tier].append(idx)
            freq_indices[freq_tier].append(idx)
            cross_indices[(freq_tier, length_tier)].append(idx)
        
        # Calculate metrics by length tier
        for length_tier, indices in length_indices.items():
            indices = np.array(indices)
            metrics = evaluation.all_metrics(
                y_pred[indices], y_true[indices], 
                k=[5, 8, 15], yhat_raw=y_scores[indices], calc_auc=True
            )
            stratified['by_length'][length_tier] = metrics
            stratified['sample_counts'][f'length_{length_tier}'] = len(indices)
        
        # Calculate metrics by code frequency tier
        for freq_tier, indices in freq_indices.items():
            indices = np.array(indices)
            metrics = evaluation.all_metrics(
                y_pred[indices], y_true[indices], 
                k=[5, 8, 15], yhat_raw=y_scores[indices], calc_auc=True
            )
            stratified['by_code_freq'][freq_tier] = metrics
            stratified['sample_counts'][f'freq_{freq_tier}'] = len(indices)
        
        # Calculate metrics by cross-tabulation
        for (freq_tier, length_tier), indices in cross_indices.items():
            indices = np.array(indices)
            metrics = evaluation.all_metrics(
                y_pred[indices], y_true[indices], 
                k=[5, 8, 15], yhat_raw=y_scores[indices], calc_auc=True
            )
            stratified['cross_tab'][(freq_tier, length_tier)] = metrics
            stratified['sample_counts'][f'{freq_tier}_{length_tier}'] = len(indices)
        
        return stratified
    
    def calculate_summary_stats(self) -> Dict:
        """Calculate summary statistics with stratification"""
        if not self.successful:
            return {}
        
        # Get all unique codes across all samples
        all_codes = set()
        for e in self.successful:
            all_codes.update(e['actual_codes'])
            all_codes.update(e['predicted_codes'])
        
        all_codes_sorted = sorted(all_codes)
        code_to_idx = {code: idx for idx, code in enumerate(all_codes_sorted)}
        num_labels = len(all_codes_sorted)
        
        # Build multi-hot vectors and metadata
        y_true = np.zeros((len(self.successful), num_labels))
        y_pred = np.zeros((len(self.successful), num_labels))
        y_scores = np.zeros((len(self.successful), num_labels))
        metadata_list = []
        
        for i, e in enumerate(self.successful):
            # True labels
            for code in e['actual_codes']:
                if code in code_to_idx:
                    y_true[i, code_to_idx[code]] = 1
            
            # Predicted labels
            for code in e['predicted_codes']:
                if code in code_to_idx:
                    y_pred[i, code_to_idx[code]] = 1
                    y_scores[i, code_to_idx[code]] = 1.0
            
            # Metadata for stratification
            metadata_list.append({
                'length_tier': e['length_tier'],
                'code_frequency_tier': e['code_frequency_tier']
            })
        
        # Calculate overall metrics
        overall_metrics = evaluation.all_metrics(
            y_pred, y_true, 
            k=[5, 8, 15], 
            yhat_raw=y_scores, 
            calc_auc=True
        )
        
        # Calculate stratified metrics
        stratified_metrics = self.calculate_stratified_metrics(
            y_pred, y_true, y_scores, metadata_list
        )
        
        summary = {
            'model': self.metadata.get('model_path', 'Unknown'),
            'folder_name': self.results_folder.name,
            'total_samples': len(self.evaluations),
            'successful': len(self.successful),
            'failed': len(self.evaluations) - len(self.successful),
            'overall': overall_metrics,
            'stratified': stratified_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def print_metrics(self, metrics):
        """Print metrics in formatted way (matching eval_250.py)"""
        print("\n" + "="*80)
        print("OVERALL EVALUATION RESULTS")
        print("="*80)
        
        print("\n[MACRO] accuracy, precision, recall, f-measure, AUC")
        print(f"{metrics.get('acc_macro', 0):.4f}, {metrics.get('prec_macro', 0):.4f}, "
              f"{metrics.get('rec_macro', 0):.4f}, {metrics.get('f1_macro', 0):.4f}, "
              f"{metrics.get('auc_macro', 0):.4f}")
        
        print("\n[MICRO] accuracy, precision, recall, f-measure, AUC")
        print(f"{metrics.get('acc_micro', 0):.4f}, {metrics.get('prec_micro', 0):.4f}, "
              f"{metrics.get('rec_micro', 0):.4f}, {metrics.get('f1_micro', 0):.4f}, "
              f"{metrics.get('auc_micro', 0):.4f}")
        
        # @k metrics
        for k in [5, 8, 15]:
            if f'rec_at_{k}' in metrics:
                print(f"\nrec_at_{k}: {metrics[f'rec_at_{k}']:.4f}")
                print(f"prec_at_{k}: {metrics[f'prec_at_{k}']:.4f}")
                if f'f1_at_{k}' in metrics:
                    print(f"f1_at_{k}: {metrics[f'f1_at_{k}']:.4f}")
        
        print("\n" + "="*80 + "\n")
    
    def print_stratified_metrics(self, stratified, overall_metrics):
        """Print stratified metrics in matrix format (matching eval_250.py)"""
        stratified['overall'] = overall_metrics
        
        print("\n" + "="*80)
        print("STRATIFIED EVALUATION RESULTS")
        print("="*80)
        
        freq_tiers = ['common', 'medium', 'rare']
        length_tiers = ['short', 'medium', 'long']
        
        # Print sample count matrix
        print("\n" + "─"*80)
        print("SAMPLE COUNT CROSS-TABULATION MATRIX")
        print("─"*80)
        print(f"{'code_frequency_tier':<25} {'short':>12} {'medium':>12} {'long':>12} {'All':>12}")
        print("─"*80)
        
        for freq_tier in freq_tiers:
            counts = []
            for length_tier in length_tiers:
                count = stratified['sample_counts'].get(f'{freq_tier}_{length_tier}', 0)
                counts.append(count)
            total = stratified['sample_counts'].get(f'freq_{freq_tier}', sum(counts))
            print(f"{freq_tier:<25} {counts[0]:>12} {counts[1]:>12} {counts[2]:>12} {total:>12}")
        
        print("─"*80)
        totals = [stratified['sample_counts'].get(f'length_{lt}', 0) for lt in length_tiers]
        grand_total = sum(totals)
        print(f"{'All':<25} {totals[0]:>12} {totals[1]:>12} {totals[2]:>12} {grand_total:>12}")
        print()
        
        # Define metric names for cross-tabulation
        metric_names = [
            ('F1 Micro', 'f1_micro'),
            ('F1 Macro', 'f1_macro'),
            ('Precision Micro', 'prec_micro'),
            ('Precision Macro', 'prec_macro'),
            ('Recall Micro', 'rec_micro'),
            ('Recall Macro', 'rec_macro'),
            ('AUC Micro', 'auc_micro'),
            ('AUC Macro', 'auc_macro'),
        ]
        
        # Print cross-tabulation matrices for each metric with All row and column
        for metric_name, metric_key in metric_names:
            print("\n" + "─"*80)
            print(f"{metric_name.upper()} - CROSS-TABULATION MATRIX")
            print("─"*80)
            print(f"{'code_frequency_tier':<25} {'short':>12} {'medium':>12} {'long':>12} {'All':>12}")
            print("─"*80)
            
            for freq_tier in freq_tiers:
                values = []
                for length_tier in length_tiers:
                    val = stratified['cross_tab'].get((freq_tier, length_tier), {}).get(metric_key, 0)
                    values.append(val)
                # All column = metric for this freq_tier across all lengths
                all_col = stratified['by_code_freq'].get(freq_tier, {}).get(metric_key, 0)
                print(f"{freq_tier:<25} {values[0]:>12.4f} {values[1]:>12.4f} {values[2]:>12.4f} {all_col:>12.4f}")
            
            print("─"*80)
            # All row = metric for each length_tier across all freq_tiers, plus overall
            all_row = []
            for length_tier in length_tiers:
                val = stratified['by_length'].get(length_tier, {}).get(metric_key, 0)
                all_row.append(val)
            # Bottom-right = overall metric
            overall_val = stratified.get('overall', {}).get(metric_key, 0)
            print(f"{'All':<25} {all_row[0]:>12.4f} {all_row[1]:>12.4f} {all_row[2]:>12.4f} {overall_val:>12.4f}")
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self):
        """Save evaluation results with model name in filename"""
        # Extract model name from folder
        folder_name = self.results_folder.name
        
        # Generate timestamp and random suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        
        # Calculate summary
        summary = self.calculate_summary_stats()
        
        # Create comprehensive results JSON with model name
        json_filename = f"eval_results_{folder_name}_{timestamp}_{random_suffix}.json"
        json_file = self.results_folder / json_filename
        
        combined_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': summary['model'],
                'folder_name': folder_name,
                'total_samples': summary['total_samples'],
                'successful': summary['successful'],
                'failed': summary['failed']
            },
            'overall': summary['overall'],
            'stratified': {
                'by_length': summary['stratified']['by_length'],
                'by_code_freq': summary['stratified']['by_code_freq'],
                'cross_tab': {f"{k[0]}_{k[1]}": v for k, v in summary['stratified']['cross_tab'].items()},
                'sample_counts': summary['stratified']['sample_counts']
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Comprehensive results saved to: {json_filename}")
        print(f"{'='*80}\n")
        
        # Also save detailed evaluations
        eval_file = self.results_folder / "evaluations_detailed.json"
        with open(eval_file, 'w') as f:
            json.dump(self.evaluations, f, indent=2, default=str)
        
        # Save CSV
        df = pd.DataFrame(self.evaluations)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict, set)) else x)
        
        csv_file = self.results_folder / "evaluations.csv"
        df.to_csv(csv_file, index=False)
        
        logging.info(f"✅ Additional files saved:")
        logging.info(f"   - {eval_file.name}")
        logging.info(f"   - {csv_file.name}")
    
    def print_summary(self):
        """Print summary to console matching eval_250.py format"""
        summary = self.calculate_summary_stats()
        
        # Print overall metrics
        self.print_metrics(summary['overall'])
        
        # Print stratified metrics
        self.print_stratified_metrics(summary['stratified'], summary['overall'])


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Evaluate saved LLM responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 llm_evaluate_results.py --folder openai_gpt-4o_20241111_223045
  python3 llm_evaluate_results.py --folder llm_eval_results/anthropic_claude-3-5-sonnet_20241111_223100
        """
    )
    
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing saved API responses")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = LLMResultsEvaluator(results_folder=args.folder)
        
        # Run evaluation
        evaluator.evaluate_all()
        
        # Print summary (matching eval_250.py format)
        evaluator.print_summary()
        
        # Save results (JSON with model name)
        evaluator.save_results()
        
        print(f"✅ Evaluation complete!")
        print(f"📁 Results saved to: {evaluator.results_folder}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
