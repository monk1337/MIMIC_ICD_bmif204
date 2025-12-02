#!/usr/bin/env python3
"""
Ensemble LLM Arbiter - Final Code Selection
Combines CNN + GPT-4o predictions with ICD-10 Knowledge Graph
Uses GPT-4o as final arbiter to select most appropriate codes
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging
from tqdm import tqdm
import time
import traceback
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LiteLLM imports
from litellm import completion
import litellm


class EnsembleLLMArbiter:
    """LLM Arbiter for ensemble method with knowledge graph enrichment"""
    
    def __init__(self, model_name: str = "gpt-4o", output_dir: str = "llm_eval_results_900"):
        """Initialize arbiter"""
        self.model_name = model_name
        self.output_dir = Path(output_dir) / "ensemble_arbiter"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory for individual API responses
        self.api_responses_dir = self.output_dir / "api_responses"
        self.api_responses_dir.mkdir(parents=True, exist_ok=True)
        
        # Disable litellm verbose logging
        litellm.set_verbose = False
        
        self.setup_logging()
        
        logging.info("="*80)
        logging.info(f"ENSEMBLE LLM ARBITER - Model: {model_name}")
        logging.info("="*80)
    
    def setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / f"{self.model_name}_arbiter_progress.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
    
    def load_ensemble_data(self, ensemble_file: str) -> List[Dict]:
        """Load ensemble dataset with KG enrichment"""
        logging.info(f"Loading ensemble data: {ensemble_file}")
        
        # Check file extension to determine format
        if ensemble_file.endswith('.csv'):
            # Load CSV file
            df = pd.read_csv(ensemble_file)
            
            # Map CSV column names to expected names
            column_mapping = {
                'text': 'discharge_summary',
                'labels': 'actual_codes',
                'cnn_predicted_codes': 'cnn_codes',
                'gpt4o_predicted_codes': 'gpt4o_codes',
                'candidate_codes_union': 'candidate_codes',
                'kg_enriched_codes': 'enriched_codes'
            }
            
            df = df.rename(columns=column_mapping)
            data = df.to_dict('records')
            
            # Parse JSON strings in certain columns
            json_columns = ['actual_codes', 'cnn_codes', 'gpt4o_codes', 'candidate_codes', 'enriched_codes']
            for sample in data:
                for col in json_columns:
                    if col in sample and isinstance(sample[col], str):
                        try:
                            # Try to parse JSON string
                            sample[col] = json.loads(sample[col])
                        except (json.JSONDecodeError, TypeError):
                            # If it fails, try eval for Python literals
                            try:
                                import ast
                                sample[col] = ast.literal_eval(sample[col])
                            except:
                                # If both fail, set to empty dict/list
                                sample[col] = {} if col == 'enriched_codes' else []
        else:
            # Load JSON file
            with open(ensemble_file, 'r') as f:
                data = json.load(f)
        
        logging.info(f"Loaded {len(data)} samples")
        return data
    
    def create_prompt(self, sample: Dict) -> List[Dict[str, str]]:
        """
        Create detailed prompt for LLM arbiter with:
        - Discharge summary
        - Candidate codes with full ICD-10 hierarchy
        - Instructions for step-by-step reasoning
        """
        
        discharge_summary = sample['discharge_summary']
        candidate_codes = sample.get('candidate_codes', sample.get('candidate_codes_union', []))
        kg_enriched = sample.get('enriched_codes', sample.get('kg_enriched_codes', {}))
        
        # Build detailed candidate code context
        candidate_context = self._build_candidate_context(candidate_codes, kg_enriched)
        
        system_message = """You are an expert medical coding specialist with deep knowledge of ICD-10-CM coding guidelines. Your task is to select the most accurate and appropriate ICD-10 codes from a set of candidate codes.

You will be provided with:
1. A clinical discharge summary
2. A set of candidate ICD-10 codes with full hierarchical context from the ICD-10 knowledge graph
3. Each candidate code's full hierarchy including parents, children, and descriptions

Your goal is to select the BEST subset of codes that:
- Are clinically supported by evidence in the discharge summary
- Follow ICD-10 coding guidelines and conventions
- Represent the complete picture of diagnoses and conditions documented
- Avoid redundancy (don't code both parent and child unless appropriate)
- Prioritize specificity (use most specific codes when supported)

IMPORTANT: You must provide:
1. Step-by-step clinical reasoning for EACH code you consider
2. Validation against the discharge summary text
3. Clear justification for inclusion or exclusion
4. Your final selected codes"""

        user_message = f"""# CLINICAL DISCHARGE SUMMARY

{discharge_summary}

---

# CANDIDATE ICD-10 CODES WITH HIERARCHICAL CONTEXT

You have been provided with {len(candidate_codes)} validated candidate codes to choose from. Review each code carefully and select the most appropriate ones based on the clinical documentation.

{candidate_context}

---

# YOUR TASK

Please analyze the discharge summary and candidate codes, then provide your response in the following JSON format:

```json
{{
  "codes_with_reasoning": [
    {{
      "code": "ICD-10 code",
      "description": "Code description",
      "reasoning": "Detailed clinical reasoning for including/excluding this code. Reference specific text from discharge summary. Explain why this is the most appropriate level of specificity.",
      "evidence_quotes": ["quote from discharge summary supporting this code"],
      "decision": "INCLUDE or EXCLUDE",
      "confidence": "HIGH or MEDIUM or LOW"
    }}
  ],
  "final_codes": ["list", "of", "selected", "ICD10", "codes"],
  "overall_short_reasoning": "Summary of your coding decisions and any important considerations about code interactions, hierarchy, or guidelines that influenced your selections.",
  }}
```

INSTRUCTIONS:
1. Review ALL candidate codes systematically
2. For each code, identify supporting evidence in the discharge summary
3. Consider hierarchical relationships (parent/child codes)
4. Apply ICD-10 coding guidelines
5. Select the most appropriate final code set
6. Provide clear reasoning for each decision

Begin your analysis:"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def _build_candidate_context(self, candidate_codes: List[str], kg_enriched: Dict) -> str:
        """Build detailed context for each candidate code with hierarchy"""
        
        context_parts = []
        
        for i, code in enumerate(candidate_codes, 1):
            enriched = kg_enriched.get(code, {})
            
            section = f"""## Candidate {i}: {code}

**Description:** {enriched.get('description', 'N/A')}
**Type:** {enriched.get('type', 'N/A')}
**Parent:** {enriched.get('parent', 'None')}
**Children:** {', '.join(enriched.get('children', [])) if enriched.get('children') else 'None (leaf code)'}

**Hierarchical Context:**"""
            
            # Add parent chain
            parent_chain = enriched.get('parent_chain', [])
            if parent_chain:
                section += "\n```"
                for parent in parent_chain:
                    section += f"\n  ↑ {parent['code']} ({parent['type']}): {parent['description']}"
                section += "\n```"
            else:
                section += " None (top-level code)"
            
            context_parts.append(section)
        
        return "\n\n".join(context_parts)
    
    def call_llm(self, messages: List[Dict]) -> Dict:
        """Call LLM with retry logic"""
        max_retries = 3
        retry_delay = 2
        timeout = 120
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4000,
                    timeout=timeout,
                    response_format={"type": "json_object"}
                )
                
                end_time = time.time()
                
                result = {
                    'success': True,
                    'model': response.model,
                    'content': response.choices[0].message.content,
                    'finish_reason': response.choices[0].finish_reason,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'response_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                return result
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'timestamp': datetime.now().isoformat()
                    }
    
    def parse_response(self, response_content: str) -> Dict:
        """Parse LLM JSON response"""
        try:
            # Clean up response if it has markdown code blocks
            content = response_content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            parsed = json.loads(content)
            
            return {
                'codes_with_reasoning': parsed.get('codes_with_reasoning', []),
                'final_codes': parsed.get('final_codes', []),
                'overall_reasoning': parsed.get('overall_short_reasoning', parsed.get('overall_reasoning', '')),
                'coding_guidelines_applied': parsed.get('coding_guidelines_applied', []),
                'response_parsed': True
            }
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {str(e)}")
            return {
                'codes_with_reasoning': [],
                'final_codes': [],
                'overall_reasoning': response_content,
                'coding_guidelines_applied': [],
                'response_parsed': False,
                'parse_error': str(e)
            }
    
    def _save_api_response(self, sample_id: int, response: Dict, messages: List[Dict]):
        """Save individual API response to JSON file"""
        api_response_file = self.api_responses_dir / f"sample_{sample_id}_response.json"
        
        response_data = {
            'sample_id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'messages': messages,
            'response': response
        }
        
        with open(api_response_file, 'w') as f:
            json.dump(response_data, f, indent=2)
    
    def process_sample(self, sample: Dict) -> Dict:
        """Process a single sample through the arbiter"""
        
        # Create prompt
        messages = self.create_prompt(sample)
        
        # Call LLM
        response = self.call_llm(messages)
        
        # Save individual API response
        self._save_api_response(sample['sample_id'], response, messages)
        
        if not response['success']:
            return {
                'sample_id': sample['sample_id'],
                'hadm_id': sample['hadm_id'],
                'success': False,
                'error': response.get('error', 'Unknown error'),
                'actual_codes': sample['actual_codes'],
                'predicted_codes': [],
                'cnn_codes': sample.get('cnn_codes', sample.get('cnn_predicted_codes', [])),
                'gpt4o_codes': sample.get('gpt4o_codes', sample.get('gpt4o_predicted_codes', [])),
                'candidate_codes': sample.get('candidate_codes', sample.get('candidate_codes_union', []))
            }
        
        # Parse response
        parsed = self.parse_response(response['content'])
        
        # Extract predicted codes
        predicted_codes = parsed['final_codes']
        
        # Normalize codes
        predicted_codes = [c.replace('.', '').upper().strip() for c in predicted_codes]
        actual_codes = sample['actual_codes']
        
        # Calculate metrics
        actual_set = set(actual_codes)
        predicted_set = set(predicted_codes)
        
        correct = actual_set & predicted_set
        
        precision = len(correct) / len(predicted_set) if predicted_set else 0
        recall = len(correct) / len(actual_set) if actual_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        jaccard = len(correct) / len(actual_set | predicted_set) if (actual_set | predicted_set) else 0
        exact_match = 1 if actual_set == predicted_set else 0
        
        result = {
            'sample_id': sample['sample_id'],
            'hadm_id': sample['hadm_id'],
            'model': f'ensemble-arbiter-{self.model_name}',
            'success': True,
            
            # Codes
            'actual_codes': list(actual_codes),
            'predicted_codes': predicted_codes,
            'codes_actual': list(actual_codes),
            'codes_predicted': predicted_codes,
            'codes_correct': list(correct),
            
            # Source codes
            'cnn_predicted_codes': sample.get('cnn_codes', sample.get('cnn_predicted_codes', [])),
            'gpt4o_predicted_codes': sample.get('gpt4o_codes', sample.get('gpt4o_predicted_codes', [])),
            'candidate_codes_union': sample.get('candidate_codes', sample.get('candidate_codes_union', [])),
            'invalid_codes_filtered': sample.get('invalid_codes', sample.get('invalid_codes_filtered', [])),
            
            # Counts
            'num_codes_actual': len(actual_codes),
            'num_codes_predicted': len(predicted_codes),
            'num_codes_correct': len(correct),
            
            # Metrics
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard_similarity': jaccard,
            'exact_match': exact_match,
            
            # Reasoning and validation
            'codes_with_reasoning': parsed['codes_with_reasoning'],
            'overall_reasoning': parsed['overall_reasoning'],
            'coding_guidelines_applied': parsed['coding_guidelines_applied'],
            'response_parsed': parsed['response_parsed'],
            
            # Metadata
            'code_frequency_tier': sample.get('code_frequency_tier', ''),
            'length_tier': sample.get('length_tier', ''),
            'stratification_group': sample.get('stratification_group', f"{sample.get('code_frequency_tier', '')}_{sample.get('length_tier', '')}"),
            
            # Response info
            'usage': response['usage'],
            'response_time': response['response_time'],
            'timestamp': response['timestamp']
        }
        
        return result
    
    def run(self, ensemble_file: str, limit: int = None):
        """Run arbiter on all samples"""
        
        # Load data
        samples = self.load_ensemble_data(ensemble_file)
        
        if limit:
            samples = samples[:limit]
            logging.info(f"Processing first {limit} samples")
        
        # Process samples
        results = []
        
        logging.info(f"Processing {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="Processing samples"):
            try:
                result = self.process_sample(sample)
                results.append(result)
                
                # Save intermediate results every 50 samples
                if len(results) % 50 == 0:
                    self._save_results(results)
                    
            except Exception as e:
                logging.error(f"Error processing sample {sample['sample_id']}: {str(e)}")
                logging.error(traceback.format_exc())
                
                results.append({
                    'sample_id': sample['sample_id'],
                    'hadm_id': sample['hadm_id'],
                    'success': False,
                    'error': str(e),
                    'actual_codes': sample['actual_codes'],
                    'predicted_codes': []
                })
        
        # Final save
        self._save_results(results)
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: List[Dict]):
        """Save results to JSON and CSV"""
        
        # Save intermediate JSON
        intermediate_file = self.output_dir / f"{self.model_name}_arbiter_intermediate.json"
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save results JSON (simplified for metrics calculation)
        results_file = self.output_dir / f"{self.model_name}_arbiter_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV
        csv_file = self.output_dir / f"{self.model_name}_arbiter_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        
        logging.info(f"Saved {len(results)} results to {self.output_dir}")
    
    def _print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            logging.warning("No successful results to summarize")
            return
        
        avg_precision = sum(r['precision'] for r in successful) / len(successful)
        avg_recall = sum(r['recall'] for r in successful) / len(successful)
        avg_f1 = sum(r['f1_score'] for r in successful) / len(successful)
        avg_jaccard = sum(r['jaccard_similarity'] for r in successful) / len(successful)
        exact_matches = sum(r['exact_match'] for r in successful)
        
        logging.info("="*80)
        logging.info("ENSEMBLE ARBITER SUMMARY")
        logging.info("="*80)
        logging.info(f"Total samples: {len(results)}")
        logging.info(f"Successful: {len(successful)}")
        logging.info(f"Failed: {len(results) - len(successful)}")
        logging.info("")
        logging.info(f"Average Precision: {avg_precision:.4f}")
        logging.info(f"Average Recall: {avg_recall:.4f}")
        logging.info(f"Average F1 Score: {avg_f1:.4f}")
        logging.info(f"Average Jaccard: {avg_jaccard:.4f}")
        logging.info(f"Exact Matches: {exact_matches} ({100*exact_matches/len(successful):.2f}%)")
        logging.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Ensemble LLM Arbiter')
    parser.add_argument('--ensemble-file', '-e', type=str, 
                       default='ensemble_dataset_900.csv',
                       help='Path to ensemble dataset CSV file')
    parser.add_argument('--model', '-m', type=str,
                       default='gpt-4o',
                       help='LLM model to use as arbiter')
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='Limit number of samples to process (for testing)')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='llm_eval_results_900',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create arbiter
    arbiter = EnsembleLLMArbiter(
        model_name=args.model,
        output_dir=args.output_dir
    )
    
    # Run
    arbiter.run(
        ensemble_file=args.ensemble_file,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
