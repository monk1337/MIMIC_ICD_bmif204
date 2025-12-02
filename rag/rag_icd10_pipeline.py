#!/usr/bin/env python3
"""
RAG ICD-10 Pipeline
Complete pipeline: NER → RAG Retrieval → LLM Prediction → Evaluation
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import pipeline components
from ner_disease_extractor import NERDiseaseExtractor
from rag_icd10_retriever import RAGRetriever
from rag_llm_predictor import RAGLLMPredictor


class RAGPipeline:
    """Complete RAG-based ICD-10 coding pipeline"""
    
    def __init__(
        self,
        index_dir: str = "icd10_vector_index",
        ner_model: str = "gpt-4o",
        prediction_model: str = "gpt-4o",
        similarity_top_k: int = 10,
        rerank_top_n: int = 5,
        output_dir: str = "llm_eval_results_900"
    ):
        """Initialize RAG pipeline"""
        
        self.output_dir = Path(output_dir) / "rag_method"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for intermediate results
        self.ner_dir = self.output_dir / "ner_responses"
        self.ner_dir.mkdir(parents=True, exist_ok=True)
        
        self.retrieval_dir = self.output_dir / "retrieval_responses"
        self.retrieval_dir.mkdir(parents=True, exist_ok=True)
        
        self.prediction_dir = self.output_dir / "prediction_responses"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        logging.info("="*80)
        logging.info("RAG ICD-10 CODING PIPELINE")
        logging.info("="*80)
        logging.info(f"NER Model: {ner_model}")
        logging.info(f"Prediction Model: {prediction_model}")
        logging.info(f"Vector Index: {index_dir}")
        logging.info(f"Similarity Top-K: {similarity_top_k}")
        logging.info(f"Rerank Top-N: {rerank_top_n}")
        logging.info("="*80)
        
        # Initialize components
        logging.info("\nInitializing pipeline components...")
        
        # Initialize NER extractor immediately (fast)
        self.ner_extractor = NERDiseaseExtractor(model_name=ner_model)
        logging.info("  ✅ NER extractor ready")
        
        # Store config for lazy loading
        self.index_dir = index_dir
        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n
        self.prediction_model = prediction_model
        
        # Lazy load these (only when needed)
        self._rag_retriever = None
        self._llm_predictor = None
        
        logging.info("  ⚡ RAG retriever and LLM predictor will load on first use (lazy loading)")
        logging.info("\n✅ Pipeline initialized successfully (fast startup enabled)\n")
    
    def setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _save_ner_result(self, sample_id: int, ner_result: dict):
        """Save NER result to file"""
        ner_file = self.ner_dir / f"sample_{sample_id}_ner.json"
        
        # Create a serializable copy (exclude non-serializable objects)
        serializable_result = {
            'success': ner_result.get('success'),
            'entities': ner_result.get('entities', []),
            'raw_response': ner_result.get('raw_response'),
            'error': ner_result.get('error'),
            'usage': {
                'prompt_tokens': ner_result.get('usage', {}).get('prompt_tokens', 0),
                'completion_tokens': ner_result.get('usage', {}).get('completion_tokens', 0),
                'total_tokens': ner_result.get('usage', {}).get('total_tokens', 0)
            } if ner_result.get('usage') else {}
        }
        
        with open(ner_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)
    
    def _load_ner_result(self, sample_id: int):
        """Load NER result from file if exists"""
        ner_file = self.ner_dir / f"sample_{sample_id}_ner.json"
        if ner_file.exists():
            try:
                with open(ner_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Failed to load NER result for sample {sample_id}, will recompute: {str(e)}")
                # Delete corrupted file
                ner_file.unlink()
        return None
    
    def _save_retrieval_result(self, sample_id: int, retrieval_result: dict):
        """Save retrieval result to file"""
        retrieval_file = self.retrieval_dir / f"sample_{sample_id}_retrieval.json"
        
        # Convert to JSON-serializable format
        try:
            with open(retrieval_file, 'w') as f:
                json.dump(retrieval_result, f, indent=2, default=str)
        except Exception as e:
            logging.warning(f"Failed to save retrieval result for sample {sample_id}: {str(e)}")
    
    def _load_retrieval_result(self, sample_id: int):
        """Load retrieval result from file if exists"""
        retrieval_file = self.retrieval_dir / f"sample_{sample_id}_retrieval.json"
        if retrieval_file.exists():
            try:
                with open(retrieval_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Failed to load retrieval result for sample {sample_id}, will recompute: {str(e)}")
                # Delete corrupted file
                retrieval_file.unlink()
        return None
    
    def _save_prediction_result(self, sample_id: int, prediction_result: dict):
        """Save prediction result to file"""
        prediction_file = self.prediction_dir / f"sample_{sample_id}_prediction.json"
        
        # Convert to JSON-serializable format
        try:
            with open(prediction_file, 'w') as f:
                json.dump(prediction_result, f, indent=2, default=str)
        except Exception as e:
            logging.warning(f"Failed to save prediction result for sample {sample_id}: {str(e)}")
    
    def _load_prediction_result(self, sample_id: int):
        """Load prediction result from file if exists"""
        prediction_file = self.prediction_dir / f"sample_{sample_id}_prediction.json"
        if prediction_file.exists():
            try:
                with open(prediction_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Failed to load prediction result for sample {sample_id}, will recompute: {str(e)}")
                # Delete corrupted file
                prediction_file.unlink()
        return None
    
    @property
    def rag_retriever(self):
        """Lazy load RAG retriever (only when first needed)"""
        if self._rag_retriever is None:
            logging.info("\n⏳ Loading RAG retriever (first time, ~60 seconds)...")
            self._rag_retriever = RAGRetriever(
                index_dir=self.index_dir,
                similarity_top_k=self.similarity_top_k,
                rerank_top_n=self.rerank_top_n,
                use_reranking=False  # Disable Jina (use vector-only)
            )
            logging.info("  ✅ RAG retriever loaded (vector-only mode)\n")
        return self._rag_retriever
    
    @property
    def llm_predictor(self):
        """Lazy load LLM predictor (only when first needed)"""
        if self._llm_predictor is None:
            logging.info("⏳ Loading LLM predictor...")
            self._llm_predictor = RAGLLMPredictor(model_name=self.prediction_model)
            logging.info("  ✅ LLM predictor ready\n")
        return self._llm_predictor
    
    def process_sample(self, sample: dict) -> dict:
        """
        Process a single sample through the RAG pipeline
        
        Steps:
        1. Extract medical entities (NER)
        2. Retrieve relevant codes via RAG
        3. Predict final codes with LLM
        """
        
        sample_id = sample['sample_id']
        hadm_id = sample['hadm_id']
        discharge_summary = sample['discharge_summary']
        actual_codes = sample['actual_codes']
        
        # Step 1: NER extraction (check cache first)
        ner_result = self._load_ner_result(sample_id)
        if ner_result:
            logging.info(f"Sample {sample_id}: Using cached NER result")
        else:
            ner_result = self.ner_extractor.extract_diseases(discharge_summary)
            # Save NER result
            self._save_ner_result(sample_id, ner_result)
        
        if not ner_result['success']:
            logging.warning(f"Sample {sample_id}: NER extraction failed - {ner_result.get('error')}")
            return {
                'sample_id': sample_id,
                'hadm_id': hadm_id,
                'success': False,
                'error': f"NER failed: {ner_result.get('error')}",
                'actual_codes': actual_codes,
                'predicted_codes': []
            }
        
        entities = ner_result['entities']
        
        if not entities:
            logging.warning(f"Sample {sample_id}: No entities extracted")
            return {
                'sample_id': sample_id,
                'hadm_id': hadm_id,
                'success': False,
                'error': 'No medical entities extracted',
                'actual_codes': actual_codes,
                'predicted_codes': []
            }
        
        # Step 2: RAG retrieval (check cache first)
        rag_result = self._load_retrieval_result(sample_id)
        if rag_result:
            logging.info(f"Sample {sample_id}: Using cached retrieval result")
            enriched_codes = rag_result['enriched_codes']
        else:
            try:
                rag_result = self.rag_retriever.retrieve_and_enrich(entities)
                enriched_codes = rag_result['enriched_codes']
                # Save retrieval result
                self._save_retrieval_result(sample_id, rag_result)
            except Exception as e:
                logging.error(f"Sample {sample_id}: RAG retrieval failed - {str(e)}")
                return {
                    'sample_id': sample_id,
                    'hadm_id': hadm_id,
                    'success': False,
                    'error': f"RAG retrieval failed: {str(e)}",
                    'actual_codes': actual_codes,
                    'predicted_codes': [],
                    'entities_extracted': entities
                }
        
        # Check if we have enriched codes
        if not enriched_codes:
            logging.warning(f"Sample {sample_id}: No codes retrieved")
            return {
                'sample_id': sample_id,
                'hadm_id': hadm_id,
                'success': False,
                'error': 'No codes retrieved from RAG',
                'actual_codes': actual_codes,
                'predicted_codes': [],
                'entities_extracted': entities,
                'codes_retrieved': 0
            }
        
        # Step 3: LLM prediction
        prediction_result = self.llm_predictor.predict(
            discharge_summary=discharge_summary,
            extracted_entities=entities,
            enriched_codes=enriched_codes
        )
        
        # Save prediction result
        self._save_prediction_result(sample_id, prediction_result)
        
        if not prediction_result['success']:
            logging.warning(f"Sample {sample_id}: Prediction failed - {prediction_result.get('error')}")
            return {
                'sample_id': sample_id,
                'hadm_id': hadm_id,
                'success': False,
                'error': f"Prediction failed: {prediction_result.get('error')}",
                'actual_codes': actual_codes,
                'predicted_codes': [],
                'entities_extracted': entities,
                'codes_retrieved': len(enriched_codes)
            }
        
        # Successful prediction
        predicted_codes = prediction_result['predicted_codes']
        
        # Calculate sample-level metrics
        actual_set = set(actual_codes)
        pred_set = set(predicted_codes)
        correct = actual_set & pred_set
        
        precision = len(correct) / len(pred_set) if pred_set else 0
        recall = len(correct) / len(actual_set) if actual_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'sample_id': sample_id,
            'hadm_id': hadm_id,
            'success': True,
            'actual_codes': actual_codes,
            'predicted_codes': predicted_codes,
            'entities_extracted': entities,
            'num_entities': len(entities),
            'codes_retrieved': len(enriched_codes),
            'codes_with_reasoning': prediction_result.get('codes_with_reasoning', []),
            'overall_reasoning': prediction_result.get('overall_reasoning', ''),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'exact_match': actual_set == pred_set,
            'usage': prediction_result.get('usage', {})
        }
    
    def _process_final_prediction(self, sample: dict, ner_result: dict, rag_result: dict) -> dict:
        """
        Process final LLM prediction using pre-computed NER and RAG results
        
        Args:
            sample: Sample data with sample_id, hadm_id, discharge_summary, actual_codes
            ner_result: Pre-computed NER result
            rag_result: Pre-computed RAG retrieval result
            
        Returns:
            Complete result dictionary
        """
        sample_id = sample['sample_id']
        hadm_id = sample['hadm_id']
        discharge_summary = sample['discharge_summary']
        actual_codes = sample['actual_codes']
        
        # Check if NER failed
        if not ner_result or not ner_result['success']:
            return {
                'sample_id': sample_id,
                'hadm_id': hadm_id,
                'success': False,
                'error': f"NER failed: {ner_result.get('error') if ner_result else 'No result'}",
                'actual_codes': actual_codes,
                'predicted_codes': []
            }
        
        entities = ner_result.get('entities', [])
        
        # Check if no entities extracted
        if not entities:
            return {
                'sample_id': sample_id,
                'hadm_id': hadm_id,
                'success': False,
                'error': 'No medical entities extracted',
                'actual_codes': actual_codes,
                'predicted_codes': []
            }
        
        # Check if RAG retrieval failed
        if not rag_result or not rag_result.get('enriched_codes'):
            return {
                'sample_id': sample_id,
                'hadm_id': hadm_id,
                'success': False,
                'error': 'No codes retrieved from RAG',
                'actual_codes': actual_codes,
                'predicted_codes': [],
                'entities_extracted': entities,
                'codes_retrieved': 0
            }
        
        enriched_codes = rag_result['enriched_codes']
        
        # Check if prediction already cached
        prediction_result = self._load_prediction_result(sample_id)
        
        if not prediction_result:
            # LLM prediction
            prediction_result = self.llm_predictor.predict(
                discharge_summary=discharge_summary,
                extracted_entities=entities,
                enriched_codes=enriched_codes
            )
            
            # Save prediction result
            self._save_prediction_result(sample_id, prediction_result)
        
        if not prediction_result['success']:
            return {
                'sample_id': sample_id,
                'hadm_id': hadm_id,
                'success': False,
                'error': f"Prediction failed: {prediction_result.get('error')}",
                'actual_codes': actual_codes,
                'predicted_codes': [],
                'entities_extracted': entities,
                'codes_retrieved': len(enriched_codes)
            }
        
        predicted_codes = prediction_result.get('predicted_codes', [])
        
        # Calculate metrics
        actual_set = set(actual_codes)
        pred_set = set(predicted_codes)
        
        if len(pred_set) > 0:
            precision = len(actual_set & pred_set) / len(pred_set)
        else:
            precision = 0.0
        
        if len(actual_set) > 0:
            recall = len(actual_set & pred_set) / len(actual_set)
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'sample_id': sample_id,
            'hadm_id': hadm_id,
            'success': True,
            'actual_codes': actual_codes,
            'predicted_codes': predicted_codes,
            'entities_extracted': entities,
            'num_entities': len(entities),
            'codes_retrieved': len(enriched_codes),
            'codes_with_reasoning': prediction_result.get('codes_with_reasoning', []),
            'overall_reasoning': prediction_result.get('overall_reasoning', ''),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'exact_match': actual_set == pred_set,
            'usage': prediction_result.get('usage', {})
        }
    
    def run(self, data_file: str, limit: int = None, filter_rare_only: bool = False):
        """
        Run pipeline on dataset with optimized staged processing
        
        Args:
            data_file: Path to enriched CSV file
            limit: Optional limit on number of samples
            filter_rare_only: If True, only process samples with rare codes
        """
        
        logging.info(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        
        # Parse labels
        df['actual_codes'] = df['labels'].apply(
            lambda x: [c.replace('.', '').upper().strip() for c in str(x).split(';') if c.strip()]
        )
        
        # Filter for rare codes only if requested
        if filter_rare_only:
            original_count = len(df)
            df = df[df['code_frequency_tier'] == 'rare'].copy()
            logging.info(f"🔍 Filtered for RARE codes only: {len(df)} samples (from {original_count} total)")
            logging.info(f"   Rare samples: {len(df)} ({len(df)/original_count*100:.1f}%)")
        
        if limit:
            df = df.head(limit)
            logging.info(f"Processing {limit} samples (limited)")
        else:
            logging.info(f"Processing {len(df)} samples")
        
        # Prepare samples
        samples = []
        for idx, row in df.iterrows():
            samples.append({
                'sample_id': row['sample_id'],
                'hadm_id': row['hadm_id'],
                'discharge_summary': row['text'],
                'actual_codes': row['actual_codes']
            })
        
        # ==================================================================
        # STAGE 1: Process ALL NER extractions first (no vector loading yet)
        # ==================================================================
        logging.info("\n" + "="*80)
        logging.info("🔹 STAGE 1/4: NER Extraction for all samples")
        logging.info("="*80)
        ner_results = {}
        for sample in tqdm(samples, desc="  NER Extraction"):
            sample_id = sample['sample_id']
            
            # Check cache
            ner_result = self._load_ner_result(sample_id)
            if not ner_result:
                ner_result = self.ner_extractor.extract_diseases(sample['discharge_summary'])
                self._save_ner_result(sample_id, ner_result)
            
            ner_results[sample_id] = ner_result
        
        logging.info(f"  ✅ NER extraction completed for {len(samples)} samples")
        
        # ==================================================================
        # STAGE 2: Load RAG retriever ONCE (60 seconds, but only once!)
        # ==================================================================
        logging.info("\n" + "="*80)
        logging.info("🔹 STAGE 2/4: Loading RAG components (one-time, ~60 seconds)")
        logging.info("="*80)
        _ = self.rag_retriever  # Trigger lazy loading
        _ = self.llm_predictor   # Trigger lazy loading
        logging.info("  ✅ All components loaded")
        
        # ==================================================================
        # STAGE 3: Process ALL RAG retrievals
        # ==================================================================
        logging.info("\n" + "="*80)
        logging.info("🔹 STAGE 3/4: RAG Retrieval for all samples")
        logging.info("="*80)
        retrieval_results = {}
        for sample in tqdm(samples, desc="  RAG Retrieval"):
            sample_id = sample['sample_id']
            ner_result = ner_results[sample_id]
            
            if not ner_result['success'] or not ner_result.get('entities'):
                retrieval_results[sample_id] = None
                continue
            
            # Check cache
            rag_result = self._load_retrieval_result(sample_id)
            if not rag_result:
                try:
                    rag_result = self.rag_retriever.retrieve_and_enrich(ner_result['entities'])
                    self._save_retrieval_result(sample_id, rag_result)
                except Exception as e:
                    logging.error(f"  Sample {sample_id}: RAG retrieval failed - {str(e)}")
                    rag_result = None
            
            retrieval_results[sample_id] = rag_result
        
        logging.info(f"  ✅ RAG retrieval completed for {len(samples)} samples")
        
        # ==================================================================
        # STAGE 4: Process ALL LLM predictions
        # ==================================================================
        logging.info("\n" + "="*80)
        logging.info("🔹 STAGE 4/4: LLM Prediction for all samples")
        logging.info("="*80)
        results = []
        for sample in tqdm(samples, desc="  LLM Prediction"):
            result = self._process_final_prediction(
                sample, 
                ner_results[sample['sample_id']], 
                retrieval_results[sample['sample_id']]
            )
            results.append(result)
            
            # Save intermediate results periodically
            if len(results) % 50 == 0:
                self.save_results(results, intermediate=True)
        
        logging.info(f"  ✅ Pipeline completed for {len(samples)} samples")
        
        # Save final results
        self.save_results(results, intermediate=False)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def save_results(self, results: list, intermediate: bool = False):
        """Save results to JSON and CSV"""
        
        suffix = "_intermediate" if intermediate else "_results"
        
        # Save JSON (with fallback for non-serializable objects)
        json_file = self.output_dir / f"rag_method{suffix}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV (simplified)
        csv_data = []
        for r in results:
            csv_data.append({
                'sample_id': r['sample_id'],
                'hadm_id': r['hadm_id'],
                'success': r['success'],
                'actual_codes': ';'.join(r.get('actual_codes', [])),
                'predicted_codes': ';'.join(r.get('predicted_codes', [])),
                'num_entities': r.get('num_entities', 0),
                'codes_retrieved': r.get('codes_retrieved', 0),
                'precision': r.get('precision', 0),
                'recall': r.get('recall', 0),
                'f1_score': r.get('f1_score', 0),
                'exact_match': r.get('exact_match', False)
            })
        
        csv_file = self.output_dir / f"rag_method{suffix}.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        if not intermediate:
            logging.info(f"\n✅ Results saved:")
            logging.info(f"   JSON: {json_file}")
            logging.info(f"   CSV: {csv_file}")
    
    def print_summary(self, results: list):
        """Print summary statistics"""
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        logging.info("\n" + "="*80)
        logging.info("PIPELINE SUMMARY")
        logging.info("="*80)
        logging.info(f"Total samples: {len(results)}")
        logging.info(f"Successful: {len(successful)}")
        logging.info(f"Failed: {len(failed)}")
        
        if successful:
            avg_precision = sum(r['precision'] for r in successful) / len(successful)
            avg_recall = sum(r['recall'] for r in successful) / len(successful)
            avg_f1 = sum(r['f1_score'] for r in successful) / len(successful)
            exact_matches = sum(1 for r in successful if r['exact_match'])
            
            logging.info(f"\nAverage Metrics:")
            logging.info(f"  Precision: {avg_precision:.4f}")
            logging.info(f"  Recall: {avg_recall:.4f}")
            logging.info(f"  F1 Score: {avg_f1:.4f}")
            logging.info(f"  Exact Matches: {exact_matches} ({exact_matches/len(successful)*100:.1f}%)")
            
            avg_entities = sum(r['num_entities'] for r in successful) / len(successful)
            avg_retrieved = sum(r['codes_retrieved'] for r in successful) / len(successful)
            
            logging.info(f"\nPipeline Stats:")
            logging.info(f"  Avg entities extracted: {avg_entities:.1f}")
            logging.info(f"  Avg codes retrieved: {avg_retrieved:.1f}")
        
        logging.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='RAG ICD-10 Coding Pipeline')
    parser.add_argument('--data-file', default='llm_eval_900_enriched.csv',
                       help='Input CSV file with discharge summaries')
    parser.add_argument('--index-dir', default='icd10_vector_index',
                       help='Vector index directory')
    parser.add_argument('--ner-model', default='gpt-4o',
                       help='Model for NER extraction')
    parser.add_argument('--prediction-model', default='gpt-4o',
                       help='Model for final prediction')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of candidates for similarity search')
    parser.add_argument('--rerank-n', type=int, default=5,
                       help='Number of candidates after reranking')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples (for testing)')
    parser.add_argument('--rare-only', action='store_true',
                       help='Process only samples with rare codes')
    parser.add_argument('--output-dir', default='llm_eval_results_900',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        index_dir=args.index_dir,
        ner_model=args.ner_model,
        prediction_model=args.prediction_model,
        similarity_top_k=args.top_k,
        rerank_top_n=args.rerank_n,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    results = pipeline.run(
        data_file=args.data_file,
        limit=args.limit,
        filter_rare_only=args.rare_only
    )
    
    print("\n✅ Pipeline completed!")


if __name__ == '__main__':
    main()
