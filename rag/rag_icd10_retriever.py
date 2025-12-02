#!/usr/bin/env python3
"""
RAG ICD-10 Retriever
Retrieves relevant ICD-10 codes using vector similarity and Jina reranking
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import threading
import time
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Optional: Jina reranker (install with: pip install llama-index-postprocessor-jinaai-rerank)
try:
    from llama_index.postprocessor.jinaai_rerank import JinaRerank
    JINA_AVAILABLE = True
except ImportError:
    JINA_AVAILABLE = False
    print("Warning: Jina reranker not available. Install with: pip install llama-index-postprocessor-jinaai-rerank")

# Import ICD-10 knowledge graph for enrichment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import icd10


class LoadingSpinner:
    """Simple loading spinner for long operations"""
    def __init__(self, message="Loading"):
        self.message = message
        self.running = False
        self.thread = None
    
    def spinner_task(self):
        """Spinner animation"""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        idx = 0
        while self.running:
            print(f"\r   {chars[idx]} {self.message}...", end="", flush=True)
            idx = (idx + 1) % len(chars)
            time.sleep(0.1)
        print("\r" + " " * (len(self.message) + 10), end="\r", flush=True)
    
    def start(self):
        """Start the spinner"""
        self.running = True
        self.thread = threading.Thread(target=self.spinner_task, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the spinner"""
        self.running = False
        if self.thread:
            self.thread.join()


class RAGRetriever:
    """Retrieve ICD-10 codes using RAG with vector similarity and reranking"""
    
    def __init__(
        self, 
        index_dir: str,
        similarity_top_k: int = 10,
        rerank_top_n: int = 5,
        jina_api_key: str = None,
        use_reranking: bool = True
    ):
        """
        Initialize RAG retriever with vector index and optional Jina reranker
        
        Args:
            index_dir: Directory containing the vector index
            similarity_top_k: Number of candidates for similarity search
            rerank_top_n: Number of candidates after reranking
            jina_api_key: Jina AI API key for reranking
            use_reranking: Whether to use Jina reranking (default True)
        """
        self.index_dir = Path(index_dir)
        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n
        self.use_reranking = use_reranking
        
        # Setup embeddings
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        embed_model = OpenAIEmbedding(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        Settings.embed_model = embed_model
        
        # Load vector index
        print(f"Loading vector index from {index_dir}...")
        
        spinner = LoadingSpinner("Loading vector store (95K+ codes)")
        spinner.start()
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
            self.index = load_index_from_storage(storage_context)
        finally:
            spinner.stop()
        print("   ✅ Vector index loaded")
        
        # Load metadata
        spinner = LoadingSpinner("Loading code metadata")
        spinner.start()
        try:
            metadata_file = self.index_dir / "code_metadata.json"
            with open(metadata_file, 'r') as f:
                self.metadata_map = json.load(f)
        finally:
            spinner.stop()
        
        print(f"✅ Index loaded with {len(self.metadata_map)} codes")
        
        # Setup Jina reranker
        jina_api_key = os.environ.get("JINA_API_KEY")
        if not jina_api_key:
            print("⚠️  Warning: JINA_API_KEY not set, reranking will be skipped")
        
        # Initialize Jina reranker (only if enabled and available)
        if self.use_reranking and jina_api_key and JINA_AVAILABLE:
            self.jina_rerank = JinaRerank(
                api_key=jina_api_key,
                model="jina-reranker-v2-base-multilingual",
                top_n=self.rerank_top_n
            )
            
            # Create query engine with reranking
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.similarity_top_k,
                node_postprocessors=[self.jina_rerank]
            )
            print("✅ Jina reranker enabled")
        else:
            # Use only vector similarity without reranking
            if self.use_reranking and not JINA_AVAILABLE:
                print("⚠️  Warning: Reranking disabled - Jina package not installed")
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.rerank_top_n  # Use fewer results without reranking
            )
    
    def retrieve_codes_for_entity(self, entity_name: str) -> List[Dict]:
        """
        Retrieve top ICD-10 codes for a given medical entity
        
        Args:
            entity_name: Medical condition/disease name
            
        Returns:
            List of code dictionaries with scores
        """
        try:
            # Query the index (with Jina reranking if available)
            response = self.query_engine.query(entity_name)
        except Exception as e:
            # If Jina fails (e.g., out of credits), fall back to vector-only
            if "Insufficient balance" in str(e) or "quota" in str(e).lower():
                print(f"    ⚠️  Jina reranking failed (quota), using vector-only results")
                # Create query engine without reranking
                fallback_engine = self.index.as_query_engine(
                    similarity_top_k=self.rerank_top_n
                )
                response = fallback_engine.query(entity_name)
            else:
                raise  # Re-raise if it's a different error
        
        # Extract codes and scores
        codes = []
        for node in response.source_nodes:
            code = node.metadata['code']
            score = node.score
            
            codes.append({
                'code': code,
                'score': float(score),
                'entity_matched': entity_name
            })
        
        return codes
    
    def enrich_code_with_kg(self, code: str) -> Dict:
        """
        Enrich code with full ICD-10 knowledge graph hierarchy
        (Same as ensemble method)
        """
        try:
            full_data = icd10.get_full_data(code)
            
            if not full_data:
                return {
                    'code': code,
                    'description': 'Code not found',
                    'parent': None,
                    'children': [],
                    'parent_chain': []
                }
            
            return {
                'code': full_data['code'],
                'description': full_data['description'],
                'parent': full_data.get('parent'),
                'children': full_data.get('children', []),
                'parent_chain': full_data.get('parentChain', []),
                'type': full_data.get('type', 'diagnosis')
            }
            
        except Exception as e:
            return {
                'code': code,
                'description': f'Error enriching code: {str(e)}',
                'parent': None,
                'children': [],
                'parent_chain': []
            }
    
    def retrieve_and_enrich(self, disease_entities: List[str]) -> Dict:
        """
        Retrieve codes for all entities and enrich with KG hierarchy
        
        Args:
            disease_entities: List of disease/condition names
            
        Returns:
            {
                'entities': List of entities processed,
                'retrieved_codes': Dict mapping entity -> codes,
                'all_codes_deduped': List of unique codes with enrichment,
                'total_retrieved': int,
                'total_unique': int
            }
        """
        print(f"\nRetrieving codes for {len(disease_entities)} entities...")
        
        retrieved_per_entity = {}
        all_codes = {}  # Use dict to deduplicate by code
        
        for entity in disease_entities:
            print(f"  Query: '{entity}'")
            
            # Retrieve codes
            codes = self.retrieve_codes_for_entity(entity)
            retrieved_per_entity[entity] = codes
            
            print(f"    Found {len(codes)} codes")
            
            # Add to global set
            for code_info in codes:
                code = code_info['code']
                if code not in all_codes:
                    all_codes[code] = {
                        'code': code,
                        'matched_entities': [entity],
                        'scores': [code_info['score']]
                    }
                else:
                    # Code already seen from another entity
                    all_codes[code]['matched_entities'].append(entity)
                    all_codes[code]['scores'].append(code_info['score'])
        
        print(f"\nTotal unique codes retrieved: {len(all_codes)}")
        
        # Enrich all unique codes with KG
        print("Enriching codes with ICD-10 knowledge graph...")
        enriched_codes = {}
        
        for code, info in all_codes.items():
            enriched = self.enrich_code_with_kg(code)
            enriched['matched_entities'] = info['matched_entities']
            enriched['retrieval_scores'] = info['scores']
            enriched['avg_score'] = sum(info['scores']) / len(info['scores'])
            
            enriched_codes[code] = enriched
        
        print(f"✅ Enriched {len(enriched_codes)} codes")
        
        return {
            'entities': disease_entities,
            'retrieved_per_entity': retrieved_per_entity,
            'enriched_codes': enriched_codes,
            'total_retrieved': sum(len(codes) for codes in retrieved_per_entity.values()),
            'total_unique': len(enriched_codes)
        }


def test_rag_retriever():
    """Test RAG retriever with sample entities"""
    
    print("="*80)
    print("TESTING RAG ICD-10 RETRIEVER")
    print("="*80)
    print()
    
    # Sample entities
    test_entities = [
        "atrial fibrillation",
        "diabetes mellitus type 2",
        "myocardial infarction",
        "congestive heart failure"
    ]
    
    # Initialize retriever
    retriever = RAGRetriever(
        index_dir="icd10_vector_index",
        similarity_top_k=10,
        rerank_top_n=5
    )
    
    # Retrieve and enrich
    result = retriever.retrieve_and_enrich(test_entities)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print()
    
    print(f"Entities processed: {len(result['entities'])}")
    print(f"Total codes retrieved: {result['total_retrieved']}")
    print(f"Unique codes: {result['total_unique']}")
    print()
    
    # Show enriched codes
    print("Sample enriched codes:")
    print("-"*80)
    
    for i, (code, data) in enumerate(list(result['enriched_codes'].items())[:5], 1):
        print(f"\n{i}. Code: {code}")
        print(f"   Description: {data['description']}")
        print(f"   Matched entities: {', '.join(data['matched_entities'])}")
        print(f"   Avg similarity score: {data['avg_score']:.4f}")
        
        if data['parent']:
            print(f"   Parent: {data['parent']}")
        
        if data['children']:
            print(f"   Children: {len(data['children'])} codes")
    
    print()
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG retrieval for ICD-10 codes')
    parser.add_argument('--test', action='store_true',
                       help='Run test with sample entities')
    parser.add_argument('--entities', nargs='+',
                       help='Medical entities to retrieve codes for')
    parser.add_argument('--index-dir', default='icd10_vector_index',
                       help='Vector index directory')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of candidates for similarity search')
    parser.add_argument('--rerank-n', type=int, default=5,
                       help='Number of candidates after reranking')
    
    args = parser.parse_args()
    
    if args.test:
        test_rag_retriever()
    elif args.entities:
        retriever = RAGRetriever(
            index_dir=args.index_dir,
            similarity_top_k=args.top_k,
            rerank_top_n=args.rerank_n
        )
        result = retriever.retrieve_and_enrich(args.entities)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
