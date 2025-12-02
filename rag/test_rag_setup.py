#!/usr/bin/env python3
"""
Test RAG Pipeline Setup
Verifies all components are working correctly
"""

import os
import sys
from pathlib import Path

def test_api_keys():
    """Test API keys are set"""
    print("Testing API keys...")
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    jina_key = os.environ.get("JINA_API_KEY")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    print("✅ OPENAI_API_KEY found")
    
    if not jina_key:
        print("⚠️  JINA_API_KEY not set (optional, but recommended)")
    else:
        print("✅ JINA_API_KEY found")
    
    return True

def test_imports():
    """Test all required imports"""
    print("\nTesting imports...")
    
    try:
        import litellm
        print("✅ litellm")
    except ImportError:
        print("❌ litellm not installed: pip install litellm")
        return False
    
    try:
        import pandas
        print("✅ pandas")
    except ImportError:
        print("❌ pandas not installed: pip install pandas")
        return False
    
    try:
        from llama_index.core import VectorStoreIndex
        print("✅ llama_index.core")
    except ImportError:
        print("❌ llama_index not installed: pip install llama-index")
        return False
    
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        print("✅ llama_index.embeddings.openai")
    except ImportError:
        print("❌ llama-index-embeddings-openai not installed")
        print("   Run: pip install llama-index-embeddings-openai")
        return False
    
    try:
        from llama_index.postprocessor.jinaai_rerank import JinaRerank
        print("✅ llama_index.postprocessor.jinaai_rerank")
    except ImportError:
        print("⚠️  llama-index-postprocessor-jinaai not installed (optional)")
        print("   Run: pip install llama-index-postprocessor-jinaai")
    
    try:
        from tqdm import tqdm
        print("✅ tqdm")
    except ImportError:
        print("❌ tqdm not installed: pip install tqdm")
        return False
    
    try:
        from tenacity import retry
        print("✅ tenacity")
    except ImportError:
        print("❌ tenacity not installed: pip install tenacity")
        return False
    
    return True

def test_icd10_module():
    """Test ICD-10 module"""
    print("\nTesting ICD-10 module...")
    
    try:
        import icd10
        
        # Test a known code
        test_code = "I50"
        data = icd10.get_full_data(test_code)
        
        if data:
            print(f"✅ ICD-10 module working (tested code: {test_code})")
            print(f"   Description: {data.get('description', 'N/A')[:50]}...")
            return True
        else:
            print(f"❌ ICD-10 module not returning data")
            return False
    except Exception as e:
        print(f"❌ ICD-10 module error: {str(e)}")
        return False

def test_data_files():
    """Test required data files exist"""
    print("\nTesting data files...")
    
    enriched_file = Path("llm_eval_900_enriched.csv")
    if enriched_file.exists():
        print(f"✅ Found: {enriched_file}")
    else:
        print(f"❌ Missing: {enriched_file}")
        return False
    
    return True

def test_pipeline_modules():
    """Test pipeline modules can be imported"""
    print("\nTesting pipeline modules...")
    
    try:
        from ner_disease_extractor import NERDiseaseExtractor
        print("✅ ner_disease_extractor")
    except Exception as e:
        print(f"❌ ner_disease_extractor: {str(e)}")
        return False
    
    # Don't test RAG retriever or predictor yet (need index)
    print("✅ Pipeline modules ready")
    
    return True

def main():
    print("="*80)
    print("RAG PIPELINE SETUP TEST")
    print("="*80)
    print()
    
    all_passed = True
    
    # Run tests
    all_passed = test_api_keys() and all_passed
    all_passed = test_imports() and all_passed
    all_passed = test_icd10_module() and all_passed
    all_passed = test_data_files() and all_passed
    all_passed = test_pipeline_modules() and all_passed
    
    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nYou're ready to:")
        print("  1. Build vector index: python3 icd10_vector_index_builder.py")
        print("  2. Run pipeline: python3 rag_icd10_pipeline.py --limit 10")
        print("  Or use: ./RUN_RAG_PIPELINE.sh")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
    print("="*80)
    print()
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
