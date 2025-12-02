#!/usr/bin/env python3
"""
ICD-10 Vector Index Builder
Builds a vector store index from the ICD-10 knowledge graph
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Import ICD-10 knowledge graph
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import icd10


def create_embedding_text(code_data):
    """
    Create rich text representation of ICD-10 code for embedding
    Includes: code, description, parent hierarchy, children
    """
    code = code_data['code']
    description = code_data['description']
    code_type = code_data.get('type', 'diagnosis')
    parent_chain = code_data.get('parentChain', [])
    children = code_data.get('children', [])
    
    # Build parent hierarchy text
    parent_text = ""
    if parent_chain:
        parent_names = []
        for p in parent_chain:
            # Handle both dict and string formats
            if isinstance(p, dict):
                p_code = p.get('code', '')
                p_desc = p.get('description', p.get('name', ''))
                parent_names.append(f"{p_code} - {p_desc}")
            else:
                parent_names.append(str(p))
        parent_text = " | Parent Hierarchy: " + " → ".join(parent_names)
    
    # Build children text (limit to first 5 for brevity)
    children_text = ""
    if children:
        # Children are returned as a list of strings (codes)
        if isinstance(children, list) and children:
            # Take first 5 children codes
            child_codes = children[:5]
            # Try to get descriptions for children
            child_with_desc = []
            for child_code in child_codes:
                try:
                    child_data = icd10.get_full_data(child_code)
                    if child_data:
                        child_desc = child_data.get('description', '')
                        child_with_desc.append(f"{child_code} - {child_desc[:30]}")
                    else:
                        child_with_desc.append(child_code)
                except:
                    child_with_desc.append(child_code)
            
            children_text = " | Children: " + ", ".join(child_with_desc)
            if len(children) > 5:
                children_text += f" (and {len(children) - 5} more)"
    
    # Combine all information
    embedding_text = (
        f"ICD-10 Code: {code} | "
        f"Description: {description} | "
        f"Type: {code_type}"
        f"{parent_text}"
        f"{children_text}"
    )
    
    return embedding_text


def build_icd10_vector_index(output_dir="icd10_vector_index"):
    """
    Build vector index from ICD-10 knowledge graph
    """
    print("="*80)
    print("ICD-10 VECTOR INDEX BUILDER")
    print("="*80)
    print()
    
    # Setup OpenAI embeddings
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    embed_model = OpenAIEmbedding(
        api_key=openai_api_key,
        model="text-embedding-3-small"  # Fast and cost-effective
    )
    
    # Configure settings
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    print("1️⃣  Loading ICD-10 knowledge graph...")
    
    # Use the code_to_node dictionary which contains all loaded codes
    # This is much faster than traversing the tree
    all_codes_list = list(icd10.code_to_node.keys())
    
    print(f"   Found {len(all_codes_list)} codes in ICD-10")
    print("   Loading full data for all codes...")
    
    all_codes_data = []
    failed_codes = []
    
    for code in tqdm(all_codes_list, desc="Loading codes"):
        try:
            code_data = icd10.get_full_data(code)
            if code_data:
                all_codes_data.append(code_data)
            else:
                failed_codes.append(code)
        except Exception as e:
            failed_codes.append(code)
            # Uncomment for debugging:
            # print(f"Error loading {code}: {str(e)}")
    
    all_codes = all_codes_data
    
    if failed_codes:
        print(f"   ⚠️  {len(failed_codes)} codes failed to load")
    
    print(f"   ✅ Successfully loaded {len(all_codes)} unique ICD-10 codes")
    print()
    
    # Create documents for embedding
    print("2️⃣  Creating embedding documents...")
    
    documents = []
    metadata_map = {}
    
    for code_data in tqdm(all_codes, desc="Creating documents"):
        code = code_data['code']
        
        # Create rich embedding text
        embedding_text = create_embedding_text(code_data)
        
        # Create document
        doc = Document(
            text=embedding_text,
            metadata={
                'code': code,
                'description': code_data['description'],
                'type': code_data.get('type', 'diagnosis')
            }
        )
        
        documents.append(doc)
        
        # Store full data for later retrieval
        metadata_map[code] = code_data
    
    print(f"   ✅ Created {len(documents)} documents")
    print()
    
    # Build vector index
    print("3️⃣  Building vector index (this may take a while)...")
    print("   Embedding documents with OpenAI...")
    
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    print("   ✅ Vector index built successfully")
    print()
    
    # Save index to disk
    print("4️⃣  Saving index to disk...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the index
    index.storage_context.persist(persist_dir=str(output_path))
    
    # Save metadata map
    metadata_file = output_path / "code_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata_map, f, indent=2)
    
    print(f"   ✅ Index saved to: {output_path}")
    print(f"   ✅ Metadata saved to: {metadata_file}")
    print()
    
    # Print statistics
    print("="*80)
    print("INDEX STATISTICS")
    print("="*80)
    print(f"Total codes indexed: {len(all_codes)}")
    print(f"Total documents: {len(documents)}")
    print(f"Embedding model: text-embedding-3-small")
    print(f"Output directory: {output_path}")
    print("="*80)
    print()
    
    return index


def test_index(index_dir="icd10_vector_index"):
    """
    Test the built index with a sample query
    """
    print("\n" + "="*80)
    print("TESTING INDEX")
    print("="*80)
    print()
    
    from llama_index.core import load_index_from_storage, StorageContext
    
    # Load index
    print("Loading index...")
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)
    
    # Load metadata
    metadata_file = Path(index_dir) / "code_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata_map = json.load(f)
    
    print("✅ Index loaded")
    print()
    
    # Test query
    test_queries = [
        "atrial fibrillation",
        "diabetes mellitus type 2",
        "myocardial infarction"
    ]
    
    query_engine = index.as_query_engine(similarity_top_k=5)
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 80)
        
        response = query_engine.query(query)
        
        print("Top 5 results:")
        for i, node in enumerate(response.source_nodes, 1):
            code = node.metadata['code']
            desc = node.metadata['description']
            score = node.score
            print(f"  {i}. {code} - {desc}")
            print(f"     Similarity: {score:.4f}")
        print()
    
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build ICD-10 vector index')
    parser.add_argument('--output-dir', default='icd10_vector_index',
                       help='Output directory for index')
    parser.add_argument('--test', action='store_true',
                       help='Test the index after building')
    
    args = parser.parse_args()
    
    # Build index
    index = build_icd10_vector_index(args.output_dir)
    
    # Test if requested
    if args.test:
        test_index(args.output_dir)
    
    print("✅ Done!")
