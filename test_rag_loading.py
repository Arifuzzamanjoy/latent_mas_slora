#!/usr/bin/env python3
"""Test RAG document loading"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src import LatentMASSystem
import pandas as pd
import json


def load_data_documents(system, data_dir: Path):
    """Load documents from CSV and JSON files"""
    doc_count = 0
    
    # Load CSV files
    csv_files = list(data_dir.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Convert CSV to readable text format
            content = f"# {csv_file.stem}\n\n"
            content += f"Data from {csv_file.name}\n\n"
            content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            
            # Convert rows to text
            for idx, row in df.iterrows():
                content += f"Entry {idx + 1}:\n"
                for col, val in row.items():
                    if pd.notna(val):
                        content += f"  {col}: {val}\n"
                content += "\n"
            
            # Add document to RAG system
            system.rag.store.add_document(
                content=content,
                title=csv_file.stem,
                source=str(csv_file),
                metadata={"type": "csv", "rows": len(df), "columns": list(df.columns)}
            )
            doc_count += 1
            print(f"✓ Loaded CSV: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Failed to load {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Load JSON files
    json_files = list(data_dir.glob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = f"# {json_file.stem}\n\n"
            content += f"Data from {json_file.name}\n\n"
            
            if isinstance(data, list):
                content += f"Total items: {len(data)}\n\n"
                for idx, item in enumerate(data):
                    content += f"Item {idx + 1}:\n"
                    if isinstance(item, dict):
                        for key, val in item.items():
                            content += f"  {key}: {val}\n"
                    else:
                        content += f"  {item}\n"
                    content += "\n"
            
            system.rag.store.add_document(
                content=content,
                title=json_file.stem,
                source=str(json_file),
                metadata={"type": "json", "items": len(data) if isinstance(data, list) else 1}
            )
            doc_count += 1
            print(f"✓ Loaded JSON: {json_file.name}")
        except Exception as e:
            print(f"✗ Failed to load {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Build RAG index
    if doc_count > 0:
        print("\nBuilding RAG index...")
        system.rag.retriever.build_index(force=True)
        system.rag._indexed = True
        total_chunks = len(system.rag.store.get_all_chunks())
        print(f"✓ RAG index built with {total_chunks} chunks")
    
    return doc_count


def main():
    print("Initializing minimal system...")
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        dtype="bfloat16",
    )
    
    print("\nEnabling RAG...")
    system.enable_rag()
    
    print("\nLoading documents...")
    data_dir = Path("data")
    doc_count = load_data_documents(system, data_dir)
    print(f"\n✓ Total documents loaded: {doc_count}")
    
    # Test retrieval
    print("\n" + "="*70)
    print("Testing RAG Retrieval")
    print("="*70)
    
    test_queries = [
        "What is the price of TRON?",
        "Bitcoin market cap",
        "medical questions about asthma"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = system.rag.retrieve(query, top_k=3)
        print(f"Found {len(result.chunks)} chunks:")
        for i, chunk in enumerate(result.chunks[:2], 1):
            print(f"\n  Chunk {i} (score: {result.scores[i-1]:.3f}):")
            print(f"  {chunk.text[:200]}...")


if __name__ == "__main__":
    main()
