#!/usr/bin/env python3
"""
Interactive Chat Interface for LatentMAS

Run with: python chat.py [options]
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import LatentMASSystem
import pandas as pd
import json


def load_data_documents(system, data_dir: Path) -> int:
    """Load documents from CSV and JSON files in data directory"""
    doc_count = 0
    
    # Load CSV files
    csv_files = list(data_dir.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Convert CSV to readable text format
            content = f"# {csv_file.stem}\n\n"
            content += f"Data from {csv_file.name}\n\n"
            
            # Add column information
            content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            
            # Convert rows to text with better formatting for retrieval
            for idx, row in df.iterrows():
                # Add coin name prominently for better search
                if 'Coin Name' in df.columns:
                    coin_name = row.get('Coin Name', '')
                    symbol = row.get('Symbol', '')
                    content += f"=== {coin_name} ({symbol}) ===\n"
                else:
                    content += f"Entry {idx + 1}:\n"
                
                for col, val in row.items():
                    if pd.notna(val):
                        content += f"{col}: {val}\n"
                content += "\n"
            
            # Add document to RAG system
            system.rag.store.add_document(
                content=content,
                title=csv_file.stem,
                source=str(csv_file),
                metadata={"type": "csv", "rows": len(df), "columns": list(df.columns)}
            )
            doc_count += 1
            print(f"  ✓ Loaded CSV: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"  ✗ Failed to load {csv_file.name}: {e}")
    
    # Load JSON files
    json_files = list(data_dir.glob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text format
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
            elif isinstance(data, dict):
                for key, val in data.items():
                    content += f"{key}: {val}\n"
            else:
                content += str(data)
            
            # Add document to RAG system
            system.rag.store.add_document(
                content=content,
                title=json_file.stem,
                source=str(json_file),
                metadata={"type": "json", "items": len(data) if isinstance(data, list) else 1}
            )
            doc_count += 1
            print(f"  ✓ Loaded JSON: {json_file.name} ({len(data) if isinstance(data, list) else 1} items)")
        except Exception as e:
            print(f"  ✗ Failed to load {json_file.name}: {e}")
    
    # Build RAG index after loading all documents
    if doc_count > 0:
        print("  Building RAG index...")
        system.rag.retriever.build_index(force=True)
        system.rag._indexed = True
        print(f"  ✓ RAG index built with {len(system.rag.store.get_all_chunks())} chunks")
    
    return doc_count


def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print("LatentMAS Interactive Chat - Hybrid Domain + Role System")
    print("="*70)
    print("Ultra-fast Multi-Agent Reasoning with Domain Expertise")
    print("\nCommands:")
    print("  /help     - Show this help message")
    print("  /clear    - Clear conversation history")
    print("  /save     - Save conversation to file")
    print("  /exit     - Exit chat")
    print("\nFeatures:")
    print("  ⚡ TRUE LatentMAS - Only final agent generates text (3-5x faster)")
    print("  🧠 4 role agents: Planner→Critic→Refiner→Judger")
    print("  🎯 Domain routing - Auto-selects expert LoRAs (medical/math/code)")
    print("  📚 RAG document intelligence (always on)")
    print("  💬 Conversation continuity with context")
    print("  🔧 Optional tool use (--enable-tools)")
    print("="*70 + "\n")


def initialize_system(args):
    """Initialize the LatentMAS system"""
    print("Initializing system...")
    print(f"Model: {args.model}")
    print(f"Mode: TRUE LatentMAS (latent-space collaboration)")
    print(f"Precision: Full (bfloat16)")
    
    # Initialize system with TRUE LatentMAS settings
    system = LatentMASSystem(
        model_name=args.model,
        device=args.device,
        dtype="bfloat16",
        latent_steps=10,  # Optimized for true_latent mode
        latent_realign=True
    )
    
    # Add agents with TRUE LatentMAS config
    # Only final agent generates text for speed
    print("Adding agents (TRUE LatentMAS mode)...")
    from src.agents.configs import AgentConfig
    system.add_agent(AgentConfig.planner(max_tokens=100))   # Latent only
    system.add_agent(AgentConfig.critic(max_tokens=100))    # Latent only
    system.add_agent(AgentConfig.refiner(max_tokens=100))   # Latent only
    system.add_agent(AgentConfig.judger(max_tokens=800))    # Final output
    
    # Enable conversations
    print("Enabling conversation continuity...")
    system.enable_conversations(
        session_path=args.session_path,
        default_system_prompt=args.system_prompt
    )
    
    # Always enable RAG
    print("Enabling RAG (document intelligence)...")
    system.enable_rag(
        chunk_size=512,
        embedding_model="all-MiniLM-L6-v2",
        top_k=5
    )
    
    # Enable domain routing for intelligent LoRA selection
    print("Enabling domain-based routing...")
    system.enable_domain_routing(
        embedding_model="all-MiniLM-L6-v2",
        auto_load_adapters=False,  # Don't preload to save memory
    )
    
    # Load documents from data directory by default (unless disabled)
    if not getattr(args, 'no_default_data', False):
        data_dir = Path(__file__).parent.parent / "data"
        if data_dir.exists():
            print(f"Loading documents from data directory: {data_dir}")
            doc_count = load_data_documents(system, data_dir)
            print(f"✓ Loaded {doc_count} documents")
    else:
        print("Skipping default data loading (--no-default-data)")
    
    # Load additional documents if provided
    if args.rag_docs:
        print(f"Loading additional documents from: {args.rag_docs}")
        system.load_documents(args.rag_docs)
    
    # Enable tools if requested
    if args.enable_tools:
        print("Enabling tools...")
        system.enable_tools(register_defaults=True)
    
    print("\n✓ System ready!\n")
    return system


def chat_loop(system, use_tools=False):
    """Main chat loop"""
    conversation_id = None
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()
                
                if command == "/exit" or command == "/quit":
                    print("\nGoodbye! 👋")
                    break
                
                elif command == "/help":
                    print_header()
                    continue
                
                elif command == "/clear":
                    # Create new conversation
                    print("\n[Cleared conversation history]\n")
                    conversation_id = None
                    continue
                
                elif command == "/save":
                    # Save conversation
                    if conversation_id:
                        filename = f"conversation_{conversation_id[:8]}.md"
                        system.conversations.export_conversation(
                            conversation_id,
                            filename,
                            format="markdown"
                        )
                        print(f"\n[Saved to {filename}]\n")
                    else:
                        print("\n[No conversation to save]\n")
                    continue
                
                else:
                    print(f"\n[Unknown command: {command}]\n")
                    continue
            
            # Process message
            print("\nAssistant: ", end="", flush=True)
            
            if use_tools:
                # Use tools if enabled
                result = system.run_with_tools(
                    user_input,
                    conversation_id=conversation_id,
                    max_tool_calls=5
                )
                response = result.get('answer', '') if isinstance(result, dict) else str(result)
                
                # Show tool usage if any
                if isinstance(result, dict) and result.get('tool_calls_made', 0) > 0:
                    print(f"[Used {result['tool_calls_made']} tools] ", end="")
            else:
                # Chat with TRUE LatentMAS (latent-space collaboration)
                # Always use conversation system for continuity
                if conversation_id is None:
                    # First message - create new conversation
                    conv = system.new_conversation()
                    conversation_id = conv.conversation_id
                
                # Continue conversation with history
                response = system.chat(
                    user_input,
                    conversation_id=conversation_id
                )
            
            # Store conversation ID for context
            if isinstance(response, dict) and 'conversation_id' in response:
                conversation_id = response['conversation_id']
                response = response.get('response', response.get('answer', ''))
            
            # Display response
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except EOFError:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n[Error: {e}]\n")
            import traceback
            traceback.print_exc()


def run_single_prompt(system, args):
    """Run a single prompt and exit (non-interactive mode)"""
    prompt = args.prompt
    
    # Create conversation
    conv = system.new_conversation()
    conversation_id = conv.conversation_id
    
    print(f"\n[Prompt]: {prompt}\n")
    print("[Response]:", end=" ", flush=True)
    
    if args.enable_tools:
        result = system.run_with_tools(
            prompt,
            conversation_id=conversation_id,
            max_tool_calls=5
        )
        response = result.get('answer', '') if isinstance(result, dict) else str(result)
    else:
        response = system.chat(prompt, conversation_id=conversation_id)
    
    print(response)
    
    # Output JSON if requested
    if args.output_json:
        import json
        output = {
            "prompt": prompt,
            "response": response,
            "conversation_id": conversation_id,
            "model": args.model,
        }
        print(f"\n[JSON Output]:\n{json.dumps(output, indent=2)}")
    
    return response


def load_external_rag_data(system, rag_data_url: str):
    """Load RAG data from URL or base64"""
    import requests
    import base64
    from io import StringIO
    
    try:
        if rag_data_url.startswith(("http://", "https://")):
            print(f"Downloading RAG data from: {rag_data_url[:50]}...")
            response = requests.get(rag_data_url, timeout=30)
            response.raise_for_status()
            content = response.text
        else:
            print("Decoding base64 RAG data...")
            content = base64.b64decode(rag_data_url).decode('utf-8')
        
        # Try JSON first
        try:
            data = json.loads(content)
            doc_content = json.dumps(data, indent=2)
            system.rag.store.add_document(
                content=doc_content,
                title="external_data",
                source="external_url",
                metadata={"type": "json"}
            )
            print("  ✓ Loaded external JSON data")
        except json.JSONDecodeError:
            # Load as text
            system.rag.store.add_document(
                content=content,
                title="external_data",
                source="external_url",
                metadata={"type": "text"}
            )
            print("  ✓ Loaded external text data")
        
        system.rag.retriever.build_index(force=True)
        system.rag._indexed = True
        
    except Exception as e:
        print(f"  ✗ Failed to load external RAG data: {e}")


def load_rag_documents_json(system, rag_docs_json: str):
    """Load RAG documents from JSON array"""
    try:
        docs = json.loads(rag_docs_json)
        if isinstance(docs, list):
            for i, doc in enumerate(docs):
                if isinstance(doc, dict):
                    content = doc.get("content", json.dumps(doc))
                    title = doc.get("title", f"doc_{i+1}")
                else:
                    content = str(doc)
                    title = f"doc_{i+1}"
                
                system.rag.store.add_document(
                    content=content,
                    title=title,
                    source="json_input",
                    metadata={"type": "json_array"}
                )
            print(f"  ✓ Loaded {len(docs)} documents from JSON")
        
        system.rag.retriever.build_index(force=True)
        system.rag._indexed = True
        
    except Exception as e:
        print(f"  ✗ Failed to load RAG documents JSON: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Interactive chat with LatentMAS system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python chat.py --model Qwen/Qwen2.5-3B-Instruct
  
  # Single prompt mode (non-interactive)
  python chat.py --prompt "What is the treatment for hypertension?"
  
  # With external RAG data from URL
  python chat.py --prompt "Summarize the data" --rag-data-url "https://example.com/data.json"
  
  # With RAG documents as JSON array
  python chat.py --prompt "What does doc1 say?" --rag-docs-json '[{"title":"doc1","content":"..."}]'
  
  # With custom system prompt and tools
  python chat.py --system-prompt "You are a medical expert" --enable-tools
"""
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    # Inference parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum tokens to generate (default: 800)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    
    # Feature arguments
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant with access to multiple specialized agents for reasoning and problem-solving.",
        help="System prompt for the conversation"
    )
    parser.add_argument(
        "--session-path",
        type=str,
        default="./chat_sessions",
        help="Path to store conversation sessions (default: ./chat_sessions)"
    )
    parser.add_argument(
        "--enable-tools",
        action="store_true",
        help="Enable tool use (calculator, search, etc.)"
    )
    
    # RAG arguments
    parser.add_argument(
        "--rag-docs",
        type=str,
        help="Path to documents directory or file for RAG"
    )
    parser.add_argument(
        "--rag-data-url",
        type=str,
        help="URL to download RAG data from (JSON/CSV/text)"
    )
    parser.add_argument(
        "--rag-docs-json",
        type=str,
        help="JSON array of documents: '[{\"title\":\"...\",\"content\":\"...\"}]'"
    )
    parser.add_argument(
        "--no-default-data",
        action="store_true",
        help="Don't load default data from ./data directory"
    )
    
    # Single prompt mode (non-interactive)
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to run (non-interactive mode)"
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output response as JSON (for --prompt mode)"
    )
    parser.add_argument(
        "--conversation-id",
        type=str,
        help="Continue an existing conversation by ID"
    )
    
    # Server mode
    parser.add_argument(
        "--serve-api",
        action="store_true",
        help="Start as local API server (for testing)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Print header (unless in single prompt mode with JSON output)
    if not (args.prompt and args.output_json):
        print_header()
    
    # Initialize system
    system = initialize_system(args)
    
    # Load external RAG data if provided
    if args.rag_data_url:
        load_external_rag_data(system, args.rag_data_url)
    
    if args.rag_docs_json:
        load_rag_documents_json(system, args.rag_docs_json)
    
    # Run mode
    if args.serve_api:
        # Start local API server
        print(f"\nStarting local API server on port {args.port}...")
        print(f"Test with: curl -X POST http://localhost:{args.port}/chat -d '{{\"prompt\": \"Hello\"}}'\n")
        # This would need flask/fastapi - for now just print instructions
        print("[API server mode not implemented yet - use --prompt for single queries]")
    elif args.prompt:
        # Single prompt mode
        run_single_prompt(system, args)
    else:
        # Interactive chat loop
        chat_loop(system, use_tools=args.enable_tools)


if __name__ == "__main__":
    main()
