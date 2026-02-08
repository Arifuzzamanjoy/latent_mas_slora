#!/usr/bin/env python3
"""
LatentMAS RunPod Serverless Handler

Multi-agent reasoning with:
- Domain-specific LoRA routing (medical/math/code)
- RAG document intelligence
- Conversation continuity
- TRUE LatentMAS latent-space collaboration

API Input Schema:
{
    "input": {
        "prompt": "Your question here",
        "rag_data": "Optional: URL or base64 JSON/CSV data for RAG",
        "rag_documents": ["doc1.txt content", "doc2.txt content"],
        "system_prompt": "Optional custom system prompt",
        "conversation_id": "Optional: continue existing conversation",
        "max_tokens": 800,
        "temperature": 0.7,
        "enable_tools": false,
        "model": "Qwen/Qwen2.5-3B-Instruct"
    }
}
"""

import runpod
import torch
import os
import sys
import json
import base64
import tempfile
from io import BytesIO, StringIO
from pathlib import Path

# Add src to path
sys.path.insert(0, '/app')

# Set cache directories
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/home/caches/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "/home/caches/huggingface")
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/home/caches/huggingface/hub")
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "/home/caches/torch")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "/home/caches/sentence_transformers")

# Create cache directories
for cache_dir in ["/home/caches/huggingface/hub", "/home/caches/torch", "/home/caches/sentence_transformers"]:
    os.makedirs(cache_dir, exist_ok=True)

from src import LatentMASSystem
from src.agents.configs import AgentConfig

# Global system instance (loaded once on cold start)
SYSTEM = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] LatentMAS Serverless Worker starting...")
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def load_system(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load and configure the LatentMAS system"""
    global SYSTEM
    
    if SYSTEM is not None and SYSTEM.config.model_name == model_name:
        print("[INFO] Reusing existing system instance")
        return SYSTEM
    
    print(f"[INFO] Loading LatentMAS system with model: {model_name}")
    
    SYSTEM = LatentMASSystem(
        model_name=model_name,
        device=DEVICE,
        dtype="bfloat16",
        latent_steps=10,
        latent_realign=True,
    )
    
    # Add TRUE LatentMAS agents (only final agent generates text)
    print("[INFO] Adding agents (TRUE LatentMAS mode)...")
    SYSTEM.add_agent(AgentConfig.planner(max_tokens=100))
    SYSTEM.add_agent(AgentConfig.critic(max_tokens=100))
    SYSTEM.add_agent(AgentConfig.refiner(max_tokens=100))
    SYSTEM.add_agent(AgentConfig.judger(max_tokens=800))
    
    # Enable RAG
    print("[INFO] Enabling RAG (document intelligence)...")
    SYSTEM.enable_rag(
        chunk_size=512,
        embedding_model="all-MiniLM-L6-v2",
        top_k=5,
    )
    
    # Enable domain routing
    print("[INFO] Enabling domain-based routing...")
    SYSTEM.enable_domain_routing(
        embedding_model="all-MiniLM-L6-v2",
        auto_load_adapters=False,
    )
    
    # Enable conversations
    print("[INFO] Enabling conversation continuity...")
    SYSTEM.enable_conversations(
        session_path="/tmp/chat_sessions",
        default_system_prompt="You are a helpful AI assistant with multi-agent reasoning capabilities.",
    )
    
    # Load default data documents
    data_dir = Path("/app/data")
    if data_dir.exists():
        print(f"[INFO] Loading default documents from {data_dir}")
        load_data_documents(SYSTEM, data_dir)
    
    print("[INFO] System loaded successfully!")
    return SYSTEM


def load_data_documents(system, data_dir: Path) -> int:
    """Load documents from CSV and JSON files"""
    import pandas as pd
    
    doc_count = 0
    
    # Load CSV files
    for csv_file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            content = f"# {csv_file.stem}\n\n"
            content += f"Data from {csv_file.name}\n\n"
            content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            
            for idx, row in df.iterrows():
                for col, val in row.items():
                    if pd.notna(val):
                        content += f"{col}: {val}\n"
                content += "\n"
            
            system.rag.store.add_document(
                content=content,
                title=csv_file.stem,
                source=str(csv_file),
                metadata={"type": "csv", "rows": len(df)}
            )
            doc_count += 1
            print(f"  ✓ Loaded CSV: {csv_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {csv_file.name}: {e}")
    
    # Load JSON files
    for json_file in data_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            content = f"# {json_file.stem}\n\n"
            content += json.dumps(data, indent=2)
            
            system.rag.store.add_document(
                content=content,
                title=json_file.stem,
                source=str(json_file),
                metadata={"type": "json"}
            )
            doc_count += 1
            print(f"  ✓ Loaded JSON: {json_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {json_file.name}: {e}")
    
    if doc_count > 0:
        system.rag.retriever.build_index(force=True)
        system.rag._indexed = True
    
    return doc_count


def load_external_rag_data(system, rag_data: str) -> int:
    """Load RAG data from URL or base64 encoded content"""
    import pandas as pd
    import requests
    
    doc_count = 0
    
    try:
        # Check if URL
        if rag_data.startswith(("http://", "https://")):
            print(f"[INFO] Downloading RAG data from URL: {rag_data[:50]}...")
            response = requests.get(rag_data, timeout=30)
            response.raise_for_status()
            content = response.text
        else:
            # Assume base64 encoded
            print("[INFO] Decoding base64 RAG data...")
            content = base64.b64decode(rag_data).decode('utf-8')
        
        # Try to parse as JSON
        try:
            data = json.loads(content)
            doc_content = json.dumps(data, indent=2)
            system.rag.store.add_document(
                content=doc_content,
                title="external_data",
                source="external",
                metadata={"type": "json", "source": "api_input"}
            )
            doc_count += 1
            print("[INFO] Loaded external JSON data")
        except json.JSONDecodeError:
            # Try as CSV
            try:
                df = pd.read_csv(StringIO(content))
                doc_content = df.to_string()
                system.rag.store.add_document(
                    content=doc_content,
                    title="external_csv",
                    source="external",
                    metadata={"type": "csv", "source": "api_input"}
                )
                doc_count += 1
                print("[INFO] Loaded external CSV data")
            except Exception:
                # Load as plain text
                system.rag.store.add_document(
                    content=content,
                    title="external_text",
                    source="external",
                    metadata={"type": "text", "source": "api_input"}
                )
                doc_count += 1
                print("[INFO] Loaded external text data")
        
        if doc_count > 0:
            system.rag.retriever.build_index(force=True)
            system.rag._indexed = True
            
    except Exception as e:
        print(f"[ERROR] Failed to load external RAG data: {e}")
    
    return doc_count


def handler(job):
    """
    RunPod Serverless Handler for LatentMAS
    
    Input Schema:
    {
        "input": {
            "prompt": "Your question here (required)",
            "rag_data": "URL or base64 JSON/CSV data (optional)",
            "rag_documents": ["doc1 content", "doc2 content"] (optional),
            "system_prompt": "Custom system prompt (optional)",
            "conversation_id": "Continue conversation (optional)",
            "max_tokens": 800 (optional),
            "temperature": 0.7 (optional),
            "enable_tools": false (optional),
            "model": "Qwen/Qwen2.5-3B-Instruct" (optional)
        }
    }
    
    Output Schema:
    {
        "response": "AI response text",
        "conversation_id": "conversation_uuid",
        "domain": "detected domain (medical/math/code/general)",
        "domain_confidence": 0.85,
        "tokens_used": 150,
        "rag_chunks_used": 3
    }
    """
    job_input = job.get("input", {})
    
    # Required parameters
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required parameter: 'prompt'"}
    
    # Optional parameters
    model_name = job_input.get("model", "Qwen/Qwen2.5-3B-Instruct")
    max_tokens = job_input.get("max_tokens", 800)
    temperature = job_input.get("temperature", 0.7)
    system_prompt = job_input.get("system_prompt")
    conversation_id = job_input.get("conversation_id")
    enable_tools = job_input.get("enable_tools", False)
    rag_data = job_input.get("rag_data")
    rag_documents = job_input.get("rag_documents", [])
    
    try:
        # Load or reuse system
        system = load_system(model_name)
        
        # Load external RAG data if provided
        if rag_data:
            load_external_rag_data(system, rag_data)
        
        # Load document list if provided
        if rag_documents:
            for i, doc in enumerate(rag_documents):
                system.rag.store.add_document(
                    content=doc,
                    title=f"input_doc_{i+1}",
                    source="api_input",
                    metadata={"type": "text", "source": "api_input"}
                )
            if rag_documents:
                system.rag.retriever.build_index(force=True)
                system.rag._indexed = True
                print(f"[INFO] Loaded {len(rag_documents)} external documents")
        
        # Update system prompt if provided
        if system_prompt and system._conversation_manager:
            system._conversation_manager.default_system_prompt = system_prompt
        
        # Enable tools if requested
        if enable_tools and system._tool_registry is None:
            system.enable_tools(register_defaults=True)
        
        # Process the request
        print(f"[INFO] Processing prompt: '{prompt[:100]}...'")
        
        # Get or create conversation
        if conversation_id:
            conv = system.get_conversation(conversation_id)
            if conv is None:
                conv = system.new_conversation()
                conversation_id = conv.conversation_id
        else:
            conv = system.new_conversation()
            conversation_id = conv.conversation_id
        
        # Run inference
        if enable_tools:
            result = system.run_with_tools(
                prompt,
                conversation_id=conversation_id,
                max_tool_calls=5,
            )
            response_text = result.get('answer', '') if isinstance(result, dict) else str(result)
        else:
            response_text = system.chat(
                prompt,
                conversation_id=conversation_id,
            )
        
        # Get domain routing info
        domain_info = {"domain": "general", "confidence": 0.0}
        if system._domain_routing_enabled:
            try:
                if system._use_fast_router and system._fast_router:
                    domain, confidence = system._fast_router.route(prompt)
                elif system._use_advanced_router and system._advanced_router:
                    result = system._advanced_router.route(prompt)
                    domain, confidence = result.domain, result.confidence
                elif system._semantic_router:
                    domain, confidence = system._semantic_router.get_best_domain(prompt)
                else:
                    domain, confidence = "general", 0.0
                domain_info = {"domain": domain.value if hasattr(domain, 'value') else str(domain), "confidence": confidence}
            except Exception:
                pass
        
        # Build response
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "domain": domain_info.get("domain", "general"),
            "domain_confidence": domain_info.get("confidence", 0.0),
            "model": model_name,
            "rag_enabled": system._rag_pipeline is not None,
            "tools_enabled": enable_tools,
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Handler error: {e}")
        print(error_trace)
        return {
            "error": str(e),
            "traceback": error_trace,
        }


# Load system on cold start (outside handler for efficiency)
print("[INFO] Pre-loading system on cold start...")
try:
    load_system()
except Exception as e:
    print(f"[WARN] Could not pre-load system: {e}")
    print("[INFO] System will be loaded on first request")

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
