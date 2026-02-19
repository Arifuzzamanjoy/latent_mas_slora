#!/usr/bin/env python3
"""
LatentMAS RunPod Serverless Handler

Multi-agent reasoning with:
- Domain-specific LoRA routing (medical/math/code)
- RAG document intelligence
- Conversation continuity with persistent session memory
- TRUE LatentMAS latent-space collaboration
- External LoRA adapter selection via API
- Updatable LoRA registry (data/lora_registry.json)

API Input Schema:
{
    "input": {
        "prompt": "Your question here",
        "image_url": "Optional: URL of an image for VLM analysis",
        "image_base64": "Optional: base64-encoded image data",
        "rag_data": "Optional: URL or base64 JSON/CSV data for RAG",
        "rag_documents": ["doc1.txt content", "doc2.txt content"],
        "system_prompt": "Optional custom system prompt",
        "conversation_id": "Optional: continue existing conversation",
        "session_id": "Optional: group conversations into a session",
        "max_tokens": 800,
        "temperature": 0.7,
        "enable_tools": false,
        "lora_adapter": "Optional: name from registry (e.g. 'medical_reasoner')",
        "lora_hf_path": "Optional: direct HuggingFace LoRA path",
        "no_default_data": false,
        "list_loras": false,
        "list_conversations": false,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct"
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

from src.system_legacy import LatentMASSystem
from src.agents.configs import AgentConfig
from src.conversation.session import SessionStore

# Global system instance (loaded once on cold start)
SYSTEM = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Persistent session store (survives across requests within same worker)
SESSION_STORE = None

# External LoRA registry (loaded from JSON file, can be updated)
LORA_REGISTRY = {}
LORA_REGISTRY_PATH = Path("/app/data/lora_registry.json")

# Track default data state
_DEFAULT_DATA_LOADED = False

print(f"[INFO] LatentMAS Serverless Worker starting...")
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ================================================================
# LoRA Registry Management
# ================================================================

def load_lora_registry() -> dict:
    """Load the external LoRA registry from JSON file."""
    global LORA_REGISTRY
    registry_path = LORA_REGISTRY_PATH

    if not registry_path.exists():
        # Also check local dev path
        alt = Path(__file__).parent / "data" / "lora_registry.json"
        if alt.exists():
            registry_path = alt

    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                data = json.load(f)
            LORA_REGISTRY = data.get("adapters", {})
            print(f"[INFO] Loaded LoRA registry: {len(LORA_REGISTRY)} adapters from {registry_path}")
        except Exception as e:
            print(f"[WARN] Failed to load LoRA registry: {e}")
    else:
        print("[INFO] No external lora_registry.json found, using built-in registry only")

    return LORA_REGISTRY


def get_available_loras() -> dict:
    """Get combined LoRA info from external registry + built-in registry."""
    from src.lora.adapter_manager import QWEN25_LORA_REGISTRY

    combined = {}
    # Built-in entries
    for name, info in QWEN25_LORA_REGISTRY.items():
        combined[name] = {
            "hf_path": info.hf_path,
            "description": info.description,
            "domain": info.domain,
            "base_model": info.base_model,
            "source": "builtin",
        }
    # External JSON entries (override built-in if same name)
    for name, info in LORA_REGISTRY.items():
        combined[name] = {
            "hf_path": info.get("hf_path", ""),
            "description": info.get("description", ""),
            "domain": info.get("domain", "general"),
            "base_model": info.get("base_model", ""),
            "tags": info.get("tags", []),
            "source": "registry_json",
        }
    return combined


# ================================================================
# Session-persistent Conversation Memory
# ================================================================

def get_session_store() -> SessionStore:
    """Get or create the persistent session store."""
    global SESSION_STORE
    if SESSION_STORE is None:
        session_path = os.environ.get("SESSION_PATH", "/tmp/chat_sessions")
        SESSION_STORE = SessionStore(
            storage_path=session_path,
            auto_save=True,
            max_sessions=500,
        )
        print(f"[INFO] Session store initialized at {session_path}")
    return SESSION_STORE


# ================================================================
# System Loading
# ================================================================


def load_system(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
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
    
    # Enable conversations with persistent session store
    print("[INFO] Enabling conversation continuity...")
    SYSTEM.enable_conversations(
        session_path="/tmp/chat_sessions",
        default_system_prompt="You are a helpful AI assistant with multi-agent reasoning capabilities.",
    )
    # Wire the session store so conversations persist across requests
    SYSTEM._session_store = get_session_store()
    
    # Load default data documents
    global _DEFAULT_DATA_LOADED
    data_dir = Path("/app/data")
    if data_dir.exists():
        print(f"[INFO] Loading default documents from {data_dir}")
        count = load_data_documents(SYSTEM, data_dir)
        if count > 0:
            _DEFAULT_DATA_LOADED = True
    
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


def _cleanup_external_rag(system) -> None:
    """Remove external (per-request) RAG documents, keeping default data."""
    if system._rag_pipeline is None:
        return
    
    store = system.rag.store
    # Find doc_ids that came from api_input / external sources
    docs_to_remove = []
    for doc_id, doc in store._documents.items():
        if (doc.metadata.get("source") == "api_input"
                or doc.source in ("external", "api_input")):
            docs_to_remove.append(doc_id)
    
    if not docs_to_remove:
        return
    
    # Remove documents and their chunks
    for doc_id in docs_to_remove:
        doc = store._documents.pop(doc_id, None)
        if doc:
            for chunk in doc.chunks:
                store._chunks.pop(chunk.chunk_id, None)
    
    # Rebuild index with remaining docs
    if store._documents:
        system.rag.retriever.build_index(force=True)
        system.rag._indexed = True
    else:
        system.rag._indexed = False
    
    print(f"[INFO] Cleaned up {len(docs_to_remove)} external RAG document(s)")


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


# ================================================================
# LoRA Adapter Loading
# ================================================================

def load_lora_for_request(system, adapter_name: str = None, hf_path: str = None) -> dict:
    """
    Load a LoRA adapter for the current request.

    Args:
        adapter_name: Name from the registry (lora_registry.json or built-in)
        hf_path: Direct HuggingFace path (overrides registry lookup)

    Returns:
        dict with status info
    """
    if not adapter_name and not hf_path:
        return {"loaded": False, "reason": "no adapter specified"}

    # Resolve hf_path from registry if only name given
    if adapter_name and not hf_path:
        available = get_available_loras()
        if adapter_name not in available:
            return {
                "loaded": False,
                "error": f"Unknown adapter '{adapter_name}'",
                "available": list(available.keys()),
            }
        hf_path = available[adapter_name]["hf_path"]

    resolved_name = adapter_name or hf_path.split("/")[-1]

    # Check if already loaded
    if system._adapter_manager and system._adapter_manager.is_loaded(resolved_name):
        try:
            system.model.set_adapter(resolved_name)
            print(f"[INFO] Switched to already-loaded adapter: {resolved_name}")
            return {"loaded": True, "adapter": resolved_name, "cached": True}
        except Exception:
            pass

    # Load from HuggingFace
    try:
        success = system.load_external_lora(resolved_name, hf_path)
        if success:
            system.model.set_adapter(resolved_name)
            print(f"[INFO] Loaded & activated LoRA: {resolved_name} ({hf_path})")
            return {"loaded": True, "adapter": resolved_name, "hf_path": hf_path}
        else:
            return {"loaded": False, "error": f"Failed to load {hf_path}"}
    except Exception as e:
        print(f"[WARN] LoRA load failed for {hf_path}: {e}")
        return {"loaded": False, "error": str(e)}


# ================================================================
# Session-aware Conversation Helpers
# ================================================================

def get_or_create_conversation(system, conversation_id=None, session_id=None):
    """
    Get or create a conversation with session persistence.

    If conversation_id is provided, tries to restore it from:
      1. In-memory ConversationManager
      2. Persistent SessionStore (disk)
    If session_id is provided, conversations are grouped under that session.
    
    If both conversation_id and session_id are None, creates a new conversation.
    If only session_id is provided (no conversation_id), uses the active conversation
    from that session or creates a new one.

    Returns:
        (conversation, conversation_id, session_id)
    """
    store = get_session_store()

    # Try to find existing conversation by conversation_id
    if conversation_id:
        # Check in-memory first
        conv = system.get_conversation(conversation_id)
        if conv:
            return conv, conversation_id, session_id

        # Check persistent session store
        if session_id:
            session = store.get_session(session_id)
            if session:
                stored_conv = session.get_conversation(conversation_id)
                if stored_conv:
                    # Restore into ConversationManager
                    system._conversation_manager._conversations[conversation_id] = stored_conv
                    system._conversation_manager._active_conversation_id = conversation_id
                    print(f"[INFO] Restored conversation {conversation_id[:8]} from session {session_id[:8]}")
                    return stored_conv, conversation_id, session_id

        # Search all sessions for this conversation
        for sess in store.list_sessions():
            stored_conv = sess.get_conversation(conversation_id)
            if stored_conv:
                system._conversation_manager._conversations[conversation_id] = stored_conv
                system._conversation_manager._active_conversation_id = conversation_id
                session_id = sess.session_id
                print(f"[INFO] Restored conversation {conversation_id[:8]} from session {session_id[:8]}")
                return stored_conv, conversation_id, session_id

    # If session_id provided but no conversation_id, check for active conversation in session
    if session_id and not conversation_id:
        session = store.get_session(session_id)
        if session:
            active_conv = session.get_active_conversation()
            if active_conv:
                # Restore into ConversationManager
                conversation_id = active_conv.conversation_id
                system._conversation_manager._conversations[conversation_id] = active_conv
                system._conversation_manager._active_conversation_id = conversation_id
                print(f"[INFO] Restored active conversation {conversation_id[:8]} from session {session_id[:8]}")
                return active_conv, conversation_id, session_id

    # Create new conversation
    conv = system.new_conversation()
    conversation_id = conv.conversation_id

    # Create or get session
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())

    session = store.get_or_create_session(session_id)
    session.add_conversation(conv)
    store.save_session(session)
    
    print(f"[INFO] Created new conversation {conversation_id[:8]} in session {session_id[:8]}")

    return conv, conversation_id, session_id


def save_conversation_to_session(system, conversation_id, session_id):
    """Persist the current conversation state to disk."""
    store = get_session_store()
    conv = system.get_conversation(conversation_id)
    if not conv:
        return

    session = store.get_or_create_session(session_id)
    session.conversations[conversation_id] = conv
    store.save_session(session)


# ================================================================
# RunPod Serverless Handler
# ================================================================

def handler(job):
    """
    RunPod Serverless Handler for LatentMAS.

    Handles chat inference, LoRA selection, RAG injection,
    and session-persistent conversations.
    """
    job_input = job.get("input", {})

    # --- Metadata-only requests (no prompt required) ---
    if job_input.get("list_loras"):
        return {
            "loras": get_available_loras(),
            "registry_path": str(LORA_REGISTRY_PATH),
        }

    if job_input.get("list_conversations"):
        store = get_session_store()
        sessions = {}
        for sess in store.list_sessions():
            sessions[sess.session_id] = {
                "created": str(getattr(sess, 'created_at', '')),
                "conversations": list(sess.conversations.keys()),
            }
        return {"sessions": sessions}

    # --- Required parameter ---
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required parameter: 'prompt'"}

    # --- Optional parameters ---
    model_name = job_input.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
    max_tokens = job_input.get("max_tokens", 800)
    temperature = job_input.get("temperature", 0.7)
    system_prompt = job_input.get("system_prompt")
    conversation_id = job_input.get("conversation_id")
    session_id = job_input.get("session_id")
    enable_tools = job_input.get("enable_tools", False)
    rag_data = job_input.get("rag_data")
    rag_documents = job_input.get("rag_documents", [])
    no_default_data = job_input.get("no_default_data", True)
    lora_adapter = job_input.get("lora_adapter")
    lora_hf_path = job_input.get("lora_hf_path")
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")

    try:
        # Load or reuse system
        system = load_system(model_name)

        # Clean up external docs from previous requests
        _cleanup_external_rag(system)

        # --- LoRA adapter selection ---
        lora_info = {"loaded": False}
        if lora_adapter or lora_hf_path:
            lora_info = load_lora_for_request(system, lora_adapter, lora_hf_path)
            if lora_info.get("error"):
                print(f"[WARN] LoRA: {lora_info['error']}")

        # --- RAG handling ---
        # If no_default_data and no external RAG provided, skip RAG retrieval
        skip_rag = no_default_data and not rag_data and not rag_documents

        if rag_data:
            load_external_rag_data(system, rag_data)

        if rag_documents:
            for i, doc in enumerate(rag_documents):
                # Extract content and metadata from document dict
                doc_content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
                doc_metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                doc_metadata["source"] = doc_metadata.get("source", "api_input")
                doc_metadata["index"] = i + 1
                
                system.rag.store.add_document(
                    content=doc_content,
                    title=f"input_doc_{i+1}",
                    source=doc_metadata.get("source", "api_input"),
                    metadata=doc_metadata
                )
            system.rag.retriever.build_index(force=True)
            system.rag._indexed = True
            print(f"[INFO] Loaded {len(rag_documents)} external documents from client")

        # --- System prompt override ---
        if system_prompt and system._conversation_manager:
            system._conversation_manager.default_system_prompt = system_prompt

        # --- Tools ---
        if enable_tools and system._tool_registry is None:
            system.enable_tools(register_defaults=True)

        # --- Conversation with session persistence ---
        print(f"[INFO] Processing: '{prompt[:100]}...'")

        conv, conversation_id, session_id = get_or_create_conversation(
            system,
            conversation_id=conversation_id,
            session_id=session_id,
        )

        # --- Inference ---
        _saved_rag = None
        if skip_rag and system._rag_pipeline is not None:
            _saved_rag = system._rag_pipeline
            system._rag_pipeline = None

        try:
            # VLM image inference path (bypasses multi-agent pipeline)
            if (image_url or image_base64) and system._is_vlm:
                print(f"[INFO] VLM image inference mode")
                response_text = system.vlm_inference(
                    prompt=prompt,
                    image_url=image_url,
                    image_base64=image_base64,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    conversation_id=conversation_id,  # Pass conversation ID for context
                )
                # Save to conversation for continuity
                from src.conversation.manager import MessageRole
                conv.add_message(MessageRole.USER, prompt)
                conv.add_message(MessageRole.ASSISTANT, response_text)
            elif enable_tools:
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
        finally:
            if _saved_rag is not None:
                system._rag_pipeline = _saved_rag

        # --- Persist conversation ---
        save_conversation_to_session(system, conversation_id, session_id)

        # --- Domain routing info ---
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
                domain_info = {
                    "domain": domain.value if hasattr(domain, 'value') else str(domain),
                    "confidence": confidence,
                }
            except Exception:
                pass

        # --- Build response ---
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "domain": domain_info.get("domain", "general"),
            "domain_confidence": domain_info.get("confidence", 0.0),
            "model": model_name,
            "vlm": getattr(system, '_is_vlm', False),
            "image_provided": bool(image_url or image_base64),
            "rag_enabled": system._rag_pipeline is not None,
            "tools_enabled": enable_tools,
            "lora": lora_info,
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
load_lora_registry()
try:
    load_system()
except Exception as e:
    print(f"[WARN] Could not pre-load system: {e}")
    print("[INFO] System will be loaded on first request")

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
