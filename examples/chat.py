#!/usr/bin/env python3
"""
Interactive Chat Interface for LatentMAS

Supports two modes:
  1. Local mode  - Loads model locally (requires GPU)
  2. Endpoint mode - Calls a RunPod serverless endpoint (commercial deployment)

Run with: python chat.py [options]

Endpoint mode examples:
  # Interactive chat via endpoint
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY

  # With RAG data from a file
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY --rag-docs ./data/

  # Single prompt with RAG documents
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY \\
      --prompt "Summarize the report" --rag-docs-json '[{"title":"Q3","content":"Revenue was $50M..."}]'
"""

import argparse
import sys
import os
import time
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json


# =============================================================================
# Endpoint Client - Calls RunPod serverless inference endpoint
# =============================================================================

class EndpointClient:
    """
    Commercial-grade client for the LatentMAS RunPod serverless endpoint.

    Supports:
    - Conversational chat with multi-turn history (via conversation_id)
    - Optional RAG document injection (files, URLs, inline JSON)
    - Async job submission with polling
    - Configurable system prompt, temperature, max_tokens
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens: int = 800,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
        no_default_data: bool = True,
        lora_adapter: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        import requests as _requests
        self._requests = _requests

        # Normalise URL (strip trailing slash)
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.no_default_data = no_default_data

        # LoRA adapter (sent with every request)
        self.lora_adapter: Optional[str] = lora_adapter

        # Session / conversation tracking
        self._session_id: Optional[str] = session_id
        self._conversation_id: Optional[str] = None

        # Persistent RAG documents (loaded once, sent with every request)
        self._rag_documents: List[str] = []

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # --------------------------------------------------------------------- #
    #  RAG document loading helpers
    # --------------------------------------------------------------------- #

    def load_rag_from_file(self, path: str) -> int:
        """Load RAG documents from a file or directory. Returns count loaded."""
        import pandas as pd

        path = Path(path)
        count = 0

        if path.is_dir():
            for f in sorted(path.iterdir()):
                if f.suffix in (".json", ".csv", ".txt", ".md"):
                    count += self.load_rag_from_file(str(f))
            return count

        try:
            if path.suffix == ".csv":
                df = pd.read_csv(path)
                content = f"# {path.stem}\n\n"
                content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                for _, row in df.iterrows():
                    for col, val in row.items():
                        if pd.notna(val):
                            content += f"{col}: {val}\n"
                    content += "\n"
                self._rag_documents.append(content)
                count = 1
            elif path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._rag_documents.append(json.dumps(data, indent=2))
                count = 1
            else:
                with open(path, "r", encoding="utf-8") as fh:
                    self._rag_documents.append(fh.read())
                count = 1
            print(f"  Loaded: {path.name}")
        except Exception as e:
            print(f"  Failed to load {path.name}: {e}")

        return count

    def load_rag_from_json_string(self, json_str: str) -> int:
        """Load RAG documents from a JSON array string."""
        docs = json.loads(json_str)
        count = 0
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, dict):
                    self._rag_documents.append(
                        doc.get("content", json.dumps(doc))
                    )
                else:
                    self._rag_documents.append(str(doc))
                count += 1
        return count

    def load_rag_from_url(self, url: str) -> int:
        """Load RAG data from a URL."""
        resp = self._requests.get(url, timeout=30)
        resp.raise_for_status()
        self._rag_documents.append(resp.text)
        print(f"  Loaded RAG data from URL ({len(resp.text)} chars)")
        return 1

    def clear_rag(self):
        """Clear all loaded RAG documents."""
        self._rag_documents.clear()

    # --------------------------------------------------------------------- #
    #  Core chat method
    # --------------------------------------------------------------------- #

    def chat(
        self,
        message: str,
        rag_documents: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a message and get a response from the endpoint.

        Args:
            message: User message
            rag_documents: Additional one-shot RAG docs (not persisted)
            conversation_id: Override conversation_id

        Returns:
            Dict with 'response', 'conversation_id', 'domain', etc.
        """
        # Build payload
        payload: Dict[str, Any] = {
            "prompt": message,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.system_prompt:
            payload["system_prompt"] = self.system_prompt

        # Conversation continuity
        conv_id = conversation_id or self._conversation_id
        if conv_id:
            payload["conversation_id"] = conv_id

        # Session persistence
        if self._session_id:
            payload["session_id"] = self._session_id

        # LoRA adapter selection
        if self.lora_adapter:
            payload["lora_adapter"] = self.lora_adapter

        # RAG documents: persistent + one-shot
        all_rag = list(self._rag_documents)
        if rag_documents:
            all_rag.extend(rag_documents)
        if all_rag:
            payload["rag_documents"] = all_rag

        # Signal not to load default bundled data when we want clean responses
        if self.no_default_data:
            payload["no_default_data"] = True

        # Submit job
        resp = self._requests.post(
            f"{self.endpoint_url}/run",
            json={"input": payload},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        job_data = resp.json()
        job_id = job_data.get("id")

        if not job_id:
            return {"error": "No job ID returned", "raw": job_data}

        # Poll for completion
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > self.timeout:
                return {"error": f"Timeout after {self.timeout}s", "job_id": job_id}

            time.sleep(self.poll_interval)

            status_resp = self._requests.get(
                f"{self.endpoint_url}/status/{job_id}",
                headers=self._headers,
                timeout=30,
            )
            status_data = status_resp.json()
            status = status_data.get("status")

            if status == "COMPLETED":
                output = status_data.get("output", {})
                # Track conversation_id and session_id for continuity
                if output.get("conversation_id"):
                    self._conversation_id = output["conversation_id"]
                if output.get("session_id"):
                    self._session_id = output["session_id"]
                return output

            elif status in ("FAILED", "CANCELLED"):
                return {
                    "error": f"Job {status}",
                    "details": status_data.get("output", {}),
                }

    def reset_conversation(self):
        """Start a new conversation (clear conversation_id)."""
        self._conversation_id = None

    @property
    def conversation_id(self) -> Optional[str]:
        return self._conversation_id

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    # ----------------------------------------------------------------- #
    #  LoRA helpers
    # ----------------------------------------------------------------- #

    def list_loras(self) -> Dict[str, Any]:
        """Query the endpoint for available LoRA adapters."""
        payload = {"list_loras": True}
        resp = self._requests.post(
            f"{self.endpoint_url}/run",
            json={"input": payload},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            return {"error": "No job ID returned"}

        start = time.time()
        while time.time() - start < self.timeout:
            time.sleep(self.poll_interval)
            status_resp = self._requests.get(
                f"{self.endpoint_url}/status/{job_id}",
                headers=self._headers,
                timeout=30,
            )
            st = status_resp.json()
            if st.get("status") == "COMPLETED":
                return st.get("output", {})
            if st.get("status") in ("FAILED", "CANCELLED"):
                return {"error": st.get("status")}
        return {"error": "Timeout"}

    def list_sessions(self) -> Dict[str, Any]:
        """Query the endpoint for persisted sessions."""
        payload = {"list_conversations": True}
        resp = self._requests.post(
            f"{self.endpoint_url}/run",
            json={"input": payload},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            return {"error": "No job ID returned"}

        start = time.time()
        while time.time() - start < self.timeout:
            time.sleep(self.poll_interval)
            status_resp = self._requests.get(
                f"{self.endpoint_url}/status/{job_id}",
                headers=self._headers,
                timeout=30,
            )
            st = status_resp.json()
            if st.get("status") == "COMPLETED":
                return st.get("output", {})
            if st.get("status") in ("FAILED", "CANCELLED"):
                return {"error": st.get("status")}
        return {"error": "Timeout"}


# =============================================================================
# Endpoint-mode interactive chat loop
# =============================================================================

def print_endpoint_header(endpoint_url: str):
    """Print header for endpoint mode."""
    print("\n" + "=" * 70)
    print("LatentMAS Chat - Serverless Endpoint Mode")
    print("=" * 70)
    print(f"Endpoint: {endpoint_url}")
    print("\nCommands:")
    print("  /help      - Show this help message")
    print("  /clear     - Clear conversation history (start new)")
    print("  /rag       - Show loaded RAG document count")
    print("  /addrag    - Add RAG document interactively")
    print("  /clearrag  - Clear all RAG documents")
    print("  /lora NAME - Switch LoRA adapter (e.g. /lora medical_reasoner)")
    print("  /lora off  - Disable LoRA adapter")
    print("  /listlora  - List available LoRA adapters")
    print("  /session   - Show session & conversation info")
    print("  /sessions  - List all persisted sessions on the server")
    print("  /exit      - Exit chat")
    print("\nFeatures:")
    print("  Multi-agent reasoning via serverless GPU endpoint")
    print("  Conversation continuity with persistent session memory")
    print("  Optional RAG data injection (--rag-docs, --rag-data-url)")
    print("  Dynamic LoRA adapter selection (--lora-adapter)")
    print("=" * 70 + "\n")


def endpoint_chat_loop(client: EndpointClient):
    """Interactive chat loop using the RunPod endpoint."""
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]

                if cmd in ("/exit", "/quit"):
                    print("\nGoodbye!")
                    break

                elif cmd == "/help":
                    print_endpoint_header(client.endpoint_url)
                    continue

                elif cmd == "/clear":
                    client.reset_conversation()
                    print("\n[Conversation cleared]\n")
                    continue

                elif cmd == "/rag":
                    print(f"\n[{len(client._rag_documents)} RAG document(s) loaded]\n")
                    continue

                elif cmd == "/addrag":
                    print("Enter document text (end with an empty line):")
                    lines = []
                    while True:
                        line = input()
                        if not line:
                            break
                        lines.append(line)
                    if lines:
                        client._rag_documents.append("\n".join(lines))
                        print(f"[Added 1 document, total: {len(client._rag_documents)}]\n")
                    continue

                elif cmd == "/clearrag":
                    client.clear_rag()
                    print("\n[RAG documents cleared]\n")
                    continue

                elif cmd == "/listlora":
                    print("\n[Fetching available LoRA adapters...]")
                    lora_result = client.list_loras()
                    loras = lora_result.get("loras", {})
                    if loras:
                        print(f"\n  Available LoRA adapters ({len(loras)}):")
                        for name, info in loras.items():
                            desc = info.get("description", "")
                            domain = info.get("domain", "")
                            marker = " ← active" if name == client.lora_adapter else ""
                            print(f"    {name:25s}  [{domain}]  {desc}{marker}")
                    else:
                        print("  No LoRA adapters available")
                    print()
                    continue

                elif cmd.startswith("/lora"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        active = client.lora_adapter or "none"
                        print(f"\n  Active LoRA: {active}")
                        print("  Usage: /lora NAME  or  /lora off\n")
                    elif parts[1].strip().lower() == "off":
                        client.lora_adapter = None
                        print("\n[LoRA adapter disabled]\n")
                    else:
                        client.lora_adapter = parts[1].strip()
                        print(f"\n[LoRA adapter set to: {client.lora_adapter}]\n")
                    continue

                elif cmd == "/session":
                    sid = client.session_id or "(auto-assigned on first message)"
                    cid = client.conversation_id or "(not started)"
                    lora = client.lora_adapter or "none"
                    print(f"\n  Session ID:      {sid}")
                    print(f"  Conversation ID: {cid}")
                    print(f"  Active LoRA:     {lora}")
                    print(f"  RAG documents:   {len(client._rag_documents)}\n")
                    continue

                elif cmd == "/sessions":
                    print("\n[Fetching persisted sessions from server...]")
                    sess_result = client.list_sessions()
                    sessions = sess_result.get("sessions", {})
                    if sessions:
                        for sid, info in sessions.items():
                            convs = info.get("conversations", [])
                            print(f"  Session {sid[:12]}... ({len(convs)} conversation(s))")
                    else:
                        print("  No sessions found")
                    print()
                    continue

                else:
                    print(f"\n[Unknown command: {cmd}]\n")
                    continue

            # Send to endpoint
            print("\nAssistant: ", end="", flush=True)
            result = client.chat(user_input)

            if "error" in result:
                print(f"\n[Error: {result['error']}]")
                if result.get("details"):
                    print(f"[Details: {result['details']}]\n")
            else:
                response_text = result.get("response", "")
                domain = result.get("domain", "")
                confidence = result.get("domain_confidence", 0)

                print(response_text)
                meta_parts = []
                if domain and domain != "general":
                    meta_parts.append(f"{domain} domain ({confidence:.0%})")
                lora_info = result.get("lora", {})
                if lora_info.get("loaded"):
                    meta_parts.append(f"LoRA: {lora_info.get('adapter', '?')}")
                if meta_parts:
                    print(f"  [{' | '.join(meta_parts)}]")
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error: {e}]\n")
            import traceback
            traceback.print_exc()


def endpoint_single_prompt(client: EndpointClient, args):
    """Run a single prompt against the endpoint and exit."""
    prompt = args.prompt

    print(f"\n[Prompt]: {prompt}")
    print("[Waiting for response...]\n")

    result = client.chat(prompt, conversation_id=args.conversation_id)

    if "error" in result:
        print(f"[Error]: {result['error']}")
        if result.get("details"):
            print(f"[Details]: {result['details']}")
    else:
        response = result.get("response", "")
        print(f"[Response]: {response}")

        if args.output_json:
            print(f"\n[JSON Output]:\n{json.dumps(result, indent=2)}")

    return result


def load_data_documents(system, data_dir: Path) -> int:
    """Load documents from CSV and JSON files in data directory"""
    import pandas as pd

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
    from src import LatentMASSystem
    from src.agents.configs import AgentConfig

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
  # ===== ENDPOINT MODE (commercial deployment) =====
  # Interactive chat via RunPod endpoint
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY

  # Single prompt via endpoint
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY \\
      --prompt "What is banking?"

  # With RAG data from local files
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY \\
      --rag-docs ./my_documents/

  # With a specific LoRA adapter
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY \\
      --lora-adapter medical_reasoner --prompt "Explain cardiomyopathy"

  # Resume a previous session
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY \\
      --session-id "abc-123-def"

  # With RAG documents as JSON
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY \\
      --prompt "Summarize" --rag-docs-json '[{"title":"report","content":"Q3 rev $50M"}]'

  # Keep default bundled data (medical/crypto) from server
  python chat.py --endpoint https://api.runpod.ai/v2/YOUR_ID --api-key YOUR_KEY \\
      --use-default-data

  # ===== LOCAL MODE (requires GPU) =====
  # Interactive mode (loads model locally)
  python chat.py --model Qwen/Qwen2.5-VL-7B-Instruct
  
  # Single prompt mode (non-interactive)
  python chat.py --prompt "What is the treatment for hypertension?"
  
  # With external RAG data from URL
  python chat.py --prompt "Summarize the data" --rag-data-url "https://example.com/data.json"
  
  # With custom system prompt and tools
  python chat.py --system-prompt "You are a medical expert" --enable-tools
"""
    )
    
    # ---- Endpoint mode arguments ----
    parser.add_argument(
        "--endpoint",
        type=str,
        help="RunPod endpoint URL (e.g. https://api.runpod.ai/v2/YOUR_ID). "
             "Enables endpoint mode instead of local inference."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("RUNPOD_API_KEY", ""),
        help="RunPod API key (default: RUNPOD_API_KEY env var)"
    )
    parser.add_argument(
        "--use-default-data",
        action="store_true",
        help="In endpoint mode, keep the server's bundled RAG data (medical/crypto). "
             "By default, endpoint mode requests clean responses without bundled data."
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="LoRA adapter name from the server registry (e.g. medical_reasoner, coder_7b)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for grouping conversations. Auto-generated if omitted."
    )

    # ---- Model arguments (local mode) ----
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-VL-7B-Instruct)"
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
    
    # RAG arguments (shared between both modes)
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
        help="Don't load default data from ./data directory (local mode)"
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
    
    # Server mode (local)
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
    
    # ================================================================
    # ENDPOINT MODE - calls RunPod serverless endpoint
    # ================================================================
    if args.endpoint:
        if not args.api_key:
            print("[Error] --api-key required (or set RUNPOD_API_KEY env var)")
            sys.exit(1)

        # Create endpoint client
        client = EndpointClient(
            endpoint_url=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=args.system_prompt,
            no_default_data=not args.use_default_data,
            lora_adapter=args.lora_adapter,
            session_id=args.session_id,
        )

        # Load RAG data if provided
        rag_loaded = 0
        if args.rag_docs:
            print(f"Loading RAG documents from: {args.rag_docs}")
            rag_loaded += client.load_rag_from_file(args.rag_docs)
        if args.rag_data_url:
            print(f"Loading RAG data from URL: {args.rag_data_url}")
            rag_loaded += client.load_rag_from_url(args.rag_data_url)
        if args.rag_docs_json:
            rag_loaded += client.load_rag_from_json_string(args.rag_docs_json)

        if rag_loaded:
            print(f"Loaded {rag_loaded} RAG document(s)\n")

        # Set conversation_id if continuing
        if args.conversation_id:
            client._conversation_id = args.conversation_id

        # Run mode
        if args.prompt:
            endpoint_single_prompt(client, args)
        else:
            if not (args.prompt and args.output_json):
                print_endpoint_header(args.endpoint)
            endpoint_chat_loop(client)

        return

    # ================================================================
    # LOCAL MODE - loads model on this machine (original behaviour)
    # ================================================================

    # Lazy imports for local mode (heavy dependencies)
    from src import LatentMASSystem
    import pandas as pd

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
        print("[API server mode not implemented yet - use --prompt for single queries]")
    elif args.prompt:
        # Single prompt mode
        run_single_prompt(system, args)
    else:
        # Interactive chat loop
        chat_loop(system, use_tools=args.enable_tools)


if __name__ == "__main__":
    main()
