# Usage Guide

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────┐
│           handler.py (RunPod)           │
│  ┌─────────┐ ┌─────┐ ┌──────────────┐  │
│  │ Domain   │ │ RAG │ │ LoRA Manager │  │
│  │ Router   │ │     │ │ (4 adapters) │  │
│  └────┬─────┘ └──┬──┘ └──────┬───────┘  │
│       └──────────┴───────────┘          │
│                  │                      │
│       ┌──────────▼──────────┐           │
│       │   LatentMAS System  │           │
│       │                     │           │
│       │ Planner ──► Critic  │           │
│       │   (text)   (latent) │           │
│       │              │      │           │
│       │ Judger  ◄── Refiner │           │
│       │  (text)    (latent) │           │
│       └─────────────────────┘           │
│                  │                      │
│       ┌──────────▼──────────┐           │
│       │  Qwen2.5-VL-7B      │           │
│       │  (base model, BF16) │           │
│       └─────────────────────┘           │
└─────────────────────────────────────────┘
```

- **Planner**: Decomposes the problem, generates text (~8s)
- **Critic**: Evaluates in hidden-state space, no text (~0.2s)
- **Refiner**: Refines in hidden-state space, no text (~0.2s)
- **Judger**: Produces final answer as text (~10-35s depending on `max_tokens`)
- **VLM path**: Image+text queries bypass the pipeline, go directly to the model

---

## Serverless Endpoint (RunPod)

Base URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/run` | Async request, returns `id` |
| POST | `/runsync` | Sync request, waits for response |
| GET | `/status/<id>` | Check async request status |

### Request Schema

```json
{
  "input": {
    "prompt": "string (required)",
    "max_tokens": 800,
    "temperature": 0.7,
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "system_prompt": "custom system prompt",
    "image_url": "https://...",
    "image_base64": "base64-string",
    "session_id": "user-123",
    "conversation_id": "conv-456",
    "lora_adapter": "medical_vl",
    "lora_hf_path": "username/custom-lora",
    "no_default_data": true,
    "rag_data": "https://example.com/data.json",
    "rag_documents": [{"title": "doc1", "content": "..."}],
    "enable_tools": false,
    "list_loras": true,
    "list_conversations": true
  }
}
```

### Response Schema

```json
{
  "response": "The answer...",
  "conversation_id": "conv-456",
  "session_id": "user-123",
  "domain": "math",
  "domain_confidence": 0.85,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "vlm": true,
  "image_provided": false,
  "rag_enabled": true,
  "tools_enabled": false,
  "lora": {"loaded": false}
}
```

---

## Use Cases

### 1. Basic Text Query

```bash
curl -X POST "https://api.runpod.ai/v2/<ID>/runsync" \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is quantum entanglement? 2 sentences.", "max_tokens": 200}}'
```

### 2. Image Analysis (VLM)

```bash
curl -X POST "https://api.runpod.ai/v2/<ID>/runsync" \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Describe this image",
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
      "max_tokens": 300
    }
  }'
```

### 3. RAG with Injected Documents

```bash
curl -X POST "https://api.runpod.ai/v2/<ID>/runsync" \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What is the revenue?",
      "rag_documents": [
        {"title": "Report", "content": "Acme Corp reported $4.2B revenue in 2025."}
      ],
      "max_tokens": 200
    }
  }'
```

### 4. RAG from URL

```bash
curl -X POST "https://api.runpod.ai/v2/<ID>/runsync" \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Summarize the key data points",
      "rag_data": "https://example.com/data.json",
      "max_tokens": 400
    }
  }'
```

### 5. Custom System Prompt

```bash
curl -X POST "https://api.runpod.ai/v2/<ID>/runsync" \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Explain neural networks",
      "system_prompt": "You are a teacher for 10-year-olds. Use simple words.",
      "max_tokens": 300
    }
  }'
```

### 6. Session Continuity

First request:
```json
{"input": {"prompt": "My name is Alex", "session_id": "user-1", "max_tokens": 200}}
```

Follow-up (remembers context):
```json
{"input": {"prompt": "What is my name?", "session_id": "user-1", "max_tokens": 200}}
```

### 7. LoRA Adapter Selection

```json
{"input": {"prompt": "Diagnose chest pain with dyspnea", "lora_adapter": "medical_vl", "max_tokens": 400}}
```

Load any HuggingFace LoRA:
```json
{"input": {"prompt": "Your query", "lora_hf_path": "username/my-vl7b-lora", "max_tokens": 400}}
```

### 8. List Available LoRAs

```json
{"input": {"list_loras": true}}
```

### 9. List Saved Conversations

```json
{"input": {"list_conversations": true}}
```

### 10. Use Built-in Data (crypto, medical)

```json
{"input": {"prompt": "Bitcoin market cap?", "no_default_data": false, "max_tokens": 300}}
```

---

## CLI (chat.py)

### Local Inference

```bash
# Interactive chat
python examples/chat.py

# Single prompt
python examples/chat.py --prompt "What is AI?" --max-tokens 200

# Skip default data files
python examples/chat.py --prompt "2+2?" --no-default-data --max-tokens 200

# Custom system prompt
python examples/chat.py --system-prompt "You are a doctor" --prompt "Headache causes?"

# With local RAG documents
python examples/chat.py --rag-docs ./my_documents/ --prompt "Summarize docs"

# With RAG from URL
python examples/chat.py --rag-data-url "https://example.com/data.json" --prompt "Analyze"

# With inline RAG documents
python examples/chat.py --rag-docs-json '[{"title":"doc","content":"..."}]' --prompt "Query"

# Enable tools
python examples/chat.py --enable-tools --prompt "Calculate 123 * 456"

# JSON output
python examples/chat.py --prompt "What is AI?" --output-json
```

### Endpoint Mode (via RunPod)

```bash
# Single prompt via endpoint
python examples/chat.py \
  --endpoint https://api.runpod.ai/v2/<ID> \
  --api-key <KEY> \
  --prompt "What is AI?"

# Interactive chat via endpoint
python examples/chat.py \
  --endpoint https://api.runpod.ai/v2/<ID> \
  --api-key <KEY>

# With LoRA adapter
python examples/chat.py \
  --endpoint https://api.runpod.ai/v2/<ID> \
  --api-key <KEY> \
  --lora-adapter medical_vl \
  --prompt "Diagnose fever and cough"

# With RAG data URL
python examples/chat.py \
  --endpoint https://api.runpod.ai/v2/<ID> \
  --api-key <KEY> \
  --rag-data-url "https://example.com/data.csv" \
  --prompt "Summarize"

# With session persistence
python examples/chat.py \
  --endpoint https://api.runpod.ai/v2/<ID> \
  --api-key <KEY> \
  --session-id my-session
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | — | Single prompt (non-interactive) |
| `--endpoint` | — | RunPod endpoint URL (enables endpoint mode) |
| `--api-key` | `$RUNPOD_API_KEY` | RunPod API key |
| `--model` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model name (local mode) |
| `--max-tokens` | 800 | Max tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--system-prompt` | default | Custom system prompt |
| `--no-default-data` | false | Skip built-in data files |
| `--use-default-data` | false | Use server's bundled data (endpoint mode) |
| `--lora-adapter` | — | LoRA adapter name from registry |
| `--session-id` | auto | Session ID for continuity |
| `--conversation-id` | — | Continue existing conversation |
| `--rag-docs` | — | Local docs path for RAG |
| `--rag-data-url` | — | URL to download RAG data |
| `--rag-docs-json` | — | Inline JSON documents |
| `--enable-tools` | false | Enable tool use |
| `--output-json` | false | Output as JSON |
| `--serve-api` | false | Start local API server |
| `--port` | 8000 | Local API server port |

---

## LoRA Adapters

Available in `data/lora_registry.json` (base model: `Qwen2.5-VL-7B-Instruct`):

| Name | HuggingFace | Domain |
|------|-------------|--------|
| `medical_vl` | sarathi-balakrishnan/Qwen2.5-VL-7B-Medical-LoRA | Medical |
| `reward_vl` | DJ-Kim/Qwen2.5_VL_7B_Reward_LoRA | General |
| `comics_vl` | VLR-CVC/Qwen2.5-VL-7B-Instruct-lora-ComicsPAP | Comics |
| `point_detect_vl` | SimulaMet/PointDetectCount-Qwen2.5-VL-7B-LoRA | Detection |

To add a new adapter, edit `data/lora_registry.json`:
```json
"my_adapter": {
  "hf_path": "username/My-LoRA",
  "description": "What it does",
  "domain": "general",
  "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "tags": ["tag1"],
  "verified": true
}
```

Constraint: LoRA must be trained on `Qwen2.5-VL-7B-Instruct`.
