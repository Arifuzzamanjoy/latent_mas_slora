# LatentMAS + S-LoRA

Multi-agent reasoning system with VLM support, latent-space collaboration, and dynamic LoRA serving. Runs as a RunPod serverless endpoint.

[![CI](https://github.com/Arifuzzamanjoy/latent_mas_slora/actions/workflows/ci.yml/badge.svg)](https://github.com/Arifuzzamanjoy/latent_mas_slora/actions/workflows/ci.yml)
[![Docker CD](https://github.com/Arifuzzamanjoy/latent_mas_slora/actions/workflows/cd-docker.yml/badge.svg)](https://github.com/Arifuzzamanjoy/latent_mas_slora/actions/workflows/cd-docker.yml)

## Architecture

**Base Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (8B params, BF16, VLM)

**Pipeline**: Planner → Critic (latent) → Refiner (latent) → Judger

- Critic and Refiner operate in hidden-state space (~200ms each), no text generation
- Planner decomposes the problem, Judger produces the final answer
- Domain routing auto-selects the best LoRA adapter per query

## Features

- **Text + Vision**: Process text queries and image+text queries (VLM)
- **Multi-Agent Pipeline**: 4 specialized agents with LoRA adapters
- **RAG**: Inject documents via URL, base64, JSON, or inline
- **Domain Routing**: Auto-classifies queries (medical, math, code, reasoning, general)
- **LoRA Adapters**: 4 VL-7B adapters, hot-swappable via registry
- **Session Persistence**: Conversations saved across requests
- **RunPod Serverless**: Production-ready Docker deployment

## Setup

```bash
git clone https://github.com/Arifuzzamanjoy/latent_mas_slora.git
cd latent_mas_slora
pip install -r requirements.txt
```

## Deployment

```bash
# Docker image
docker.io/s1710374103/latent-mas-slora:latest
```

1. Create a RunPod serverless endpoint with the Docker image
2. Configure: **24GB+ VRAM**, **30GB+ disk**

## API Reference

### Request

```json
{
  "input": {
    "prompt": "What is the treatment for hypertension?",
    "max_tokens": 800,
    "temperature": 0.7,
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "system_prompt": "You are a helpful AI assistant.",
    "image_url": "https://example.com/image.jpg",
    "image_base64": "<base64-string>",
    "session_id": "user-123",
    "conversation_id": "conv-456",
    "lora_adapter": "medical_vl",
    "lora_hf_path": "username/custom-lora",
    "no_default_data": true,
    "rag_data": "https://example.com/data.json",
    "rag_documents": [{"title": "doc1", "content": "..."}],
    "enable_tools": false,
    "list_loras": false,
    "list_conversations": false
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | The query text |
| `max_tokens` | int | 800 | Max tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `image_url` | string | null | Image URL for VLM inference |
| `image_base64` | string | null | Base64 image for VLM inference |
| `session_id` | string | auto | Session ID for persistence |
| `conversation_id` | string | auto | Conversation ID for context |
| `lora_adapter` | string | null | LoRA adapter name from registry |
| `no_default_data` | bool | false | Skip built-in RAG data |
| `rag_data` | string | null | External RAG data URL |
| `rag_documents` | array | [] | Inline documents for RAG |
| `system_prompt` | string | null | Custom system prompt |
| `list_loras` | bool | false | Return available LoRA adapters |
| `list_conversations` | bool | false | Return saved sessions |

### Response

```json
{
  "response": "The treatment for hypertension includes...",
  "conversation_id": "conv-456",
  "session_id": "user-123",
  "domain": "medical",
  "domain_confidence": 0.85,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "vlm": true,
  "image_provided": false,
  "rag_enabled": true,
  "tools_enabled": false,
  "lora": {"loaded": false}
}
```

### curl Example

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is AI?", "max_tokens": 500}}'
```

## CLI

```bash
# Single prompt
python examples/chat.py --prompt "What is AI?" --no-default-data

# With RAG data
python examples/chat.py --prompt "Summarize this" \
  --rag-data-url "https://example.com/data.json"

# With image (VLM)
python examples/chat.py --prompt "Describe this image" \
  --image-url "https://example.com/photo.jpg"
```

## LoRA Adapters

Available in `data/lora_registry.json`:

| Name | HuggingFace Path | Domain |
|------|------------------|--------|
| medical_vl | sarathi-balakrishnan/Qwen2.5-VL-7B-Medical-LoRA | Medical |
| reward_vl | DJ-Kim/Qwen2.5-VL-7B-Reward-LoRA | Reward |
| comics_vl | VLR-CVC/Qwen2.5-VL-7B-Comics-LoRA | Comics |
| point_detect_vl | SimulaMet/Qwen2.5-VL-7B-PointDetection-LoRA | Detection |

## Project Structure

```
├── handler.py              # RunPod serverless handler
├── Dockerfile              # Production Docker image
├── requirements.txt
├── data/
│   └── lora_registry.json  # External LoRA adapter registry
├── src/
│   ├── system.py           # Core LatentMAS system
│   ├── agents/             # Agent configs and pool
│   ├── core/               # Latent memory and reasoner
│   ├── lora/               # LoRA adapter manager
│   ├── pipelines/          # Hierarchical and sequential pipelines
│   ├── rag/                # RAG document store and retriever
│   ├── routing/            # Domain routing (semantic, fast, advanced)
│   ├── conversation/       # Session and conversation manager
│   └── tools/              # Tool registry
├── examples/
│   └── chat.py             # CLI client
└── .github/workflows/      # CI/CD pipelines
```

## References

- [LatentMAS](https://arxiv.org/abs/2511.20639) — Latent Multi-Agent Collaboration
- [S-LoRA](https://arxiv.org/abs/2311.03285) — Scalable LoRA Serving
- [Coconut](https://arxiv.org/abs/2412.06769) — Chain of Continuous Thought

## License

MIT — see [LICENSE](LICENSE)
