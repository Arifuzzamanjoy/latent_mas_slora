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

### Latent Reasoning

The core engine implements the LatentMAS realignment equation (W_align from the paper's Eq. 3) to map output hidden states back to input embedding space, enabling multi-step reasoning without token decoding. Features include adaptive step counts based on query complexity, early exit when hidden states converge (cosine similarity threshold), and multiple fusion strategies (mean, weighted, attention, concat-project) for combining agent outputs.

### LoRA Serving — PEFT, not S-LoRA

> **Important**: This project uses [HuggingFace PEFT](https://github.com/huggingface/peft) (`peft>=0.10.0`) for all LoRA operations — loading, switching, merging, and training. The original [S-LoRA](https://arxiv.org/abs/2311.03285) system requires custom CUDA kernels for its unified paging and heterogeneous batching, which are **not** used here.
>
> What this project implements via PEFT:
> - **Dynamic loading** from HuggingFace Hub (`model.load_adapter()`)
> - **Hot-swapping** between adapters (`model.set_adapter()`)
> - **Weighted merging** of multiple adapters (`model.add_weighted_adapter()`)
> - **LRU eviction** when loaded adapters exceed the cap (default: 20)
> - **Custom training** pipeline with `LoraConfig` + `get_peft_model`
>
> What's missing vs. true S-LoRA:
> - No custom CUDA kernels for unified paging
> - No heterogeneous batching (different adapters in the same forward pass)
> - No unified memory pool across base weights + adapter weights + KV cache
>
> Building toward a full S-LoRA-grade serving backend is the long-term vision. See [Contributing](#contributing) if you want to help get there.

## Features

- **Text + Vision**: Process text queries and image+text queries via Qwen2.5-VL
- **Multi-Agent Pipeline**: 4 specialized agents (Planner, Critic, Refiner, Judger) with per-agent LoRA adapters
- **Latent-Space Collaboration**: Agents communicate via hidden states, not generated text — 50–80% fewer tokens, 3–7× faster
- **RAG**: Inject documents via URL, base64, JSON, or inline with domain-aware retrieval
- **Domain Routing**: Auto-classifies queries (medical, math, code, reasoning, general) using keyword, semantic, or hybrid routing
- **LoRA Adapters**: 4 VL-7B adapters in registry, hot-swappable via PEFT, with weighted merging support
- **Custom LoRA Training**: Built-in `LoRATrainer` with gradient checkpointing, mixed precision, and cosine scheduling
- **Session Persistence**: Conversations saved across requests with session grouping
- **RunPod Serverless**: Production-ready Docker deployment on 24–48GB VRAM

## Setup

```bash
git clone https://github.com/Arifuzzamanjoy/latent_mas_slora.git
cd latent_mas_slora
pip install -r requirements.txt
```

### Key Dependencies

```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0          # All LoRA operations
accelerate>=0.27.0
bitsandbytes>=0.43.0  # 4-bit quantization support
sentence-transformers>=2.2.0  # RAG embeddings
qwen-vl-utils[decord]         # VLM image/video processing
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
| `lora_hf_path` | string | null | Direct HuggingFace LoRA path |
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

Adapters are loaded on-demand from HuggingFace Hub, cached locally, and evicted via LRU when the loaded count exceeds `max_loaded_adapters` (default 20). You can also pass any HuggingFace LoRA path directly via the `lora_hf_path` API parameter without adding it to the registry.

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
│   ├── core/
│   │   ├── latent_memory.py   # KV cache manager
│   │   └── latent_reasoner.py # Realignment, continuous thought, fusion
│   ├── lora/
│   │   └── adapter_manager.py # PEFT-based load/switch/merge/evict
│   ├── pipelines/          # Hierarchical and sequential pipelines
│   ├── rag/                # Document store and retriever
│   ├── routing/            # Domain routing (keyword, semantic, hybrid)
│   ├── conversation/       # Session and conversation manager
│   ├── tools/              # Tool registry
│   └── training/           # LoRA trainer (LoraConfig + get_peft_model)
├── examples/
│   └── chat.py             # CLI client
└── .github/workflows/      # CI/CD pipelines
```

## Contributing

This project is open for contributions — especially toward closing the gap between the current PEFT-based LoRA serving and a true S-LoRA-grade backend.

**High-impact areas:**

- **Custom CUDA kernels** — Implement unified paging and heterogeneous batching for multi-adapter serving (the core S-LoRA contribution that's missing today)
- **Multi-adapter batching** — Serve requests targeting different LoRA adapters in a single forward pass, replacing the current one-adapter-at-a-time `set_adapter()` approach
- **Unified memory pool** — Manage base model weights, adapter weights, and KV cache in a single memory pool with dynamic allocation, instead of relying on PEFT's per-adapter memory model
- **Benchmarks** — Throughput and latency benchmarks comparing current PEFT implementation vs. vLLM S-LoRA vs. custom kernel targets
- **New LoRA adapters** — Train and contribute domain-specific VL-7B adapters to the registry
- **Latent-space agents** — Improve fusion strategies, add new agent roles, or optimize the realignment matrix computation

If you're interested, open an issue to discuss your approach before submitting a PR. All contributions welcome — from CUDA veterans to first-time contributors adding adapters or writing tests.

## References

- [LatentMAS](https://arxiv.org/abs/2511.20639) — Latent Multi-Agent Collaboration
- [S-LoRA](https://arxiv.org/abs/2311.03285) — Scalable LoRA Serving
- [Coconut](https://arxiv.org/abs/2412.06769) — Chain of Continuous Thought
- [PEFT](https://github.com/huggingface/peft) — Parameter-Efficient Fine-Tuning

## License

MIT — see [LICENSE](LICENSE)
