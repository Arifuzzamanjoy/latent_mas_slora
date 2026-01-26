# LatentMAS + S-LoRA Architecture Documentation

> Complete technical reference with example dataflows

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Structures](#data-structures)
4. [Execution Modes](#execution-modes)
5. [Detailed Dataflows](#detailed-dataflows)
6. [LoRA Adapter System](#lora-adapter-system)
7. [Memory Management](#memory-management)
8. [Performance Characteristics](#performance-characteristics)
9. [Mathematical Foundations](#mathematical-foundations)

---

## System Overview

LatentMAS + S-LoRA is a production-grade multi-agent reasoning system that combines two key innovations:

1. **LatentMAS**: Agents communicate via continuous hidden states instead of text tokens
2. **S-LoRA**: Scalable serving of multiple LoRA adapters with near-zero switching overhead

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           LatentMAS + S-LoRA System                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    User Query                                                                   │
│         │                                                                       │
│         ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐      │
│   │                         LatentMASSystem                              │      │
│   │  ┌────────────────────────────────────────────────────────────────┐ │      │
│   │  │                     Pipeline Orchestrator                       │ │      │
│   │  │   (Hierarchical / Sequential / TRUE_LATENT)                    │ │      │
│   │  └──────────────────────────┬─────────────────────────────────────┘ │      │
│   │                              │                                       │      │
│   │     ┌────────────────────────┼────────────────────────┐             │      │
│   │     │                        │                        │             │      │
│   │     ▼                        ▼                        ▼             │      │
│   │  ┌──────────┐         ┌──────────────┐        ┌──────────────┐     │      │
│   │  │  Agent   │         │   Latent     │        │    LoRA      │     │      │
│   │  │   Pool   │◄───────►│   Memory     │◄──────►│   Manager    │     │      │
│   │  │          │         │              │        │              │     │      │
│   │  └────┬─────┘         └──────┬───────┘        └──────┬───────┘     │      │
│   │       │                      │                        │             │      │
│   │       └──────────────────────┼────────────────────────┘             │      │
│   │                              │                                       │      │
│   │                              ▼                                       │      │
│   │              ┌───────────────────────────────────┐                  │      │
│   │              │        Latent Reasoner            │                  │      │
│   │              │  (Realignment + Continuous Loop)  │                  │      │
│   │              └───────────────┬───────────────────┘                  │      │
│   │                              │                                       │      │
│   │                              ▼                                       │      │
│   │              ┌───────────────────────────────────┐                  │      │
│   │              │         Base LLM Model            │                  │      │
│   │              │     (Qwen2.5-3B-Instruct)         │                  │      │
│   │              │     + PEFT LoRA Adapters          │                  │      │
│   │              └───────────────────────────────────┘                  │      │
│   └─────────────────────────────────────────────────────────────────────┘      │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. LatentMASSystem (`src/system.py`)

The main entry point that orchestrates all components.

```python
class LatentMASSystem:
    """
    Unified interface for the multi-agent system.
    
    Attributes:
        config: SystemConfig with model/runtime settings
        model: Base LLM + PEFT wrapper
        tokenizer: HuggingFace tokenizer
        _memory: LatentMemory for inter-agent communication
        _reasoner: LatentReasoner for latent space operations
        _pool: AgentPool for agent lifecycle
        _adapter_manager: LoRAAdapterManager for external LoRAs
        _pipeline: HierarchicalPipeline for execution
    """
```

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `add_agent(config)` | Register agent with LoRA adapter |
| `run(question, pipeline, ...)` | Execute reasoning pipeline |
| `load_external_lora(name, path)` | Load HuggingFace LoRA |
| `clear_memory()` | Reset latent memory between queries |

### 2. AgentPool (`src/agents/agent_pool.py`)

Manages agent lifecycle and LoRA adapter switching.

```python
class AgentPool:
    """
    Pool of agents with dynamic LoRA adapter management.
    
    Key Operations:
    1. register(config) → Add agent + load LoRA
    2. activate(name) → Switch to agent's LoRA adapter
    3. get_active() → Current active agent
    """
```

**LoRA Switching Mechanism:**
```
┌─────────────────────────────────────────────────────┐
│              Agent Activation Flow                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│   activate("MathExpert")                            │
│         │                                            │
│         ▼                                            │
│   ┌─────────────────┐                               │
│   │ Check if loaded │──No──► _load_adapter(config)  │
│   └────────┬────────┘                ↓              │
│            │                  model.add_adapter()   │
│           Yes                        ↓              │
│            │                  LoRA weights loaded   │
│            ▼                                        │
│   ┌─────────────────────────────────┐              │
│   │  model.set_adapter("math_lora") │  ← ~5ms      │
│   └─────────────────────────────────┘              │
│            │                                        │
│            ▼                                        │
│   Agent ready for inference                         │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 3. LatentMemory (`src/core/latent_memory.py`)

Shared working memory for inter-agent latent communication.

```python
class LatentMemory:
    """
    Shared Latent Working Memory.
    
    Storage Types:
    - _agent_states: Dict[str, Tensor] → Per-agent hidden states
    - _latent_buffer: List[Tensor] → Sequential latent accumulator
    - _kv_cache: Tuple → KV cache for continuous generation
    - _agent_outputs: Dict[str, str] → Text outputs (for hybrid mode)
    """
```

**Memory Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│                    LatentMemory Storage                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   _agent_states = {                                          │
│       "Planner": Tensor[1, 2048],    # Hidden state          │
│       "Critic":  Tensor[1, 2048],    # After reasoning       │
│       "Refiner": Tensor[1, 2048],    # Accumulated           │
│   }                                                          │
│                                                              │
│   _latent_buffer = [                                         │
│       Tensor[1, 2048],  # Step 1                            │
│       Tensor[1, 2048],  # Step 2                            │
│       ...               # Up to max_history (50)             │
│   ]                                                          │
│                                                              │
│   _kv_cache = (                                              │
│       (K₀, V₀),  # Layer 0                                   │
│       (K₁, V₁),  # Layer 1                                   │
│       ...        # All 32 layers                             │
│   )  # Shape per layer: [batch, heads, seq_len, head_dim]    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. LatentReasoner (`src/core/latent_reasoner.py`)

Core engine for latent-space reasoning operations.

```python
class LatentReasoner:
    """
    Implements:
    1. Latent space realignment (output → input embedding space)
    2. Continuous thought generation (no token decoding)
    3. Multi-step latent reasoning
    """
```

**Realignment Matrix Construction:**
```
W_align = (W_out^T @ W_out + λI)^(-1) @ W_out^T @ W_in

Where:
- W_in:  Input embedding matrix  [vocab_size, hidden_dim]
- W_out: Output embedding matrix [vocab_size, hidden_dim]
- λ:     Regularization (1e-5)
```

### 5. HierarchicalPipeline (`src/pipelines/hierarchical.py`)

Orchestrates the multi-agent reasoning flow.

```python
class HierarchicalPipeline:
    """
    Pipeline: Planner → Critic → Refiner → Judger
    
    Modes:
    1. run() → Standard (latent + text at each step)
    2. run_true_latent() → Only final agent generates text
    3. run_with_self_consistency() → Multiple samples + voting
    """
```

---

## Data Structures

### AgentConfig

```python
@dataclass
class AgentConfig:
    name: str                    # "Planner", "MathExpert", etc.
    role: AgentRole              # Enum: PLANNER, CRITIC, MATH, etc.
    adapter_name: str            # "planner_lora", "math_lora"
    lora_spec: LoRASpec          # LoRA hyperparameters
    temperature: float           # 0.2 - 0.7 typical
    max_tokens: int              # 350 - 800 typical
    top_p: float                 # 0.9 default
    system_prompt: str           # Role-specific instructions
    user_prompt_template: str    # Input formatting
```

### LoRASpec

```python
@dataclass
class LoRASpec:
    rank: int = 32               # Adapter rank (32-64 typical)
    alpha: int = 64              # Scaling factor (2x rank)
    dropout: float = 0.05        # Regularization
    target_modules: List[str]    # QKV + MLP projections
```

**Default Target Modules for Qwen:**
```python
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    question: str                    # Input query
    final_answer: str                # Generated response
    agent_outputs: List[Dict]        # Per-agent traces
    total_tokens: int                # Input + output tokens
    total_latency_ms: int            # End-to-end time
    latent_steps_total: int          # Sum of all latent steps
    metadata: Dict[str, Any]         # Additional info
```

---

## Execution Modes

### Mode 1: Standard Hierarchical (Latent + Text)

Each agent performs latent reasoning AND generates text.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STANDARD HIERARCHICAL MODE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Question: "What is 25 × 17?"                                          │
│                                                                          │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌────────┐│
│   │   PLANNER   │     │   CRITIC    │     │   REFINER   │     │ JUDGER ││
│   │  (LoRA-P)   │     │  (LoRA-C)   │     │  (LoRA-R)   │     │(LoRA-J)││
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └───┬────┘│
│          │                   │                   │                 │     │
│          ▼                   ▼                   ▼                 ▼     │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│   │ 15 Latent    │   │ 15 Latent    │   │ 15 Latent    │   │ 15 Latent│ │
│   │ Reasoning    │   │ Reasoning    │   │ Reasoning    │   │ Reasoning│ │
│   │ Steps        │   │ Steps        │   │ Steps        │   │ Steps    │ │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └────┬─────┘ │
│          │                   │                   │                │      │
│          ▼                   ▼                   ▼                ▼      │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│   │ Generate     │   │ Generate     │   │ Generate     │   │ Generate │ │
│   │ ~400 tokens  │   │ ~350 tokens  │   │ ~450 tokens  │   │ ~500 tok │ │
│   │ "Break down  │   │ "Check the   │   │ "25×17 =     │   │ "\\boxed │ │
│   │  into..."    │   │  logic..."   │   │  425..."     │   │  {425}"  │ │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └────┬─────┘ │
│          │                   │                   │                │      │
│          ▼                   ▼                   ▼                ▼      │
│        Store               Store               Store             FINAL  │
│      to Memory           to Memory           to Memory          ANSWER  │
│                                                                          │
│   Total: ~2000 tokens, ~70 seconds                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mode 2: TRUE_LATENT (Latent Only Until Final)

Only the final agent generates text—intermediate agents operate entirely in latent space.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       TRUE_LATENT MODE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Question: "What is 25 × 17?"                                          │
│                                                                          │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌────────┐│
│   │   PLANNER   │     │   CRITIC    │     │   REFINER   │     │ JUDGER ││
│   │  (LoRA-P)   │     │  (LoRA-C)   │     │  (LoRA-R)   │     │(LoRA-J)││
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └───┬────┘│
│          │                   │                   │                 │     │
│          ▼                   ▼                   ▼                 ▼     │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│   │ 15 Latent    │   │ 15 Latent    │   │ 15 Latent    │   │ 15 Latent│ │
│   │ Reasoning    │   │ Reasoning    │   │ Reasoning    │   │ Reasoning│ │
│   │ Steps        │   │ Steps        │   │ Steps        │   │ Steps    │ │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └────┬─────┘ │
│          │                   │                   │                │      │
│          ▼                   ▼                   ▼                ▼      │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│   │  NO TEXT     │   │  NO TEXT     │   │  NO TEXT     │   │ Generate │ │
│   │  GENERATION  │   │  GENERATION  │   │  GENERATION  │   │ ~500 tok │ │
│   │  (0 tokens)  │   │  (0 tokens)  │   │  (0 tokens)  │   │ "\\boxed │ │
│   │              │   │              │   │              │   │  {425}"  │ │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └────┬─────┘ │
│          │                   │                   │                │      │
│          ▼                   ▼                   ▼                ▼      │
│     Hidden State →    Hidden State →    Hidden State →        FINAL    │
│     to KV-Cache       to KV-Cache       to KV-Cache          ANSWER    │
│                                                                          │
│   Total: ~800 tokens, ~25 seconds (2.8x faster!)                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mode 3: Fast Mode (Reduced Agents)

Uses 3 agents instead of 4 for faster execution.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FAST MODE                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐       │
│   │   PLANNER   │────────►│   CRITIC    │────────►│   JUDGER    │       │
│   │  (LoRA-P)   │         │  (LoRA-C)   │         │  (LoRA-J)   │       │
│   └─────────────┘         └─────────────┘         └─────────────┘       │
│                                                                          │
│   Skips: REFINER                                                         │
│   Total: ~1500 tokens, ~50 seconds                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Dataflows

### Example: Medical Diagnosis Question

```
Input: "A 45-year-old diabetic patient presents with chest pain radiating to 
        the left arm, diaphoresis, and shortness of breath. ECG shows ST 
        elevation in leads V1-V4. What is the most likely diagnosis?"
```

#### Step-by-Step Dataflow (TRUE_LATENT Mode):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: TOKENIZATION & INITIAL EMBEDDING                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Input Text ──► Tokenizer ──► input_ids [1, 156]                               │
│                                                                                  │
│   Token IDs: [151644, 8948, ..., 151645]                                        │
│   attention_mask: [1, 1, 1, ..., 1]  (all 1s)                                   │
│                                                                                  │
│   Memory State:                                                                  │
│   ┌─────────────────────────────────────┐                                       │
│   │ _agent_states = {}                  │                                       │
│   │ _latent_buffer = []                 │                                       │
│   │ _kv_cache = None                    │                                       │
│   └─────────────────────────────────────┘                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: PLANNER AGENT (LoRA: planner_lora, rank=32)                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   2a. Activate Agent                                                             │
│       pool.activate("Planner")                                                   │
│       └─► model.set_adapter("planner_lora")  [~5ms]                             │
│                                                                                  │
│   2b. Initial Forward Pass                                                       │
│       ┌──────────────────────────────────────────────────────────────────┐      │
│       │ outputs = model(                                                  │      │
│       │     input_ids=[1, 156],                                          │      │
│       │     attention_mask=[1, 156],                                     │      │
│       │     use_cache=True,                                              │      │
│       │     output_hidden_states=True                                    │      │
│       │ )                                                                 │      │
│       │                                                                   │      │
│       │ Returns:                                                          │      │
│       │   hidden_states[-1]: [1, 156, 2048]  ← Last layer output         │      │
│       │   past_key_values: Tuple of 32 layers                            │      │
│       └──────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│   2c. Latent Reasoning Loop (15 steps, NO TOKEN DECODING)                        │
│       ┌──────────────────────────────────────────────────────────────────┐      │
│       │ for step in range(15):                                            │      │
│       │     # Extract last position hidden state                          │      │
│       │     last_hidden = hidden_states[-1][:, -1:, :]  # [1, 1, 2048]   │      │
│       │                                                                   │      │
│       │     # REALIGNMENT: Map output space → input embedding space       │      │
│       │     aligned = realign(last_hidden)                               │      │
│       │     #   aligned = last_hidden @ W_align                          │      │
│       │     #   W_align shape: [2048, 2048]                              │      │
│       │     #   Result: [1, 1, 2048] (ready for next forward)            │      │
│       │                                                                   │      │
│       │     # Forward with embedding (NOT tokens!)                        │      │
│       │     outputs = model(                                              │      │
│       │         inputs_embeds=aligned,      # [1, 1, 2048]               │      │
│       │         attention_mask=[1, 157+step],                            │      │
│       │         past_key_values=kv_cache,                                │      │
│       │         use_cache=True                                           │      │
│       │     )                                                             │      │
│       │                                                                   │      │
│       │     # Update for next iteration                                   │      │
│       │     kv_cache = outputs.past_key_values                           │      │
│       │     hidden_states = outputs.hidden_states                        │      │
│       └──────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│   2d. Store Results (NO TEXT GENERATED)                                          │
│       memory.store_hidden_state("Planner", final_hidden)  # [1, 2048]           │
│       memory.update_kv_cache(kv_cache)                                          │
│                                                                                  │
│   Memory State After Planner:                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │ _agent_states = {"Planner": Tensor[1, 2048]}                        │       │
│   │ _latent_buffer = [Tensor[1, 2048]]                                  │       │
│   │ _kv_cache = (K₀V₀, K₁V₁, ...) with seq_len=171                      │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│   Time: ~6s, Tokens: 156 (input only)                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: CRITIC AGENT (LoRA: critic_lora, rank=32)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   3a. Activate Agent                                                             │
│       pool.activate("Critic")                                                    │
│       └─► model.set_adapter("critic_lora")  [~5ms]                              │
│                                                                                  │
│   3b. Build Minimal Prompt (relies on latent context)                            │
│       prompt = "[System] You are a Critic Agent...                              │
│                [User] Question: A 45-year-old diabetic...                       │
│                       [Latent context from previous agents]"                    │
│                                                                                  │
│   3c. Forward with INHERITED KV-Cache                                            │
│       ┌──────────────────────────────────────────────────────────────────┐      │
│       │ outputs = model(                                                  │      │
│       │     input_ids=[1, 145],                                          │      │
│       │     attention_mask=[1, 171 + 145],  # prev_cache + new           │      │
│       │     past_key_values=memory.get_kv_cache(),  ← FROM PLANNER       │      │
│       │     use_cache=True                                               │      │
│       │ )                                                                 │      │
│       └──────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│   3d. Latent Reasoning Loop (15 steps)                                           │
│       [Same as Planner but with expanded KV-cache context]                       │
│                                                                                  │
│   3e. Store Results (NO TEXT GENERATED)                                          │
│       memory.store_hidden_state("Critic", final_hidden)                          │
│                                                                                  │
│   Memory State After Critic:                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │ _agent_states = {"Planner": ..., "Critic": Tensor[1, 2048]}         │       │
│   │ _latent_buffer = [Tensor, Tensor]  (2 vectors)                      │       │
│   │ _kv_cache seq_len = 171 + 145 + 15 = 331                            │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│   Time: ~6s, Tokens: 145 (input only)                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: REFINER AGENT (LoRA: refiner_lora, rank=48)                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   [Similar flow to Critic - latent only, no text]                                │
│                                                                                  │
│   Memory State After Refiner:                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │ _agent_states = {"Planner": ..., "Critic": ..., "Refiner": ...}     │       │
│   │ _latent_buffer = [Tensor, Tensor, Tensor]  (3 vectors)              │       │
│   │ _kv_cache seq_len = ~490                                            │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
│   Time: ~6s, Tokens: 140 (input only)                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: JUDGER AGENT - FINAL (LoRA: judger_lora, rank=64)                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   5a. Activate Agent                                                             │
│       pool.activate("Judger")                                                    │
│       └─► model.set_adapter("judger_lora")  [~5ms]                              │
│                                                                                  │
│   5b. Latent Reasoning (15 steps)                                                │
│       [Uses accumulated KV-cache from all previous agents]                       │
│                                                                                  │
│   5c. TEXT GENERATION (Final agent ONLY)                                         │
│       ┌──────────────────────────────────────────────────────────────────┐      │
│       │ outputs = model.generate(                                         │      │
│       │     input_ids=[1, 148],                                          │      │
│       │     attention_mask=[1, ~500],  # Full context                    │      │
│       │     max_new_tokens=500,                                          │      │
│       │     temperature=0.2,           # Low for decisive                │      │
│       │     do_sample=True,                                              │      │
│       │     top_p=0.9                                                    │      │
│       │ )                                                                 │      │
│       │                                                                   │      │
│       │ # Decode to text                                                  │      │
│       │ generated = tokenizer.decode(outputs[0][148:])                   │      │
│       └──────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│   Output: """                                                                    │
│   Based on the clinical presentation:                                            │
│   - 45-year-old diabetic with risk factors                                       │
│   - Classic triad: chest pain radiating to left arm, diaphoresis, dyspnea       │
│   - ECG: ST elevation in V1-V4 (anterior leads)                                  │
│                                                                                  │
│   This presentation is pathognomonic for:                                        │
│                                                                                  │
│   \boxed{Acute Anterior STEMI (ST-Elevation Myocardial Infarction)}             │
│                                                                                  │
│   The ST elevation in leads V1-V4 indicates anterior wall involvement,           │
│   typically due to occlusion of the left anterior descending (LAD) artery.       │
│   """                                                                            │
│                                                                                  │
│   Time: ~8s, Tokens: 148 (input) + 287 (output) = 435                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  FINAL RESULT                                                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   PipelineResult(                                                                │
│       question="A 45-year-old diabetic...",                                     │
│       final_answer="Based on the clinical...\boxed{Acute Anterior STEMI}",      │
│       agent_outputs=[                                                            │
│           {"agent": "Planner", "mode": "latent_only", "output_tokens": 0},      │
│           {"agent": "Critic",  "mode": "latent_only", "output_tokens": 0},      │
│           {"agent": "Refiner", "mode": "latent_only", "output_tokens": 0},      │
│           {"agent": "Judger",  "mode": "latent+text", "output_tokens": 287},    │
│       ],                                                                         │
│       total_tokens=876,                                                          │
│       total_latency_ms=26000,                                                    │
│       latent_steps_total=60,                                                     │
│       metadata={                                                                 │
│           "mode": "true_latent",                                                 │
│           "latent_agents": 3,                                                    │
│           "text_agents": 1                                                       │
│       }                                                                          │
│   )                                                                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Latent Reasoning Step Detail

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│           SINGLE LATENT REASONING STEP (Inside the loop)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Input: last_hidden [1, 1, 2048]  (from previous step or initial forward)      │
│                                                                                  │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │  REALIGNMENT                                                            │    │
│   │  ─────────────                                                          │    │
│   │                                                                         │    │
│   │  Purpose: Map output hidden state back to input embedding space         │    │
│   │                                                                         │    │
│   │  h_input = h_output @ W_align                                          │    │
│   │                                                                         │    │
│   │  Where W_align = (W_out^T @ W_out + λI)^(-1) @ W_out^T @ W_in          │    │
│   │                                                                         │    │
│   │  Mathematically:                                                        │    │
│   │  - W_out projects hidden → logits                                       │    │
│   │  - W_in projects tokens → embeddings                                    │    │
│   │  - W_align creates a bridge: hidden → pseudo-embedding                  │    │
│   │                                                                         │    │
│   │  This allows continuous reasoning without token discretization!         │    │
│   │                                                                         │    │
│   │  [1, 1, 2048] @ [2048, 2048] → [1, 1, 2048]                            │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                           │                                                      │
│                           ▼                                                      │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │  FORWARD PASS WITH EMBEDDING                                            │    │
│   │  ──────────────────────────────                                         │    │
│   │                                                                         │    │
│   │  outputs = model(                                                       │    │
│   │      inputs_embeds=aligned_hidden,   # NOT input_ids!                  │    │
│   │      attention_mask=expanded_mask,                                      │    │
│   │      past_key_values=kv_cache,                                         │    │
│   │      use_cache=True,                                                   │    │
│   │      output_hidden_states=True                                         │    │
│   │  )                                                                      │    │
│   │                                                                         │    │
│   │  Key insight: We're feeding a continuous vector, not discrete tokens    │    │
│   │  The model "thinks" without producing words                             │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                           │                                                      │
│                           ▼                                                      │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │  UPDATE STATE                                                           │    │
│   │  ────────────                                                           │    │
│   │                                                                         │    │
│   │  kv_cache = outputs.past_key_values    # Expanded by 1 position        │    │
│   │  last_hidden = outputs.hidden_states[-1][:, -1:, :]                    │    │
│   │                                                                         │    │
│   │  The KV-cache now contains the "thought" at this step                  │    │
│   │  This accumulates reasoning context for subsequent steps               │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│   Output: last_hidden [1, 1, 2048], kv_cache (seq_len + 1)                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## LoRA Adapter System

### Adapter Configuration Per Agent

| Agent | Adapter Name | Rank | Alpha | Target Modules | Memory |
|-------|--------------|------|-------|----------------|--------|
| Planner | `planner_lora` | 32 | 64 | QKV+MLP | ~50MB |
| Critic | `critic_lora` | 32 | 64 | QKV+MLP | ~50MB |
| Refiner | `refiner_lora` | 48 | 96 | QKV+MLP | ~75MB |
| Judger | `judger_lora` | 64 | 128 | QKV+MLP | ~100MB |
| MathExpert | `math_lora` | 48 | 96 | QKV+MLP | ~75MB |
| MedicalExpert | `medical_lora` | 64 | 128 | QKV+MLP | ~100MB |
| CodeExpert | `coder_lora` | 64 | 128 | QKV+MLP | ~100MB |

### Dynamic Switching Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      S-LoRA DYNAMIC SWITCHING                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Base Model Weights (Frozen): W_base [hidden_dim, hidden_dim]                  │
│                                                                                  │
│   LoRA Decomposition:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                          │   │
│   │   W_adapted = W_base + (α/r) × B × A                                    │   │
│   │                                                                          │   │
│   │   Where:                                                                 │   │
│   │   - A: [hidden_dim, rank]  (down projection)                            │   │
│   │   - B: [rank, hidden_dim]  (up projection)                              │   │
│   │   - α: scaling factor (typically 2×rank)                                │   │
│   │   - r: rank (32-64)                                                     │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   Adapter Storage:                                                               │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │  model.peft_config = {                                                    │  │
│   │      "planner_lora": LoraConfig(r=32, ...),                              │  │
│   │      "critic_lora":  LoraConfig(r=32, ...),                              │  │
│   │      "refiner_lora": LoraConfig(r=48, ...),                              │  │
│   │      "judger_lora":  LoraConfig(r=64, ...),                              │  │
│   │      "math_lora":    LoraConfig(r=48, ...),                              │  │
│   │      ...                                                                  │  │
│   │  }                                                                        │  │
│   │                                                                           │  │
│   │  Total: 20+ adapters can be loaded simultaneously                        │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│   Switching Operation:                                                           │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │  model.set_adapter("math_lora")                                          │  │
│   │                                                                           │  │
│   │  Internally:                                                              │  │
│   │  1. Deactivate current adapter weights                                   │  │
│   │  2. Activate new adapter weights (A_new, B_new)                          │  │
│   │  3. Update forward hooks                                                 │  │
│   │                                                                           │  │
│   │  Time: ~5ms (no weight copying, just pointer update)                     │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### External LoRA Registry

```python
QWEN25_LORA_REGISTRY = {
    "medical_reasoner": ExternalLoRAInfo(
        hf_path="iimran/Qwen2.5-3B-R1-MedicalReasoner-lora-adapter",
        domain="medical"
    ),
    "math_instruct": ExternalLoRAInfo(
        hf_path="SKNahin/Qwen2.5-Math-7B-Instruct-bnb-4bit-lora",
        domain="math"
    ),
    "coder_7b": ExternalLoRAInfo(
        hf_path="Alexis-Az/Qwen-2.5-Coder-7B-Instruct-LoRA",
        domain="code"
    ),
    ...
}
```

---

## Memory Management

### VRAM Budget (48GB System)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         VRAM ALLOCATION (48GB)                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Base Model (Qwen2.5-3B-Instruct, BF16)              │    ~6.0 GB       │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  LoRA Adapters (20 × ~50MB average)                  │    ~1.0 GB       │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  KV-Cache (32K tokens, BF16)                         │    ~3.0 GB       │   │
│   │    Formula: 2 × num_layers × num_heads × head_dim × seq_len × 2 bytes  │   │
│   │    = 2 × 32 × 32 × 128 × 32768 × 2 ≈ 3.2GB                             │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  Hidden State Buffer (50 vectors × 2048 × 2 bytes)   │    ~0.2 GB       │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  Activation Memory (peak during forward)             │    ~4.0 GB       │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  PyTorch/CUDA Overhead                               │    ~2.0 GB       │   │
│   ├─────────────────────────────────────────────────────────────────────────┤   │
│   │  AVAILABLE HEADROOM                                  │   ~32.0 GB       │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   Total Used: ~16GB | Available: ~32GB                                          │
│   This allows: Larger batches, longer contexts, more adapters                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### KV-Cache Compression

When context exceeds threshold (16K tokens), compression is applied:

```python
def _compress_cache(kv_cache, keep_ratio=0.5):
    """
    Strategy: Keep first 25% (system prompt) + last 50% (recent context)
    
    Before: [tok_0, tok_1, ..., tok_16000]
    After:  [tok_0...tok_4000, tok_8000...tok_16000]
    
    This preserves critical context while reducing memory.
    """
```

---

## Performance Characteristics

### Mode Comparison

| Metric | Normal | Fast | TRUE_LATENT |
|--------|--------|------|-------------|
| **Agents** | 4 | 3 | 4 |
| **Text Gen** | 4× | 3× | 1× (final only) |
| **Latent Steps** | 60 | 45 | 60 |
| **Output Tokens** | ~2100 | ~1500 | ~800 |
| **Latency** | ~73s | ~50s | ~25s |
| **Speedup** | 1.0× | 1.5× | **2.8-3.0×** |

### Latency Breakdown (TRUE_LATENT Mode)

```
┌───────────────────────────────────────────────────────────────────┐
│  Agent          │ Latent (ms) │ Text Gen (ms) │ Total (ms)       │
├───────────────────────────────────────────────────────────────────┤
│  Planner        │    5,800    │       0       │   5,800          │
│  Critic         │    5,500    │       0       │   5,500          │
│  Refiner        │    5,700    │       0       │   5,700          │
│  Judger         │    5,600    │   3,200       │   8,800          │
├───────────────────────────────────────────────────────────────────┤
│  TOTAL          │   22,600    │   3,200       │  25,800          │
└───────────────────────────────────────────────────────────────────┘

Key insight: 87% of time is latent reasoning, 13% is text generation
In Normal mode, text generation would be 4× → ~12,800ms additional
```

---

## Mathematical Foundations

### Latent Space Realignment (LatentMAS Paper)

The core innovation enabling continuous reasoning:

$$W_{align} = (W_{out}^T W_{out} + \lambda I)^{-1} W_{out}^T W_{in}$$

Where:
- $W_{in} \in \mathbb{R}^{V \times D}$ : Input embedding matrix
- $W_{out} \in \mathbb{R}^{V \times D}$ : Output embedding matrix  
- $\lambda = 10^{-5}$ : Regularization term
- $V$ : Vocabulary size (~150K for Qwen)
- $D$ : Hidden dimension (2048 for Qwen-3B)

**Interpretation:**
- $W_{out}^T W_{out}$ : Gram matrix of output embeddings
- $(...)^{-1}$ : Pseudo-inverse with regularization
- Maps hidden states back to input embedding space
- Enables feeding hidden states as "pseudo-tokens"

### LoRA Low-Rank Adaptation

$$W_{adapted} = W_{base} + \frac{\alpha}{r} \cdot B \cdot A$$

Where:
- $W_{base} \in \mathbb{R}^{D_{out} \times D_{in}}$ : Frozen base weights
- $A \in \mathbb{R}^{r \times D_{in}}$ : Down projection (rank $r$)
- $B \in \mathbb{R}^{D_{out} \times r}$ : Up projection
- $\alpha$ : Scaling factor (typically $2r$)

**Memory Savings:**
- Full fine-tune: $D_{out} \times D_{in}$ parameters
- LoRA: $r \times (D_{in} + D_{out})$ parameters
- For $D = 2048$, $r = 32$: 97% parameter reduction

### Latent Fusion Strategies

```python
# Mean Fusion
h_fused = mean([h_1, h_2, ..., h_n])

# Weighted Fusion (recent = more important)
weights = exp(linspace(-2, 0, n))  # Exponential decay
h_fused = sum(w_i * h_i) / sum(w_i)

# Attention Fusion
scores = softmax(query @ [h_1, h_2, ..., h_n]^T / sqrt(d))
h_fused = scores @ [h_1, h_2, ..., h_n]
```

---

## References

1. **LatentMAS**: "Latent Multi-Agent Collaboration" - Hidden state communication
2. **S-LoRA**: "Scalable Serving of Thousands of LoRA Adapters" (arXiv:2311.03285)
3. **LoRA**: "Low-Rank Adaptation of Large Language Models" (arXiv:2106.09685)
4. **Coconut**: "Training Large Language Models to Reason in a Continuous Latent Space"
5. **PEFT**: Parameter-Efficient Fine-Tuning library by HuggingFace

---

## Quick Reference

### Initialization
```python
system = LatentMASSystem(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    dtype="bfloat16",
    latent_steps=15,
    max_loaded_adapters=20
)
system.add_agent(AgentConfig.planner())
system.add_agent(AgentConfig.critic())
system.add_agent(AgentConfig.judger())
```

### Execution
```python
# TRUE_LATENT (fastest)
result = system.run(question, true_latent=True)

# Standard hierarchical
result = system.run(question, pipeline="hierarchical")

# With self-consistency voting
result = system.run(question, self_consistency=3)
```

### Custom Agents
```python
custom = AgentConfig(
    name="DomainExpert",
    role=AgentRole.CUSTOM,
    adapter_name="custom_lora",
    lora_spec=LoRASpec(rank=48, alpha=96),
    temperature=0.4,
    max_tokens=600,
    system_prompt="You are an expert in..."
)
system.add_agent(custom)
```
