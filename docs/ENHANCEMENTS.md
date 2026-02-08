# LatentMAS-SLoRA Enhancement Roadmap

## Current Performance Baseline
- **Accuracy**: 80% (vs 60% Traditional)
- **Speed**: 1.2-1.3x faster than Traditional 4-Agent RAG
- **Token Reduction**: 65-70% fewer tokens

## Research-Backed Enhancements

### 1. 🚀 Speculative Latent Decoding (High Priority)
**Paper Reference**: [Medusa](https://arxiv.org/abs/2401.10774), [SpecInfer](https://arxiv.org/abs/2305.09781)

**Concept**: Instead of generating tokens one-by-one in the final agent, predict multiple tokens in parallel and verify.

**Implementation**:
```python
# Add speculative heads that predict future tokens from latent state
class SpeculativeLatentHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_heads=4):
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size) for _ in range(num_heads)
        ])
    
    def forward(self, latent_state):
        # Predict next 4 tokens in parallel
        return [head(latent_state) for head in self.heads]
```

**Expected Speedup**: 2-3x for final text generation

---

### 2. 🧠 Coconut-Style Continuous Thought (High Priority)
**Paper Reference**: [Coconut](https://arxiv.org/abs/2412.06769) - "Training LLMs to Reason in Continuous Latent Space"

**Concept**: Enable breadth-first search (BFS) reasoning in latent space by maintaining multiple parallel thought paths.

**Key Insight**: Current LatentMAS commits to a single reasoning path. Coconut shows that latent thoughts can encode MULTIPLE alternative next steps, allowing parallel exploration.

**Implementation**:
```python
def reason_bfs(self, input_ids, num_branches=3, depth=5):
    """BFS reasoning in latent space"""
    current_states = [self.get_initial_latent(input_ids)]
    
    for step in range(depth):
        next_states = []
        for state in current_states:
            # Each state spawns multiple branches
            branches = self.branch_latent(state, num_branches)
            next_states.extend(branches)
        
        # Prune to top-k by quality score
        current_states = self.prune_states(next_states, top_k=num_branches)
    
    return self.aggregate_states(current_states)
```

**Expected Improvement**: Better reasoning accuracy, especially for complex multi-step problems

---

### 3. ⚡ Parallel Agent Reasoning (Medium Priority)
**Paper Reference**: [DyLAN](https://arxiv.org/abs/2310.02170) - Dynamic LLM-Powered Agent Network

**Concept**: Run independent agents in parallel instead of sequentially.

**Current Flow**:
```
Planner → Critic → Refiner → Judger (Sequential: 4x latency)
```

**Optimized Flow**:
```
        ┌─→ Critic ──┐
Planner─┤            ├→ Aggregator → Judger (Parallel: ~2x latency)
        └─→ Refiner ─┘
```

**Implementation**:
```python
async def run_parallel_agents(self, question, latent_state):
    """Run Critic and Refiner in parallel"""
    async with asyncio.TaskGroup() as tg:
        critic_task = tg.create_task(self.run_agent("Critic", latent_state))
        refiner_task = tg.create_task(self.run_agent("Refiner", latent_state))
    
    # Aggregate latent states
    combined = self.aggregate_latent([critic_task.result(), refiner_task.result()])
    return combined
```

**Expected Speedup**: 1.5-2x for multi-agent pipeline

---

### 4. 💾 KV Cache Compression (Medium Priority)
**Paper Reference**: [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180)

**Concept**: Reduce memory footprint of KV cache using paging and compression.

**Implementation Options**:
1. **Sliding Window**: Keep only last N tokens in cache
2. **Token Merging**: Merge similar KV pairs
3. **Quantized Cache**: Use INT8 for cached values

```python
class CompressedKVCache:
    def __init__(self, max_tokens=8192, compression='sliding'):
        self.max_tokens = max_tokens
        self.compression = compression
    
    def update(self, new_kv):
        if self.compression == 'sliding':
            # Keep only last max_tokens
            return self._sliding_window(new_kv)
        elif self.compression == 'merge':
            # Merge similar tokens
            return self._token_merge(new_kv)
```

**Expected Improvement**: 2x longer context, 30% memory reduction

---

### 5. 🎯 Adaptive Latent Steps (Medium Priority)
**Concept**: Dynamically adjust the number of latent reasoning steps based on question complexity.

**Current**: Fixed 10-15 steps for all questions
**Proposed**: 3-20 steps based on complexity detection

**Implementation**:
```python
def estimate_complexity(self, question):
    """Estimate question complexity"""
    features = {
        'length': len(question.split()),
        'has_math': bool(re.search(r'[\d+\-*/=]', question)),
        'has_code': 'code' in question.lower() or 'function' in question.lower(),
        'has_comparison': 'compare' in question.lower() or 'vs' in question.lower(),
    }
    
    score = sum([
        features['length'] / 20,
        features['has_math'] * 0.3,
        features['has_code'] * 0.3,
        features['has_comparison'] * 0.2,
    ])
    
    # Map to latent steps: simple=3, medium=10, complex=20
    return int(3 + score * 17)
```

**Expected Improvement**: 30% faster for simple questions, same quality

---

### 6. 🔄 Dynamic Agent Selection (Low Priority)
**Paper Reference**: [DyLAN](https://arxiv.org/abs/2310.02170)

**Concept**: Not all questions need all 4 agents. Select minimal agent set.

**Example**:
- Simple factual → Planner + Judger only
- Math → Planner + MathCritic + Judger
- Complex reasoning → Full 4-agent pipeline

---

### 7. 📊 Inference Scaling Laws (Low Priority)
**Paper Reference**: [Inference Scaling Laws](https://arxiv.org/abs/2408.00724)

**Key Insight**: "Smaller models + advanced inference > Larger models + basic inference"

**Application**: Use 3B model with enhanced latent reasoning instead of 7B with basic decoding.

---

## Implementation Priority

| Enhancement | Complexity | Expected Speedup | Priority |
|------------|------------|------------------|----------|
| Speculative Latent Decoding | High | 2-3x | 🔴 High |
| Coconut BFS Reasoning | High | +10-20% accuracy | 🔴 High |
| Parallel Agent Reasoning | Medium | 1.5-2x | 🟡 Medium |
| KV Cache Compression | Medium | +30% memory | 🟡 Medium |
| Adaptive Latent Steps | Low | 30% faster (simple Q) | 🟡 Medium |
| Dynamic Agent Selection | Low | 20% faster (simple Q) | 🟢 Low |

---

## Quick Wins (Implemented)

### ✅ Suppress Unnecessary Warnings
- Removed verbose embedding model warnings
- Cleaner output logs

### ✅ Medical Domain Boosting
- Enhanced medical keyword detection
- Improved domain routing for medical questions

### ✅ Better Answer Extraction
- Added `\boxed{X}` pattern recognition
- More robust MCQ answer parsing

---

## Next Steps

1. **Phase 1**: Implement Adaptive Latent Steps (quick win)
2. **Phase 2**: Add Parallel Agent Reasoning
3. **Phase 3**: Integrate Speculative Decoding for final agent
4. **Phase 4**: Implement Coconut-style BFS reasoning

---

## References

1. [Medusa: Simple LLM Inference Acceleration](https://arxiv.org/abs/2401.10774)
2. [Coconut: Training LLMs to Reason in Continuous Latent Space](https://arxiv.org/abs/2412.06769)
3. [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)
4. [DyLAN: Dynamic LLM-Powered Agent Network](https://arxiv.org/abs/2310.02170)
5. [SpecInfer: Tree-based Speculative Inference](https://arxiv.org/abs/2305.09781)
6. [Inference Scaling Laws](https://arxiv.org/abs/2408.00724)
