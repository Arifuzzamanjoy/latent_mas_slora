#!/usr/bin/env python3
"""Diagnose why multi-agent configs lose accuracy vs single model."""
import sys, os, re, torch, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.benchmarks import BenchmarkRunner, _DATASET_CATALOGUE

# ---- Load model once ----
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading model...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir="/home/caches", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", cache_dir="/home/caches", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
print("Model loaded.\n")

# ---- Load GSM8K ----
runner_a = BenchmarkRunner(pipeline=None, tokenizer=tok, cache_dir="/home/caches")
items = runner_a.load_dataset("gsm8k")
random.seed(42)
random.shuffle(items)
items = items[:5]  # just 5 for diagnosis

# ---- Config A: Single model ----
@torch.no_grad()
def single_model_pipeline(question, **kw):
    prompt = f"Question: {question}\n\nAnswer:"
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")
    out = model.generate(ids, attention_mask=attn, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

# ---- Config B: LatentMAS system ----
from src.system import LatentMASSystem, SystemConfig
from src.agents.configs import AgentConfig, LoRASpec
from src.core.embedding_guard import EmbeddingGuard
from src.core.latent_memory import LatentMemory

# Build system that reuses our model
system = LatentMASSystem.__new__(LatentMASSystem)
system.config = SystemConfig(model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda", cache_dir="/home/caches", dtype="bfloat16", latent_steps=10)
system.device = "cuda"
system.tokenizer = tok
system.model = model
system.guard = EmbeddingGuard(model)
system.memory = LatentMemory(device="cuda")
system._reasoner = None
system._pool = None
system._lora_manager = None
system._pipeline = None
system._initialized = False

noop_spec = LoRASpec(rank=1, alpha=1, dropout=0.0)
system.add_agent(AgentConfig.planner(lora_spec=noop_spec))
system.add_agent(AgentConfig.critic(lora_spec=noop_spec))
system.add_agent(AgentConfig.refiner(lora_spec=noop_spec))
system.add_agent(AgentConfig.judger(lora_spec=noop_spec))

print("="*80)
print("COMPARING CONFIG A (single) vs CONFIG B (LatentMAS) on 5 GSM8K questions")
print("="*80)

for i, item in enumerate(items):
    q = item["question"]
    gold = item["answer"]
    
    print(f"\n{'='*80}")
    print(f"Q{i}: {q[:120]}...")
    print(f"GOLD: {gold}")
    
    # Config A
    resp_a = single_model_pipeline(q)
    ext_a = BenchmarkRunner.extract_answer(resp_a, "gsm8k")
    correct_a = BenchmarkRunner._check_answer(ext_a, gold, "gsm8k")
    
    # Config B
    result_b = system.run(q)
    resp_b = result_b.final_answer
    ext_b = BenchmarkRunner.extract_answer(resp_b, "gsm8k")
    correct_b = BenchmarkRunner._check_answer(ext_b, gold, "gsm8k")
    
    print(f"\n--- Config A (single model) ---")
    print(f"  Response (last 200 chars): ...{resp_a[-200:]}")
    print(f"  Extracted: '{ext_a}'  Correct: {correct_a}")
    
    print(f"\n--- Config B (LatentMAS) ---")
    print(f"  # agent outputs: {len(result_b.agent_outputs)}")
    for ao in result_b.agent_outputs:
        print(f"    [{ao['agent']}] {ao['output_tokens']} tokens, first 100: {ao['output'][:100]}...")
    print(f"  Final answer (last 200 chars): ...{resp_b[-200:]}")
    print(f"  Extracted: '{ext_b}'  Correct: {correct_b}")
    
    if correct_a and not correct_b:
        print(f"  >>> A CORRECT, B WRONG — pipeline is LOSING information")
    elif not correct_a and correct_b:
        print(f"  >>> A WRONG, B CORRECT — pipeline HELPED")
    elif correct_a and correct_b:
        print(f"  >>> Both correct")
    else:
        print(f"  >>> Both wrong")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
