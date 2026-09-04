#!/usr/bin/env python3
"""
LatentMAS Evaluation with Semantic Router (Latent Collaboration Only)

Evaluates 5 benchmark questions across Medical, Math, and Code domains
using:
1. Semantic Router (embedding similarity + keyword boosting) for domain routing
2. Latent Collaboration (pipeline="true_latent") where intermediate agents
   reason purely in continuous latent space without token generation,
   and only the final Judger agent decodes the final answer.

Improvements v2:
- Self-Consistency Voting (×3) for robustness
- Increased Judger max_tokens (600) to prevent truncation
- Enhanced medical & judger prompts (differential diagnosis framework)
- Adaptive latent steps per domain
- Improved confidence-based router
"""

import os
import sys
import json
import time
import re
from pathlib import Path

# Ensure workspace is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# HuggingFace Token Setup
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as e:
        pass

import torch
from src import LatentMASSystem, AgentConfig, Domain
from src.routing import SemanticRouter


# ─── Configuration ───────────────────────────────────────────────────────────

# Self-consistency: run each question N times and majority-vote
SELF_CONSISTENCY_SAMPLES = 3

# Judger max tokens — increased from 300 to 600 to prevent truncation
JUDGER_MAX_TOKENS = 600

# Adaptive latent steps per domain (more steps for complex reasoning)
DOMAIN_LATENT_STEPS = {
    Domain.MEDICAL: 15,    # Complex clinical reasoning
    Domain.MATH: 12,       # Multi-step computation
    Domain.CODE: 10,       # Pattern recognition
    Domain.REASONING: 12,
    Domain.GENERAL: 8,
}

# Domain to specialized agent pipeline mapping for true_latent mode
DOMAIN_PIPELINES = {
    Domain.MEDICAL: ["Planner", "MedicalExpert", "Critic", "Judger"],
    Domain.MATH: ["Planner", "MathExpert", "Critic", "Judger"],
    Domain.CODE: ["Planner", "CodeExpert", "Critic", "Judger"],
    Domain.REASONING: ["Planner", "Critic", "Refiner", "Judger"],
    Domain.GENERAL: ["Planner", "Critic", "Refiner", "Judger"],
}


def extract_answer(text: str) -> str:
    """Extract answer choice (A, B, C, D) from generated text"""
    if not text:
        return "UNKNOWN"
    
    # 0. Check for self-consistency vote header
    sc_match = re.match(r'\[Self-Consistency Vote:\s*([A-D])\]', text)
    if sc_match:
        return sc_match.group(1).upper()
    
    # 1. Look for \boxed{X}
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        answer = boxed[-1].strip().upper()
        match = re.search(r'([ABCD])', answer)
        if match:
            return match.group(1)
        if answer in ['A', 'B', 'C', 'D']:
            return answer

    # 2. Look for explicit patterns like 'The correct answer is (B)' or 'Answer: C'
    patterns = [
        r'(?:final\s+)?answer\s*(?:is|:)\s*[\*\_]*\s*\(?([ABCD])\)?',
        r'(?:correct\s+)?option\s*(?:is|:)\s*[\*\_]*\s*\(?([ABCD])\)?',
        r'(?:choice\s+)?([ABCD])\s*(?:is\s+correct|is\s+the\s+answer)',
        r'^\s*([ABCD])[\.)\:\s]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
            
    # 3. Fallback: search for stand-alone letter near end of text
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1].upper()
        letters = re.findall(r'\b([ABCD])\b', last_line)
        if letters:
            return letters[-1]
            
    return "UNKNOWN"


def load_dataset(data_path: str):
    """Load the 5 evaluation questions"""
    if Path(data_path).exists():
        with open(data_path, "r") as f:
            return json.load(f)
    
    # Fallback to predefined 5 benchmark questions
    return [
        {
            "id": 1,
            "domain": "medical",
            "question": "A 34-year-old man comes to the physician because of a 3-week history of colicky abdominal pain and diarrhea. He has bowel movements 10–12 times daily; the stool contains blood and mucus. Colonoscopy shows a bleeding, ulcerated rectal mucosa with several pseudopolyps. Which of the following is this patient at greatest risk of developing?\nA. Hemolytic uremic syndrome\nB. Oral ulcers\nC. Colorectal cancer\nD. Pancreatic cancer",
            "gold_letter": "C"
        },
        {
            "id": 2,
            "domain": "medical",
            "question": "A 64-year-old man with asthma on high-dose fluticasone inhaler presents with sore mouth for 1 week. Exam shows white patches on oral mucosa. What is the most appropriate next step in management?\nA. Fluconazole\nB. Isotretinoin\nC. Nystatin\nD. Penicillin V",
            "gold_letter": "C"
        },
        {
            "id": 3,
            "domain": "medical",
            "question": "A 45-year-old mechanic presents with acute-onset shortness of breath while repairing a tractor. He is pale, diaphoretic with contracted pupils. Diffuse wheezes are noted. What is the best treatment?\nA. Succinylcholine\nB. Inhaled ipratropium and oxygen\nC. Atropine and pralidoxime\nD. Inhaled albuterol and oxygen",
            "gold_letter": "C"
        },
        {
            "id": 4,
            "domain": "math",
            "question": "What is the result of 15 × 7 + 23?\nA. 105\nB. 128\nC. 135\nD. 142",
            "gold_letter": "B"
        },
        {
            "id": 5,
            "domain": "code",
            "question": "Which data structure uses LIFO (Last In First Out) principle?\nA. Queue\nB. Stack\nC. Array\nD. Linked List",
            "gold_letter": "B"
        }
    ]


def main():
    print("=" * 80)
    print("⚡ LatentMAS EVALUATION v2 - IMPROVED LATENT COLLABORATION")
    print("=" * 80)
    print("Mode: TRUE LATENT COLLABORATION (pipeline='true_latent')")
    print("      Intermediate agents communicate strictly in continuous latent space.")
    print("      Only the final Judger agent generates text.")
    print(f"      + Self-Consistency Voting (×{SELF_CONSISTENCY_SAMPLES})")
    print(f"      + Judger max_tokens: {JUDGER_MAX_TOKENS}")
    print(f"      + Enhanced medical & judger prompts")
    print(f"      + Adaptive latent steps per domain")
    print("Router: Semantic Router (Embeddings + Keyword Boosting + Confidence Thresholding)")
    print("=" * 80)
    
    # 1. Initialize Semantic Router
    print("\n[1/3] Initializing Semantic Router...")
    router_start = time.time()
    router = SemanticRouter()
    print(f"      ✓ Semantic router ready ({time.time() - router_start:.2f}s)")
    
    # 2. Initialize LatentMAS System with all domain agents
    print("\n[2/3] Initializing LatentMAS Base System & LoRA Adapters...")
    system_start = time.time()
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        latent_steps=10,  # Default; overridden per domain below
    )
    
    # Register agents needed for all domains
    # Intermediate agents: low max_tokens (latent only, no text generated)
    # Judger: high max_tokens (generates final text answer)
    agents_to_register = [
        AgentConfig.planner(max_tokens=50),
        AgentConfig.medical(max_tokens=50),
        AgentConfig.math(max_tokens=50),
        AgentConfig.coder(max_tokens=50),
        AgentConfig.critic(max_tokens=50),
        AgentConfig.refiner(max_tokens=50),
        AgentConfig.judger(max_tokens=JUDGER_MAX_TOKENS),
    ]
    
    for agent in agents_to_register:
        system.add_agent(agent)
        
    print(f"      ✓ System ready with {len(system._pool.list_agents())} agents in {time.time() - system_start:.1f}s")
    print(f"      Registered: {', '.join(system._pool.list_agents())}")
    
    # 3. Load Questions
    data_path = SCRIPT_DIR / "data" / "sample_data.json"
    questions = load_dataset(str(data_path))
    print(f"\n[3/3] Loaded {len(questions)} evaluation questions from {data_path.name}")
    
    results = []
    correct_count = 0
    total_tokens = 0
    total_latency_ms = 0
    
    # Execute Evaluation
    print("\n" + "=" * 80)
    print(f"🚀 STARTING {len(questions)}-QUESTION LATENT COLLABORATION RUN (SC×{SELF_CONSISTENCY_SAMPLES})")
    print("=" * 80)
    
    for idx, item in enumerate(questions):
        q_num = idx + 1
        question_text = item["question"]
        gold = item.get("gold_letter", item.get("answer", "")).strip().upper()
        if len(gold) > 1:
            gold_match = re.search(r'([ABCD])', gold)
            gold = gold_match.group(1) if gold_match else gold
            
        print(f"\n{'─' * 80}")
        print(f"📝 Question {q_num}/{len(questions)}:")
        first_line = question_text.split('\n')[0]
        print(f"   Prompt: {first_line[:100]}...")
        
        # Step A: Semantic Routing
        domain, confidence = router.get_best_domain(question_text)
        pipeline_agents = DOMAIN_PIPELINES.get(domain, DOMAIN_PIPELINES[Domain.GENERAL])
        latent_steps = DOMAIN_LATENT_STEPS.get(domain, 10)
        
        print(f"   🧭 Semantic Router -> Domain: {domain.value.upper()} (Confidence: {confidence:.1%})")
        print(f"   👥 Latent Pipeline : {' -> '.join(pipeline_agents)}")
        print(f"   🔄 Latent Mode     : true_latent (Zero intermediate text tokens)")
        print(f"   🧠 Latent Steps    : {latent_steps} (domain-adaptive)")
        print(f"   🗳️  Self-Consistency: ×{SELF_CONSISTENCY_SAMPLES}")
        
        # Step B: Set adaptive latent steps for this domain
        system._pipeline.latent_steps = latent_steps
        
        # Step C: Latent Collaboration Run with Self-Consistency
        q_start = time.time()
        try:
            res = system.run(
                question=question_text,
                pipeline="true_latent",
                agents=pipeline_agents,
                max_new_tokens=JUDGER_MAX_TOKENS,
                temperature=0.4,
                self_consistency=SELF_CONSISTENCY_SAMPLES,
            )
            q_latency_ms = int((time.time() - q_start) * 1000)
            
            # Step D: Answer extraction and scoring
            predicted = extract_answer(res.final_answer)
            
            # Also check self_consistency metadata for voted answer
            sc_meta = res.metadata.get("self_consistency", {})
            if sc_meta:
                voted_answer = sc_meta.get("final_answer", predicted)
                all_answers = sc_meta.get("all_answers", [])
                vote_counts = sc_meta.get("vote_counts", {})
                predicted = voted_answer
                print(f"   🗳️  SC Votes       : {all_answers} -> Winner: {voted_answer} ({vote_counts})")
            
            is_correct = (predicted == gold)
            if is_correct:
                correct_count += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
                
            total_tokens += res.total_tokens
            total_latency_ms += q_latency_ms
            
            print(f"   ⏱️ Latency         : {q_latency_ms / 1000:.2f}s")
            print(f"   🔢 Tokens Used     : {res.total_tokens} (Latent steps: {res.latent_steps_total})")
            print(f"   🎯 Prediction      : {predicted} | Expected: {gold} | {status}")
            
            # Show a snippet of the Judger's output (skip SC header)
            answer_text = res.final_answer
            if answer_text.startswith("[Self-Consistency"):
                # Skip header, show actual reasoning
                parts = answer_text.split("\n\n", 1)
                if len(parts) > 1:
                    answer_text = parts[1]
            print(f"   💡 Judger Output   : {answer_text.strip().replace(chr(10), ' ')[:200]}...")
            
            results.append({
                "id": q_num,
                "domain": domain.value,
                "confidence": round(confidence, 3),
                "pipeline": pipeline_agents,
                "mode": "true_latent",
                "latent_steps": latent_steps,
                "self_consistency": SELF_CONSISTENCY_SAMPLES,
                "sc_votes": sc_meta.get("all_answers", []),
                "sc_vote_counts": sc_meta.get("vote_counts", {}),
                "predicted": predicted,
                "gold": gold,
                "correct": is_correct,
                "tokens": res.total_tokens,
                "latency_ms": q_latency_ms,
                "final_answer": res.final_answer
            })
            
        except Exception as e:
            print(f"   ❌ Error executing question {q_num}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "id": q_num,
                "error": str(e)
            })
            
    # Final Summary Table
    accuracy = (correct_count / len(questions)) * 100 if questions else 0.0
    avg_latency = (total_latency_ms / len(questions)) / 1000 if questions else 0.0
    
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS SUMMARY (LATENT COLLABORATION v2)")
    print("=" * 80)
    print(f"{'#':<3} {'Domain':<10} {'Conf':<8} {'SC Votes':<20} {'Pred':<6} {'Gold':<6} {'Status':<10} {'Latency':<10} {'Tokens':<8}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['id']:<3} ERROR")
            continue
        stat = "PASS" if r['correct'] else "FAIL"
        sc_info = ','.join(r.get('sc_votes', []))
        print(f"{r['id']:<3} {r['domain']:<10} {r['confidence']*100:>5.1f}%  {sc_info:<20} {r['predicted']:<6} {r['gold']:<6} {stat:<10} {r['latency_ms']/1000:>6.2f}s    {r['tokens']:<8}")
    print("-" * 100)
    print(f"Accuracy         : {accuracy:.1f}% ({correct_count}/{len(questions)})")
    print(f"Average Latency  : {avg_latency:.2f}s per question")
    print(f"Total Tokens     : {total_tokens}")
    print(f"SC Samples       : {SELF_CONSISTENCY_SAMPLES}x per question")
    print("=" * 80)
    
    # Comparison with v1
    print("\n📈 Comparison with v1 (baseline):")
    print(f"   v1: 60.0% (3/5) @ 15.74s avg, 4294 tokens")
    print(f"   v2: {accuracy:.1f}% ({correct_count}/{len(questions)}) @ {avg_latency:.2f}s avg, {total_tokens} tokens")
    if accuracy > 60:
        print(f"   Δ : +{accuracy - 60:.1f}% accuracy improvement 🎉")
    elif accuracy == 60:
        print(f"   Δ : Same accuracy (try increasing SC samples)")
    else:
        print(f"   Δ : -{60 - accuracy:.1f}% (investigate)")
    
    # Save output to json
    output_file = SCRIPT_DIR / "results_latent_eval_v2.json"
    with open(output_file, "w") as f:
        json.dump({
            "version": "v2",
            "improvements": [
                f"self_consistency_x{SELF_CONSISTENCY_SAMPLES}",
                f"judger_max_tokens_{JUDGER_MAX_TOKENS}",
                "enhanced_medical_prompts",
                "enhanced_judger_prompts",
                "adaptive_latent_steps",
                "router_confidence_thresholding",
            ],
            "mode": "true_latent",
            "router": "SemanticRouter_v2",
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(questions),
            "avg_latency_s": round(avg_latency, 2),
            "total_tokens": total_tokens,
            "results": results
        }, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
