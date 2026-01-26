#!/usr/bin/env python3
"""
Medical QA Evaluation with LatentMAS

Demonstrates:
1. Loading external medical LoRA
2. Running medical reasoning pipeline
3. Evaluation with accuracy metrics
"""

import sys
import json
import re
from pathlib import Path
sys.path.insert(0, '/workspace/latent_mas_slora')

from src import LatentMASSystem, AgentConfig, AgentRole
from src.agents.configs import LoRASpec


def load_medqa_data(path: str, max_samples: int = 10):
    """Load MedQA dataset"""
    with open(path) as f:
        data = json.load(f)
    return data[:max_samples] if max_samples > 0 else data


def extract_answer(text: str) -> str:
    """Extract answer choice from generated text"""
    # Look for \boxed{X}
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        answer = boxed[-1].strip().upper()
        match = re.search(r'([ABCD])', answer)
        return match.group(1) if match else answer
    
    # Fallback patterns
    patterns = [
        r'(?:final\s+)?answer\s*:?\s*([ABCD])',
        r'(?:correct\s+)?option\s*:?\s*([ABCD])',
        r'^([ABCD])[.)\s]',
    ]
    
    text_upper = text.upper()
    for pattern in patterns:
        match = re.search(pattern, text_upper, re.MULTILINE)
        if match:
            return match.group(1)
    
    return "UNKNOWN"


def extract_gold_choice(question: str, gold_answer: str) -> str:
    """Extract gold answer letter"""
    gold_clean = gold_answer.strip().upper()
    
    if gold_clean in ['A', 'B', 'C', 'D']:
        return gold_clean
    
    # Match to options
    for line in question.split('\n'):
        match = re.match(r'^([ABCD])[\.\):\s]+(.+)$', line.strip())
        if match:
            letter, text = match.groups()
            if gold_answer.lower() in text.lower() or text.lower() in gold_answer.lower():
                return letter.upper()
    
    return None


def main():
    print("=" * 60)
    print("Medical QA Evaluation with LatentMAS")
    print("=" * 60)
    
    # Initialize system
    print("\n[1] Initializing system...")
    system = LatentMASSystem(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        dtype="bfloat16",
        latent_steps=15,
    )
    
    # Add medical-specialized agents
    print("\n[2] Adding medical reasoning agents...")
    
    # Planner with medical focus
    system.add_agent(AgentConfig(
        name="MedicalPlanner",
        role=AgentRole.PLANNER,
        adapter_name="med_planner_lora",
        lora_spec=LoRASpec(rank=32, alpha=64),
        temperature=0.6,
        max_tokens=350,
        system_prompt=(
            "You are a Medical Planning Agent. Analyze clinical presentations "
            "systematically: identify key symptoms, relevant history, and clinical signs. "
            "Create a diagnostic reasoning framework."
        ),
    ))
    
    # Medical domain expert
    system.add_agent(AgentConfig.medical())
    
    # Medical critic
    system.add_agent(AgentConfig(
        name="MedicalCritic",
        role=AgentRole.CRITIC,
        adapter_name="med_critic_lora",
        lora_spec=LoRASpec(rank=32, alpha=64),
        temperature=0.4,
        max_tokens=300,
        system_prompt=(
            "You are a Medical Critic Agent. Evaluate differential diagnoses, "
            "check for missing considerations, and validate clinical reasoning. "
            "Consider common pitfalls and rare presentations."
        ),
    ))
    
    # Final judger
    system.add_agent(AgentConfig.judger())
    
    print(f"    Agents: {system._pool.list_agents()}")
    
    # Try to load external medical LoRA (optional)
    print("\n[3] Attempting to load external medical LoRA...")
    try:
        success = system.load_external_lora(
            name="medical_ext",
            hf_path="iimran/Qwen2.5-3B-R1-MedicalReasoner-lora-adapter",
        )
        if success:
            print("    ✓ External medical LoRA loaded!")
    except Exception as e:
        print(f"    ✗ Could not load external LoRA: {e}")
        print("    (Continuing with built-in adapters)")
    
    # Load data
    print("\n[4] Loading MedQA data...")
    data_path = "/workspace/LatentMAS/data/medqa.json"
    
    if not Path(data_path).exists():
        print(f"    ✗ Data not found at {data_path}")
        print("    Creating sample questions...")
        data = [
            {
                "query": """A 34-year-old man has colicky abdominal pain, bloody diarrhea, and pseudopolyps on colonoscopy. What is he at greatest risk of developing?
A. Hemolytic uremic syndrome
B. Oral ulcers
C. Colorectal cancer
D. Pancreatic cancer""",
                "answer": "Colorectal cancer",
            },
            {
                "query": """A 64-year-old man with asthma on high-dose fluticasone inhaler presents with sore mouth. Exam shows white patches on oral mucosa. What is the best treatment?
A. Fluconazole
B. Isotretinoin
C. Nystatin
D. Penicillin V""",
                "answer": "Nystatin",
            },
        ]
    else:
        data = load_medqa_data(data_path, max_samples=5)
    
    print(f"    Loaded {len(data)} samples")
    
    # Run evaluation
    print("\n[5] Running evaluation...")
    results = []
    correct = 0
    
    for idx, item in enumerate(data):
        question = item.get("query", item.get("question", ""))
        gold_answer = item.get("answer", "")
        
        print(f"\n--- Sample {idx + 1}/{len(data)} ---")
        print(f"Q: {question[:100]}...")
        
        result = system.run(
            question=question,
            pipeline="hierarchical",
            agents=["MedicalPlanner", "MedicalExpert", "MedicalCritic", "Judger"],
            max_new_tokens=400,
            temperature=0.5,
        )
        
        prediction = extract_answer(result.final_answer)
        gold_choice = extract_gold_choice(question, gold_answer)
        
        is_correct = prediction == gold_choice and prediction != "UNKNOWN"
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} Predicted: {prediction}, Gold: {gold_choice}")
        
        results.append({
            "idx": idx,
            "prediction": prediction,
            "gold": gold_choice,
            "correct": is_correct,
            "tokens": result.total_tokens,
            "latency_ms": result.total_latency_ms,
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    accuracy = correct / len(data) if data else 0
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(data)})")
    print(f"  Total tokens: {sum(r['tokens'] for r in results)}")
    print(f"  Avg latency: {sum(r['latency_ms'] for r in results) / len(results):.0f}ms")
    
    # Save results
    output_path = "/workspace/latent_mas_slora/results_medical.json"
    with open(output_path, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
