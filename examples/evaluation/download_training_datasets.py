#!/usr/bin/env python3
"""
Download the EXACT Training Dataset for Medical LoRA Evaluation

This script downloads the actual dataset used to train the medical LoRA adapter:
- iimran/Medical-Intelligence-Questions (for medical LoRA)

This ensures evaluation uses the same distribution as training data.

Usage:
    python download_training_datasets.py --hf-token YOUR_TOKEN
    python download_training_datasets.py --samples 100
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class EvaluationQuestion:
    """Standard evaluation question format"""
    id: str
    question: str
    options: Dict[str, str]
    answer: str
    dataset: str
    context: Optional[str] = None
    explanation: Optional[str] = None
    reasoning: Optional[str] = None


def login_huggingface(token: str):
    """Login to HuggingFace Hub"""
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("[✓] Logged in to HuggingFace Hub")


def download_medical_intelligence_questions(max_samples: Optional[int] = None, split: str = "train") -> List[EvaluationQuestion]:
    """
    Download the EXACT dataset used to train the Medical LoRA adapter
    
    Dataset: iimran/Medical-Intelligence-Questions
    This is the training dataset for: iimran/Qwen2.5-3B-R1-MedicalReasoner-lora-adapter
    
    Dataset structure:
    - original_question: The medical question
    - generated_question: Patient-style rephrasing
    - reasoning: Chain-of-thought reasoning
    - predicted_answer: Model's predicted answer
    - ground_truth: Correct answer
    """
    from datasets import load_dataset
    
    print("\n" + "="*60)
    print("📚 Downloading Medical LoRA Training Dataset")
    print("="*60)
    print("Dataset: iimran/Medical-Intelligence-Questions")
    print("Used to train: iimran/Qwen2.5-3B-R1-MedicalReasoner-lora-adapter")
    print("="*60 + "\n")
    
    try:
        dataset = load_dataset("iimran/Medical-Intelligence-Questions", split=split)
        source = "iimran/Medical-Intelligence-Questions"
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return []
    
    print(f"[✓] Source: {source}")
    print(f"[✓] Split: {split}")
    print(f"[✓] Total samples available: {len(dataset)}")
    
    # Show dataset structure
    print(f"\n[INFO] Dataset columns: {dataset.column_names}")
    
    if max_samples:
        # For evaluation, use samples from the END to avoid overlap with early training
        start_idx = max(0, len(dataset) - max_samples)
        dataset = dataset.select(range(start_idx, len(dataset)))
        print(f"[INFO] Selected last {len(dataset)} samples for evaluation")
    
    questions = []
    for i, item in enumerate(dataset):
        try:
            # This dataset has specific structure:
            # - original_question: The clinical question
            # - generated_question: Patient-style rephrasing
            # - reasoning: Chain-of-thought explanation
            # - predicted_answer: Model's answer
            # - ground_truth: Correct answer
            
            # Use original_question as the main question
            question_text = item.get("original_question", "")
            if not question_text:
                question_text = item.get("generated_question", "")
            
            # Ground truth is the correct answer (free-text format)
            ground_truth = item.get("ground_truth", "")
            
            # Create options from the question context
            # This dataset is open-ended, so we create a simplified format
            # where the correct answer is option A
            options = {
                "A": ground_truth if ground_truth else "Correct answer",
            }
            
            # For evaluation, we'll compare the model's answer to ground_truth
            answer = "A"  # Ground truth is always option A in our format
            
            # Get reasoning
            reasoning = item.get("reasoning", "")
            
            # Get model's predicted answer for comparison
            predicted = item.get("predicted_answer", "")
            
            questions.append(EvaluationQuestion(
                id=f"medical_training_{item.get('index', i):04d}",
                question=question_text,
                options=options,
                answer=answer,
                dataset="medical_intelligence_questions",
                explanation=ground_truth,  # Store ground truth as explanation
                reasoning=reasoning,
                context=predicted,  # Store model's prediction in context field
            ))
            
        except Exception as e:
            print(f"[WARN] Skipping item {i}: {e}")
            continue
    
    print(f"\n[✓] Loaded {len(questions)} questions from training dataset")
    return questions


def save_dataset(questions: List[EvaluationQuestion], output_path: Path, dataset_name: str):
    """Save questions to JSON file"""
    output_file = output_path / f"{dataset_name}_evaluation.json"
    
    data = [asdict(q) for q in questions]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[✓] Saved {len(questions)} questions to {output_file}")
    return output_file


def show_sample_questions(questions: List[EvaluationQuestion], n: int = 3):
    """Display sample questions"""
    print("\n" + "="*60)
    print("📝 Sample Questions from Training Dataset")
    print("="*60)
    
    for i, q in enumerate(questions[:n]):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {q.question[:200]}..." if len(q.question) > 200 else f"Q: {q.question}")
        if q.options:
            print(f"Options: {list(q.options.keys())}")
        print(f"Answer: {q.answer}")
        if q.reasoning:
            print(f"Reasoning: {q.reasoning[:100]}..." if len(str(q.reasoning)) > 100 else f"Reasoning: {q.reasoning}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the exact training dataset for medical LoRA evaluation"
    )
    parser.add_argument("--output-dir", type=str, default="./eval_data",
                       help="Output directory for datasets")
    parser.add_argument("--samples", type=int, default=100,
                       help="Maximum samples to download")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (train, test, validation)")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="HuggingFace token")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        login_huggingface(args.hf_token)
    
    # Download the training dataset
    questions = download_medical_intelligence_questions(
        max_samples=args.samples,
        split=args.split
    )
    
    if questions:
        # Show samples
        show_sample_questions(questions)
        
        # Save to file
        save_dataset(questions, output_path, "medical_lora_training")
        
        print("\n" + "="*60)
        print("✅ Training Dataset Downloaded Successfully!")
        print("="*60)
        print(f"📁 File: {output_path}/medical_lora_training_evaluation.json")
        print(f"📊 Total: {len(questions)} questions")
        print("\nTo evaluate with this data:")
        print("  python run_mirage_evaluation.py \\")
        print("    --dataset-source ./eval_data/medical_lora_training_evaluation.json \\")
        print("    --num-questions 50 --skip-bertscore")
        print("="*60)
    else:
        print("\n[ERROR] Failed to download dataset")


if __name__ == "__main__":
    main()
