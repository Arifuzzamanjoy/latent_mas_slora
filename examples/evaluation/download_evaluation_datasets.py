#!/usr/bin/env python3
"""
Download and Prepare Evaluation Datasets from HuggingFace

Downloads medical QA datasets commonly used for training LoRA adapters,
preparing them for evaluation. Uses the same datasets that were used
to train the medical reasoning agents.

Key datasets:
- MedQA (USMLE-style questions)
- MedMCQA (Medical MCQs from India)
- PubMedQA (PubMed research questions)
- Medical Meadow (MedAlpaca training data)

Usage:
    python download_evaluation_datasets.py --output-dir ./eval_data
    python download_evaluation_datasets.py --dataset medqa --samples 100
    python download_evaluation_datasets.py --all --samples 200
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
    metamap_phrases: Optional[List[str]] = None


def login_huggingface(token: str):
    """Login to HuggingFace Hub"""
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("[✓] Logged in to HuggingFace Hub")


def download_medqa_usmle(max_samples: Optional[int] = None) -> List[EvaluationQuestion]:
    """
    Download MedQA USMLE dataset - primary dataset for medical reasoning
    
    This is the main dataset used for training medical LoRA adapters.
    Contains USMLE-style multiple choice questions.
    
    Dataset: bigbio/med_qa or openlifescienceai/medqa
    """
    from datasets import load_dataset
    
    print("\n[MedQA] Downloading USMLE-style medical questions...")
    
    try:
        # Try the openlifescienceai version first (cleaner format)
        dataset = load_dataset("openlifescienceai/medqa", split="test")
        source = "openlifescienceai/medqa"
    except Exception as e:
        print(f"[MedQA] Trying alternative source...")
        try:
            # Fallback to bigbio version
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="test")
            source = "bigbio/med_qa"
        except Exception as e2:
            print(f"[MedQA] Error loading dataset: {e2}")
            return []
    
    print(f"[MedQA] Source: {source}")
    print(f"[MedQA] Total samples available: {len(dataset)}")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    questions = []
    for i, item in enumerate(dataset):
        try:
            # Handle openlifescienceai/medqa format with nested 'data' field
            if "data" in item and isinstance(item["data"], dict):
                data = item["data"]
                options = data.get("Options", {})
                question = data.get("Question", "")
                answer = data.get("Correct Option", "A")
                correct_answer = data.get("Correct Answer", "")
            # Handle other dataset formats
            elif "options" in item or "Options" in item:
                options = item.get("options", item.get("Options", {}))
                if isinstance(options, list):
                    options = {chr(65 + j): opt for j, opt in enumerate(options)}
                question = item.get("question", item.get("Question", ""))
                answer = item.get("answer", item.get("Correct Option", "A"))
                correct_answer = None
            else:
                continue
            
            # Normalize answer to letter
            if isinstance(answer, int):
                answer = chr(65 + answer)
            elif isinstance(answer, dict):
                answer = answer.get("key", "A")
            
            questions.append(EvaluationQuestion(
                id=f"medqa_{i:04d}",
                question=question,
                options=options,
                answer=str(answer).upper(),
                dataset="medqa",
                context=item.get("context", None),
                explanation=correct_answer if correct_answer else None,
            ))
        except Exception as e:
            print(f"[MedQA] Skipping item {i}: {e}")
            continue
    
    print(f"[MedQA] Loaded {len(questions)} questions")
    return questions


def download_medmcqa(max_samples: Optional[int] = None) -> List[EvaluationQuestion]:
    """
    Download MedMCQA dataset - Indian medical entrance exam questions
    
    Contains MCQs covering various medical subjects.
    Dataset: openlifescienceai/medmcqa
    """
    from datasets import load_dataset
    
    print("\n[MedMCQA] Downloading medical MCQ questions...")
    
    try:
        dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
        source = "openlifescienceai/medmcqa"
    except Exception:
        try:
            dataset = load_dataset("medmcqa", split="validation")
            source = "medmcqa"
        except Exception as e:
            print(f"[MedMCQA] Error loading dataset: {e}")
            return []
    
    print(f"[MedMCQA] Source: {source}")
    print(f"[MedMCQA] Total samples available: {len(dataset)}")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    questions = []
    for i, item in enumerate(dataset):
        try:
            options = {
                "A": item.get("opa", ""),
                "B": item.get("opb", ""),
                "C": item.get("opc", ""),
                "D": item.get("opd", ""),
            }
            
            # cop is 0-indexed answer
            answer_idx = item.get("cop", 0)
            answer = chr(65 + answer_idx)
            
            questions.append(EvaluationQuestion(
                id=f"medmcqa_{i:04d}",
                question=item.get("question", ""),
                options=options,
                answer=answer,
                dataset="medmcqa",
                explanation=item.get("exp", None),
            ))
        except Exception as e:
            print(f"[MedMCQA] Skipping item {i}: {e}")
            continue
    
    print(f"[MedMCQA] Loaded {len(questions)} questions")
    return questions


def download_pubmedqa(max_samples: Optional[int] = None) -> List[EvaluationQuestion]:
    """
    Download PubMedQA dataset - yes/no/maybe questions from research abstracts
    
    Dataset: bigbio/pubmed_qa or pubmed_qa
    """
    from datasets import load_dataset
    
    print("\n[PubMedQA] Downloading PubMed research questions...")
    
    try:
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        source = "qiaojin/PubMedQA"
    except Exception:
        try:
            dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
            source = "pubmed_qa"
        except Exception as e:
            print(f"[PubMedQA] Error loading dataset: {e}")
            return []
    
    print(f"[PubMedQA] Source: {source}")
    print(f"[PubMedQA] Total samples available: {len(dataset)}")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    questions = []
    for i, item in enumerate(dataset):
        try:
            # PubMedQA has yes/no/maybe format
            options = {
                "A": "Yes",
                "B": "No", 
                "C": "Maybe"
            }
            
            answer_map = {"yes": "A", "no": "B", "maybe": "C"}
            answer = answer_map.get(item.get("final_decision", "").lower(), "C")
            
            # Include context from abstract
            context = item.get("context", {})
            if isinstance(context, dict):
                context = " ".join(context.get("contexts", []))
            
            questions.append(EvaluationQuestion(
                id=f"pubmedqa_{i:04d}",
                question=item.get("question", ""),
                options=options,
                answer=answer,
                dataset="pubmedqa",
                context=context,
                explanation=item.get("long_answer", None),
            ))
        except Exception as e:
            print(f"[PubMedQA] Skipping item {i}: {e}")
            continue
    
    print(f"[PubMedQA] Loaded {len(questions)} questions")
    return questions


def download_medical_meadow(max_samples: Optional[int] = None) -> List[EvaluationQuestion]:
    """
    Download Medical Meadow MedQA dataset
    
    This is the exact dataset used by MedAlpaca for LoRA training!
    Contains instruction-formatted medical QA pairs.
    
    Dataset: medalpaca/medical_meadow_medqa
    """
    from datasets import load_dataset
    
    print("\n[Medical Meadow] Downloading MedAlpaca training data...")
    
    try:
        dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train")
        source = "medalpaca/medical_meadow_medqa"
    except Exception as e:
        print(f"[Medical Meadow] Error loading dataset: {e}")
        return []
    
    print(f"[Medical Meadow] Source: {source}")
    print(f"[Medical Meadow] Total samples available: {len(dataset)}")
    
    if max_samples:
        # Sample from the end for evaluation (avoid overlap with training)
        start_idx = max(0, len(dataset) - max_samples)
        dataset = dataset.select(range(start_idx, len(dataset)))
    
    questions = []
    for i, item in enumerate(dataset):
        try:
            instruction = item.get("instruction", item.get("input", ""))
            output = item.get("output", "")
            
            # Parse MCQ format from instruction if present
            options = {}
            question_text = instruction
            
            # Try to extract options from instruction
            import re
            option_pattern = r'([A-D])[\.\)]\s*([^\n]+)'
            matches = re.findall(option_pattern, instruction)
            if matches:
                for letter, text in matches:
                    options[letter] = text.strip()
                # Extract question part (before options)
                question_text = re.split(r'[A-D][\.\)]', instruction)[0].strip()
            
            # Try to extract answer from output
            answer = "A"  # default
            answer_match = re.search(r'([A-D])', output)
            if answer_match:
                answer = answer_match.group(1)
            
            if not options:
                # If no options found, create from output
                options = {"A": output[:100] + "..." if len(output) > 100 else output}
            
            questions.append(EvaluationQuestion(
                id=f"meadow_{i:04d}",
                question=question_text,
                options=options,
                answer=answer,
                dataset="medical_meadow",
                explanation=output,
            ))
        except Exception as e:
            print(f"[Medical Meadow] Skipping item {i}: {e}")
            continue
    
    print(f"[Medical Meadow] Loaded {len(questions)} questions")
    return questions


def download_mmlu_medical(max_samples: Optional[int] = None) -> List[EvaluationQuestion]:
    """
    Download MMLU Medical subset - academic benchmark questions
    
    Includes: anatomy, clinical knowledge, college medicine, 
    medical genetics, professional medicine, college biology
    
    Dataset: cais/mmlu
    """
    from datasets import load_dataset
    
    print("\n[MMLU Medical] Downloading medical subset questions...")
    
    medical_subjects = [
        "anatomy",
        "clinical_knowledge", 
        "college_medicine",
        "medical_genetics",
        "professional_medicine",
        "college_biology",
    ]
    
    all_questions = []
    
    for subject in medical_subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            
            samples_per_subject = max_samples // len(medical_subjects) if max_samples else None
            if samples_per_subject:
                dataset = dataset.select(range(min(samples_per_subject, len(dataset))))
            
            for i, item in enumerate(dataset):
                options = {chr(65 + j): opt for j, opt in enumerate(item["choices"])}
                answer = chr(65 + item["answer"])
                
                all_questions.append(EvaluationQuestion(
                    id=f"mmlu_{subject}_{i:04d}",
                    question=item["question"],
                    options=options,
                    answer=answer,
                    dataset=f"mmlu_{subject}",
                ))
        except Exception as e:
            print(f"[MMLU] Error loading {subject}: {e}")
            continue
    
    print(f"[MMLU Medical] Loaded {len(all_questions)} questions")
    return all_questions


def save_dataset(questions: List[EvaluationQuestion], output_path: Path, dataset_name: str):
    """Save questions to JSON file"""
    output_file = output_path / f"{dataset_name}_evaluation.json"
    
    data = [asdict(q) for q in questions]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[✓] Saved {len(questions)} questions to {output_file}")
    return output_file


def create_combined_dataset(all_questions: Dict[str, List[EvaluationQuestion]], output_path: Path):
    """Create a combined MIRAGE-style evaluation dataset"""
    combined = []
    
    for dataset_name, questions in all_questions.items():
        combined.extend(questions)
    
    # Shuffle for mixed evaluation
    import random
    random.shuffle(combined)
    
    output_file = output_path / "combined_medical_evaluation.json"
    data = [asdict(q) for q in combined]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[✓] Created combined dataset with {len(combined)} questions at {output_file}")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    for name, qs in all_questions.items():
        print(f"   - {name}: {len(qs)} questions")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download medical evaluation datasets from HuggingFace")
    parser.add_argument("--output-dir", type=str, default="./eval_data",
                       help="Output directory for datasets")
    parser.add_argument("--dataset", type=str, choices=["medqa", "medmcqa", "pubmedqa", "meadow", "mmlu", "all"],
                       default="all", help="Which dataset to download")
    parser.add_argument("--samples", type=int, default=100,
                       help="Maximum samples per dataset")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="HuggingFace token (optional, for private datasets)")
    parser.add_argument("--combined-only", action="store_true",
                       help="Only create the combined dataset file")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        login_huggingface(args.hf_token)
    
    print("=" * 60)
    print("📚 Medical Evaluation Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Max samples per dataset: {args.samples}")
    
    all_questions = {}
    
    # Download requested datasets
    if args.dataset in ["medqa", "all"]:
        questions = download_medqa_usmle(args.samples)
        if questions:
            all_questions["medqa"] = questions
            if not args.combined_only:
                save_dataset(questions, output_path, "medqa")
    
    if args.dataset in ["medmcqa", "all"]:
        questions = download_medmcqa(args.samples)
        if questions:
            all_questions["medmcqa"] = questions
            if not args.combined_only:
                save_dataset(questions, output_path, "medmcqa")
    
    if args.dataset in ["pubmedqa", "all"]:
        questions = download_pubmedqa(args.samples)
        if questions:
            all_questions["pubmedqa"] = questions
            if not args.combined_only:
                save_dataset(questions, output_path, "pubmedqa")
    
    if args.dataset in ["meadow", "all"]:
        questions = download_medical_meadow(args.samples)
        if questions:
            all_questions["medical_meadow"] = questions
            if not args.combined_only:
                save_dataset(questions, output_path, "medical_meadow")
    
    if args.dataset in ["mmlu", "all"]:
        questions = download_mmlu_medical(args.samples)
        if questions:
            all_questions["mmlu_medical"] = questions
            if not args.combined_only:
                save_dataset(questions, output_path, "mmlu_medical")
    
    # Create combined dataset
    if len(all_questions) > 0:
        create_combined_dataset(all_questions, output_path)
    
    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
