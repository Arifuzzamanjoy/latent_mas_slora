#!/usr/bin/env python3
"""
Evaluation Metrics for RAG System Comparison

Implements two popular evaluation methods:
1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence

2. BERTScore (Semantic Similarity)
   - Precision, Recall, F1 based on BERT embeddings
   - Better captures semantic meaning than ROUGE

Additional metrics:
- Accuracy (for multiple choice questions)
- Answer extraction accuracy
- Latency metrics
- Token efficiency
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class RougeScores:
    """ROUGE evaluation scores"""
    rouge1_precision: float
    rouge1_recall: float
    rouge1_f1: float
    rouge2_precision: float
    rouge2_recall: float
    rouge2_f1: float
    rougeL_precision: float
    rougeL_recall: float
    rougeL_f1: float


@dataclass
class BERTScoreResult:
    """BERTScore evaluation result"""
    precision: float
    recall: float
    f1: float
    

@dataclass
class EvaluationResult:
    """Complete evaluation result for a single question"""
    question_id: str
    question: str
    reference_answer: str
    predicted_answer: str
    is_correct: bool  # For MCQ accuracy
    extracted_choice: Optional[str]
    rouge_scores: Optional[RougeScores]
    bert_score: Optional[BERTScoreResult]
    latency_ms: int
    total_tokens: int


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all questions"""
    total_questions: int
    accuracy: float  # MCQ accuracy
    avg_rouge1_f1: float
    avg_rouge2_f1: float
    avg_rougeL_f1: float
    avg_bert_f1: float
    avg_latency_ms: float
    avg_tokens: float
    total_tokens: int
    questions_per_second: float
    by_dataset: Dict[str, Dict[str, float]]


class RougeEvaluator:
    """
    ROUGE Score Calculator
    
    ROUGE-1: Measures unigram (single word) overlap
    ROUGE-2: Measures bigram (2-word sequence) overlap
    ROUGE-L: Measures longest common subsequence
    """
    
    def __init__(self):
        self._rouge = None
    
    def _lazy_init(self):
        if self._rouge is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=True
                )
            except ImportError:
                print("[ROUGE] Installing rouge-score...")
                import subprocess
                subprocess.run(["pip", "install", "rouge-score"], check=True)
                from rouge_score import rouge_scorer
                self._rouge = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=True
                )
    
    def score(self, reference: str, prediction: str) -> RougeScores:
        """Calculate ROUGE scores between reference and prediction"""
        self._lazy_init()
        
        # Clean inputs
        reference = reference.strip().lower()
        prediction = prediction.strip().lower()
        
        if not reference or not prediction:
            return RougeScores(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        scores = self._rouge.score(reference, prediction)
        
        return RougeScores(
            rouge1_precision=scores['rouge1'].precision,
            rouge1_recall=scores['rouge1'].recall,
            rouge1_f1=scores['rouge1'].fmeasure,
            rouge2_precision=scores['rouge2'].precision,
            rouge2_recall=scores['rouge2'].recall,
            rouge2_f1=scores['rouge2'].fmeasure,
            rougeL_precision=scores['rougeL'].precision,
            rougeL_recall=scores['rougeL'].recall,
            rougeL_f1=scores['rougeL'].fmeasure,
        )
    
    def score_batch(
        self,
        references: List[str],
        predictions: List[str]
    ) -> List[RougeScores]:
        """Score multiple reference-prediction pairs"""
        return [
            self.score(ref, pred)
            for ref, pred in zip(references, predictions)
        ]


class BERTScoreEvaluator:
    """
    BERTScore Calculator
    
    Uses BERT embeddings to compute semantic similarity.
    Better at capturing meaning than surface-level ROUGE.
    """
    
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        self.model_type = model_type
        self._initialized = False
    
    def _lazy_init(self):
        if not self._initialized:
            try:
                import bert_score
                self._initialized = True
            except ImportError:
                print("[BERTScore] Installing bert-score...")
                import subprocess
                subprocess.run(["pip", "install", "bert-score"], check=True)
                self._initialized = True
    
    def score(self, reference: str, prediction: str) -> BERTScoreResult:
        """Calculate BERTScore for a single pair"""
        return self.score_batch([reference], [prediction])[0]
    
    def score_batch(
        self,
        references: List[str],
        predictions: List[str],
    ) -> List[BERTScoreResult]:
        """Calculate BERTScore for multiple pairs"""
        self._lazy_init()
        
        import bert_score
        
        # Clean inputs
        refs = [r.strip() for r in references]
        preds = [p.strip() for p in predictions]
        
        # Handle empty strings
        valid_pairs = []
        results = []
        
        for i, (r, p) in enumerate(zip(refs, preds)):
            if r and p:
                valid_pairs.append((i, r, p))
            else:
                results.append((i, BERTScoreResult(0, 0, 0)))
        
        if valid_pairs:
            indices, valid_refs, valid_preds = zip(*valid_pairs)
            
            P, R, F1 = bert_score.score(
                list(valid_preds),
                list(valid_refs),
                model_type=self.model_type,
                verbose=False,
            )
            
            for i, idx in enumerate(indices):
                results.append((
                    idx,
                    BERTScoreResult(
                        precision=P[i].item(),
                        recall=R[i].item(),
                        f1=F1[i].item(),
                    )
                ))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


class AnswerExtractor:
    """Extract multiple choice answers from generated text"""
    
    # Patterns to match answer choices (order matters - more specific first)
    PATTERNS = [
        r"\\boxed\{([A-D])\}",  # LaTeX boxed format (e.g., \boxed{B})
        r"(?:the answer is|answer is)[:\s]*\*?\*?([A-D])",  # "The answer is A" or "The answer is **A**"
        r"(?:correct answer is)[:\s]*\*?\*?([A-D])",
        r"(?:final answer is)[:\s]*\*?\*?([A-D])",
        r"(?:answer|choice|option)[:\s]*\*?\*?([A-D])",
        r"\*\*([A-D])[:\.\)]\*?\*?",  # Markdown bold: **A:** or **A.**
        r"^([A-D])[.:\s]",
        r"\b([A-D])\s*(?:is correct|is the answer)",
        r"\"answer_choice\":\s*\"([A-D])\"",
        r"'answer':\s*'([A-D])'",
        r"(?:would be|should be|is)[:\s]*\*?\*?([A-D])[:\.\)]",  # "would be: A" or "would be **A:**"
        r"(?:cause|diagnosis|treatment|answer)[^.]{0,30}is[:\s]*\n?([A-D]):",  # "cause is:\nD: xxx"
        r"\n([A-D]):\s*[A-Z]",  # Line starting with "D: Something"
    ]
    
    @classmethod
    def extract(cls, text: str, options: Dict[str, str] = None) -> Optional[str]:
        """
        Extract the answer choice (A, B, C, D) from generated text
        
        Returns:
            The extracted choice letter, or None if not found
        """
        text_upper = text.upper()
        
        # Try patterns
        for pattern in cls.PATTERNS:
            match = re.search(pattern, text_upper, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        # Try to find explicit option text matching
        if options:
            for choice, option_text in options.items():
                # Check if the full option text appears prominently
                if option_text.lower() in text.lower():
                    return choice.upper()
        
        # Try to find "X: Option text" format (common in explanations)
        option_line_pattern = r'\n([A-D]):\s*\w+'
        match = re.search(option_line_pattern, text)
        if match:
            return match.group(1).upper()
        
        # Last resort: find any standalone A, B, C, D with colon
        for choice in ['D', 'C', 'B', 'A']:  # Reverse order to catch later mentions
            if f"\n{choice}:" in text or f" {choice}:" in text:
                return choice
        
        # Very last resort: find any standalone A, B, C, D
        for choice in ['A', 'B', 'C', 'D']:
            if f" {choice} " in text_upper or f" {choice}." in text_upper:
                return choice
        
        return None


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation combining ROUGE and BERTScore
    
    Usage:
        evaluator = ComprehensiveEvaluator()
        results = evaluator.evaluate(questions, predictions)
        metrics = evaluator.aggregate(results)
    """
    
    def __init__(self, use_bertscore: bool = True):
        self.rouge = RougeEvaluator()
        self.bertscore = BERTScoreEvaluator() if use_bertscore else None
        self.answer_extractor = AnswerExtractor()
    
    def evaluate_single(
        self,
        question_id: str,
        question: str,
        reference: str,
        prediction: str,
        correct_choice: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
        latency_ms: int = 0,
        total_tokens: int = 0,
    ) -> EvaluationResult:
        """Evaluate a single question-answer pair"""
        
        # Extract answer choice
        extracted_choice = self.answer_extractor.extract(prediction, options)
        
        # Check correctness (for MCQ)
        is_correct = False
        if correct_choice and extracted_choice:
            is_correct = extracted_choice.upper() == correct_choice.upper()
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge.score(reference, prediction)
        
        # Calculate BERTScore (optional, can be slow)
        bert_result = None
        if self.bertscore:
            try:
                bert_result = self.bertscore.score(reference, prediction)
            except Exception as e:
                print(f"[BERTScore] Warning: {e}")
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            reference_answer=reference,
            predicted_answer=prediction,
            is_correct=is_correct,
            extracted_choice=extracted_choice,
            rouge_scores=rouge_scores,
            bert_score=bert_result,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
        )
    
    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of results
        
        Args:
            results: List of dicts with keys:
                - question_id, question, reference, prediction
                - correct_choice (optional), options (optional)
                - latency_ms (optional), total_tokens (optional)
        """
        evaluated = []
        
        for r in results:
            eval_result = self.evaluate_single(
                question_id=r.get("question_id", ""),
                question=r.get("question", ""),
                reference=r.get("reference", r.get("reference_answer", "")),
                prediction=r.get("prediction", r.get("predicted_answer", "")),
                correct_choice=r.get("correct_choice", r.get("answer", None)),
                options=r.get("options"),
                latency_ms=r.get("latency_ms", 0),
                total_tokens=r.get("total_tokens", 0),
            )
            evaluated.append(eval_result)
        
        return evaluated
    
    def aggregate(
        self,
        results: List[EvaluationResult],
        by_dataset: Optional[Dict[str, List[str]]] = None,
    ) -> AggregatedMetrics:
        """
        Aggregate evaluation results into summary metrics
        
        Args:
            results: List of EvaluationResult
            by_dataset: Optional mapping of dataset name to question IDs
        """
        if not results:
            return AggregatedMetrics(
                total_questions=0,
                accuracy=0,
                avg_rouge1_f1=0,
                avg_rouge2_f1=0,
                avg_rougeL_f1=0,
                avg_bert_f1=0,
                avg_latency_ms=0,
                avg_tokens=0,
                total_tokens=0,
                questions_per_second=0,
                by_dataset={},
            )
        
        n = len(results)
        
        # Calculate accuracy
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / n
        
        # Aggregate ROUGE scores
        rouge1_f1s = [r.rouge_scores.rouge1_f1 for r in results if r.rouge_scores]
        rouge2_f1s = [r.rouge_scores.rouge2_f1 for r in results if r.rouge_scores]
        rougeL_f1s = [r.rouge_scores.rougeL_f1 for r in results if r.rouge_scores]
        
        avg_rouge1 = sum(rouge1_f1s) / len(rouge1_f1s) if rouge1_f1s else 0
        avg_rouge2 = sum(rouge2_f1s) / len(rouge2_f1s) if rouge2_f1s else 0
        avg_rougeL = sum(rougeL_f1s) / len(rougeL_f1s) if rougeL_f1s else 0
        
        # Aggregate BERTScore
        bert_f1s = [r.bert_score.f1 for r in results if r.bert_score]
        avg_bert = sum(bert_f1s) / len(bert_f1s) if bert_f1s else 0
        
        # Aggregate latency and tokens
        total_latency = sum(r.latency_ms for r in results)
        total_tokens = sum(r.total_tokens for r in results)
        avg_latency = total_latency / n
        avg_tokens = total_tokens / n
        
        qps = n / (total_latency / 1000) if total_latency > 0 else 0
        
        # Per-dataset metrics
        dataset_metrics = {}
        if by_dataset:
            for dataset, qids in by_dataset.items():
                dataset_results = [r for r in results if r.question_id in qids]
                if dataset_results:
                    d_correct = sum(1 for r in dataset_results if r.is_correct)
                    d_rouge1 = [r.rouge_scores.rouge1_f1 for r in dataset_results if r.rouge_scores]
                    dataset_metrics[dataset] = {
                        "accuracy": d_correct / len(dataset_results),
                        "avg_rouge1_f1": sum(d_rouge1) / len(d_rouge1) if d_rouge1 else 0,
                        "count": len(dataset_results),
                    }
        
        return AggregatedMetrics(
            total_questions=n,
            accuracy=accuracy,
            avg_rouge1_f1=avg_rouge1,
            avg_rouge2_f1=avg_rouge2,
            avg_rougeL_f1=avg_rougeL,
            avg_bert_f1=avg_bert,
            avg_latency_ms=avg_latency,
            avg_tokens=avg_tokens,
            total_tokens=total_tokens,
            questions_per_second=qps,
            by_dataset=dataset_metrics,
        )


def print_metrics_table(
    baseline_metrics: AggregatedMetrics,
    latentmas_metrics: AggregatedMetrics,
    system_names: Tuple[str, str] = ("Traditional 4-Agent", "LatentMAS-SLoRA"),
):
    """Print formatted comparison table"""
    
    print("\n" + "="*80)
    print("EVALUATION METRICS COMPARISON")
    print("="*80)
    
    headers = ["Metric", system_names[0], system_names[1], "Δ (Improvement)"]
    
    rows = [
        ("Accuracy", 
         f"{baseline_metrics.accuracy:.2%}", 
         f"{latentmas_metrics.accuracy:.2%}",
         f"{(latentmas_metrics.accuracy - baseline_metrics.accuracy)*100:+.1f}%"),
        
        ("ROUGE-1 F1",
         f"{baseline_metrics.avg_rouge1_f1:.4f}",
         f"{latentmas_metrics.avg_rouge1_f1:.4f}",
         f"{(latentmas_metrics.avg_rouge1_f1 - baseline_metrics.avg_rouge1_f1)*100:+.2f}%"),
        
        ("ROUGE-2 F1",
         f"{baseline_metrics.avg_rouge2_f1:.4f}",
         f"{latentmas_metrics.avg_rouge2_f1:.4f}",
         f"{(latentmas_metrics.avg_rouge2_f1 - baseline_metrics.avg_rouge2_f1)*100:+.2f}%"),
        
        ("ROUGE-L F1",
         f"{baseline_metrics.avg_rougeL_f1:.4f}",
         f"{latentmas_metrics.avg_rougeL_f1:.4f}",
         f"{(latentmas_metrics.avg_rougeL_f1 - baseline_metrics.avg_rougeL_f1)*100:+.2f}%"),
        
        ("BERTScore F1",
         f"{baseline_metrics.avg_bert_f1:.4f}",
         f"{latentmas_metrics.avg_bert_f1:.4f}",
         f"{(latentmas_metrics.avg_bert_f1 - baseline_metrics.avg_bert_f1)*100:+.2f}%"),
        
        ("Avg Latency (ms)",
         f"{baseline_metrics.avg_latency_ms:.0f}",
         f"{latentmas_metrics.avg_latency_ms:.0f}",
         f"{baseline_metrics.avg_latency_ms / max(latentmas_metrics.avg_latency_ms, 1):.2f}x faster"),
        
        ("Total Tokens",
         f"{baseline_metrics.total_tokens:,}",
         f"{latentmas_metrics.total_tokens:,}",
         f"{(1 - latentmas_metrics.total_tokens / max(baseline_metrics.total_tokens, 1))*100:.1f}% reduction"),
        
        ("QPS",
         f"{baseline_metrics.questions_per_second:.2f}",
         f"{latentmas_metrics.questions_per_second:.2f}",
         f"{latentmas_metrics.questions_per_second / max(baseline_metrics.questions_per_second, 0.01):.2f}x"),
    ]
    
    # Print table
    col_widths = [25, 20, 20, 20]
    
    # Header
    print(f"\n{'│'.join(h.center(w) for h, w in zip(headers, col_widths))}")
    print("─" * sum(col_widths))
    
    # Rows
    for row in rows:
        print(f"{'│'.join(str(c).center(w) for c, w in zip(row, col_widths))}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test evaluation metrics
    rouge = RougeEvaluator()
    
    ref = "The answer is B. Hashimoto's thyroiditis causes hypothyroidism with elevated TSH."
    pred = "Based on the symptoms, the answer is B, Hashimoto's thyroiditis. This condition causes hypothyroidism."
    
    scores = rouge.score(ref, pred)
    print(f"ROUGE-1 F1: {scores.rouge1_f1:.4f}")
    print(f"ROUGE-2 F1: {scores.rouge2_f1:.4f}")
    print(f"ROUGE-L F1: {scores.rougeL_f1:.4f}")
    
    # Test answer extraction
    test_texts = [
        "The correct answer is B.",
        "After analysis, I believe A is the right choice.",
        '{"answer_choice": "C"}',
        "Based on the evidence, option D seems most appropriate.",
    ]
    
    print("\nAnswer Extraction:")
    for text in test_texts:
        choice = AnswerExtractor.extract(text)
        print(f"  '{text[:50]}...' → {choice}")
