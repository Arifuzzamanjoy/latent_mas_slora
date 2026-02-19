"""
Benchmark Runner – evaluate a LatentMAS pipeline against standard benchmarks.

Supported benchmarks
--------------------
gsm8k           – grade-school math (accuracy on final numeric answer)
medqa           – medical USMLE-style 4-option MCQ
arc_challenge   – ARC challenge set 4-option MCQ
humaneval_plus  – code generation pass@1 (functional correctness)

All datasets are loaded from HuggingFace ``datasets`` library on first use.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset catalogue – maps short name → HF identifier + split + field names
# ---------------------------------------------------------------------------

_DATASET_CATALOGUE: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",          # contains "####  <number>"
        "options_field": None,
        "task_type": "math",
    },
    "medqa": {
        "hf_path": "bigbio/med_qa",
        "hf_name": "med_qa_en_4options_bigbio_qa",
        "split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "options_field": "choices",
        "task_type": "mcq",
    },
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "split": "test",
        "question_field": "question",
        "answer_field": "answerKey",
        "options_field": "choices",
        "task_type": "mcq",
    },
    "humaneval_plus": {
        "hf_path": "evalplus/humanevalplus",
        "hf_name": None,
        "split": "test",
        "question_field": "prompt",
        "answer_field": "canonical_solution",
        "options_field": None,
        "task_type": "code",
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    """Per-question result."""
    qid: int
    question: str
    gold: str
    prediction: str
    extracted: str
    correct: bool
    tokens_generated: int
    latency_ms: int


@dataclass
class BenchmarkResult:
    """Aggregate result for a full benchmark run."""
    dataset_name: str
    accuracy: float
    num_correct: int
    num_total: int
    avg_tokens: float
    avg_latency_ms: float
    per_question: List[QuestionResult] = field(default_factory=list)
    ci_lower: float = 0.0  # 95 % bootstrap CI
    ci_upper: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @property
    def correctness_vector(self) -> List[int]:
        """Return a list of 0/1 for paired bootstrap tests."""
        return [int(q.correct) for q in self.per_question]


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run any supported benchmark through a LatentMAS pipeline (or bare model).

    Parameters
    ----------
    pipeline : callable
        Anything with a ``__call__(question: str) -> str`` protocol, *or* a
        ``LatentMASSystem`` whose ``.run()`` we call.
    tokenizer
        HF tokenizer (used only for token-counting; may be ``None``).
    device : str
        For logging / metadata only (inference device is owned by *pipeline*).
    cache_dir : str
        Where to cache HF datasets.
    """

    def __init__(
        self,
        pipeline,
        tokenizer=None,
        device: str = "cuda",
        cache_dir: str = "/home/caches",
    ):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.device = device
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def load_dataset(self, name: str) -> List[dict]:
        """Load a benchmark dataset into a flat list of dicts.

        Each dict has *at least*::

            {"question": str, "answer": str}

        and optionally ``"options": List[str]``.
        """
        from datasets import load_dataset  # lazy – heavy import

        name = name.lower()
        if name not in _DATASET_CATALOGUE:
            raise ValueError(
                f"Unknown benchmark '{name}'. "
                f"Choose from: {sorted(_DATASET_CATALOGUE)}"
            )
        spec = _DATASET_CATALOGUE[name]

        ds = load_dataset(
            spec["hf_path"],
            name=spec["hf_name"],
            split=spec["split"],
            cache_dir=self.cache_dir,
        )

        items: List[dict] = []
        for row in ds:
            item = self._normalize_row(row, spec, name)
            if item is not None:
                items.append(item)
        logger.info("Loaded %d examples from '%s'", len(items), name)
        return items

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_row(row: dict, spec: dict, dataset_name: str) -> Optional[dict]:
        """Convert a HF row into our canonical format."""
        question = row.get(spec["question_field"], "")
        answer_raw = row.get(spec["answer_field"], "")

        # ---- GSM8K: gold answer is after "####"
        if dataset_name == "gsm8k":
            match = re.search(r"####\s*(.+)", str(answer_raw))
            answer = match.group(1).strip().replace(",", "") if match else str(answer_raw).strip()
            return {"question": question, "answer": answer}

        # ---- MedQA (bigbio format)
        if dataset_name == "medqa":
            choices = row.get("choices", [])
            answer_list = row.get("answer", [])
            if isinstance(answer_list, list):
                answer = answer_list[0] if answer_list else ""
            else:
                answer = str(answer_list)
            option_texts = choices if isinstance(choices, list) else []
            letters = [chr(ord("A") + i) for i in range(len(option_texts))]
            options_str = "\n".join(f"{l}. {t}" for l, t in zip(letters, option_texts))
            full_q = f"{question}\n\n{options_str}" if options_str else question
            # Find which letter matches the gold answer text
            gold_letter = ""
            for l, t in zip(letters, option_texts):
                if t.strip().lower() == answer.strip().lower():
                    gold_letter = l
                    break
            if not gold_letter:
                gold_letter = answer  # fallback
            return {"question": full_q, "answer": gold_letter, "options": option_texts}

        # ---- ARC-Challenge
        if dataset_name == "arc_challenge":
            choices_obj = row.get("choices", {})
            labels = choices_obj.get("label", [])
            texts = choices_obj.get("text", [])
            options_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            full_q = f"{question}\n\n{options_str}" if options_str else question
            return {"question": full_q, "answer": str(answer_raw).strip(), "options": texts}

        # ---- HumanEval+
        if dataset_name == "humaneval_plus":
            return {"question": question, "answer": str(answer_raw).strip()}

        return {"question": question, "answer": str(answer_raw).strip()}

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------
    @staticmethod
    def extract_answer(response: str, dataset_name: str) -> str:
        """Extract the final answer from a free-form model response.

        * MCQ  → single letter A/B/C/D
        * Math → final numeric value
        * Code → code block contents
        """
        if not response:
            return ""

        task_type = _DATASET_CATALOGUE.get(dataset_name, {}).get("task_type", "")

        # ----------------------------------------------------------------
        # MCQ extraction – prefer \\boxed{X}, then "answer is X", then last letter
        # ----------------------------------------------------------------
        if task_type == "mcq":
            # \boxed{A}
            m = re.search(r'\\boxed\{([A-Da-d])\}', response)
            if m:
                return m.group(1).upper()
            # "the answer is (A)" / "Answer: A"
            m = re.search(r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-Da-d])\)?', response, re.I)
            if m:
                return m.group(1).upper()
            # Last standalone letter A-D
            letters = re.findall(r'\b([A-Da-d])\b', response)
            if letters:
                return letters[-1].upper()
            return response.strip()[:1].upper()

        # ----------------------------------------------------------------
        # Math extraction – last number in response (or \\boxed{…})
        # ----------------------------------------------------------------
        if task_type == "math":
            m = re.search(r'\\boxed\{([^}]+)\}', response)
            if m:
                return m.group(1).strip().replace(",", "")
            m = re.search(r'(?:answer|result)\s*(?:is|=|:)\s*([\-\d.,/]+)', response, re.I)
            if m:
                return m.group(1).strip().replace(",", "")
            numbers = re.findall(r'-?\d[\d,]*\.?\d*', response)
            if numbers:
                return numbers[-1].replace(",", "")
            return response.strip()

        # ----------------------------------------------------------------
        # Code extraction – first ```…``` block or full response
        # ----------------------------------------------------------------
        if task_type == "code":
            m = re.search(r'```(?:python)?\n?(.*?)```', response, re.S)
            if m:
                return m.group(1).strip()
            return response.strip()

        return response.strip()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    def evaluate(
        self,
        dataset_name: str,
        max_samples: int = -1,
        seed: int = 42,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run full evaluation on *dataset_name*.

        Parameters
        ----------
        dataset_name : str
            One of the supported benchmarks.
        max_samples : int
            Cap on how many examples to evaluate (``-1`` = all).
        seed : int
            For reproducible subsampling.
        pipeline_kwargs : dict | None
            Extra kwargs forwarded to ``pipeline.run()`` or ``pipeline()``.

        Returns
        -------
        BenchmarkResult
        """
        import random as _random
        _random.seed(seed)

        items = self.load_dataset(dataset_name)

        if 0 < max_samples < len(items):
            _random.shuffle(items)
            items = items[:max_samples]

        per_question: List[QuestionResult] = []
        correct = 0
        total_tokens = 0
        total_latency = 0
        pkw = pipeline_kwargs or {}

        for idx, item in enumerate(items):
            question = item["question"]
            gold = item["answer"]

            t0 = time.time()
            response = self._call_pipeline(question, **pkw)
            latency = int((time.time() - t0) * 1000)

            extracted = self.extract_answer(response, dataset_name)
            is_correct = self._check_answer(extracted, gold, dataset_name)

            n_tokens = len(self.tokenizer.encode(response)) if self.tokenizer else len(response.split())

            if is_correct:
                correct += 1
            total_tokens += n_tokens
            total_latency += latency

            per_question.append(QuestionResult(
                qid=idx,
                question=question[:200],
                gold=gold,
                prediction=response[:500],
                extracted=extracted,
                correct=is_correct,
                tokens_generated=n_tokens,
                latency_ms=latency,
            ))

            if (idx + 1) % 25 == 0:
                logger.info(
                    "[%s] %d/%d  running_acc=%.2f%%",
                    dataset_name, idx + 1, len(items),
                    100 * correct / (idx + 1),
                )

        n = len(per_question)
        accuracy = correct / n if n else 0.0
        ci_lo, ci_hi = self._bootstrap_ci([q.correct for q in per_question])

        result = BenchmarkResult(
            dataset_name=dataset_name,
            accuracy=accuracy,
            num_correct=correct,
            num_total=n,
            avg_tokens=total_tokens / n if n else 0,
            avg_latency_ms=total_latency / n if n else 0,
            per_question=per_question,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            metadata={"seed": seed, "max_samples": max_samples},
        )
        logger.info(
            "[%s] DONE  accuracy=%.2f%% (%d/%d)  95%%CI=[%.2f%%, %.2f%%]",
            dataset_name, 100 * accuracy, correct, n,
            100 * ci_lo, 100 * ci_hi,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _call_pipeline(self, question: str, **kwargs) -> str:
        """Adapter: call the pipeline and return a plain string answer."""
        # LatentMASSystem
        if hasattr(self.pipeline, "run"):
            result = self.pipeline.run(question, **kwargs)
            if hasattr(result, "final_answer"):
                return result.final_answer
            return str(result)
        # Bare callable
        if callable(self.pipeline):
            return str(self.pipeline(question, **kwargs))
        raise TypeError(f"Pipeline type {type(self.pipeline)} is not callable")

    @staticmethod
    def _check_answer(extracted: str, gold: str, dataset_name: str) -> bool:
        """Compare extracted answer to gold.  Normalises both sides."""
        e = extracted.strip().lower()
        g = gold.strip().lower()
        if not e or not g:
            return False

        task_type = _DATASET_CATALOGUE.get(dataset_name, {}).get("task_type", "")

        if task_type == "mcq":
            return e[:1] == g[:1]

        if task_type == "math":
            try:
                return abs(float(e.replace(",", "")) - float(g.replace(",", ""))) < 1e-3
            except ValueError:
                return e == g

        # code / general – exact match after whitespace norm
        return " ".join(e.split()) == " ".join(g.split())

    @staticmethod
    def _bootstrap_ci(
        correct_vec: Sequence[bool],
        n_bootstrap: int = 5000,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """95 % bootstrap confidence interval for accuracy."""
        import random as _r
        n = len(correct_vec)
        if n == 0:
            return (0.0, 0.0)
        accs = []
        for _ in range(n_bootstrap):
            sample = _r.choices(correct_vec, k=n)
            accs.append(sum(sample) / n)
        accs.sort()
        lo = accs[int(alpha / 2 * n_bootstrap)]
        hi = accs[int((1 - alpha / 2) * n_bootstrap)]
        return (lo, hi)
