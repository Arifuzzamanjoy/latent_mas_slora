"""
Answer extraction.

Extraction is part of the measurement, so it is reported rather than hidden:
every prediction carries the rule that produced it and whether it failed.
"""

import re
from dataclasses import dataclass
from typing import List

UNKNOWN = "UNKNOWN"

_LETTERS = "ABCDEFGHIJ"


@dataclass
class ExtractResult:
    answer: str  # normalized prediction, or UNKNOWN
    rule: str  # which rule fired ("boxed", "answer_is", "last_line", ...)
    failed: bool  # True when nothing matched


# ─── Multiple choice ─────────────────────────────────────────────────────────


def _letter_class(num_choices: int) -> str:
    return _LETTERS[: max(2, min(num_choices, len(_LETTERS)))]


def extract_mcq(text: str, num_choices: int = 4) -> ExtractResult:
    """Extract a choice letter from free-form model output."""
    if not text or not text.strip():
        return ExtractResult(UNKNOWN, "empty", True)

    L = _letter_class(num_choices)
    cls = f"[{L}]"

    # 0. Self-consistency header written by src/pipelines/hierarchical.py
    m = re.match(rf"\[Self-Consistency Vote:\s*({cls})\]", text)
    if m:
        return ExtractResult(m.group(1).upper(), "sc_header", False)

    # 1. \boxed{X} - the format every prompt in this repo asks for
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        m = re.search(cls, boxed[-1].strip().upper())
        if m:
            return ExtractResult(m.group(0), "boxed", False)

    # 2. Explicit natural-language commitments
    # \b guards stop "answer depends" matching the D in "depends"
    patterns = [
        (rf"(?:final\s+)?answer\s*(?:is\b)?\s*:?\s*[\*\_]*\s*\(?\b({cls})\b\)?", "answer_is"),
        (rf"(?:correct\s+)?option\s*(?:is\b)?\s*:?\s*[\*\_]*\s*\(?\b({cls})\b\)?", "option_is"),
        (rf"(?:choice\s+)?\b({cls})\b\s*(?:is\s+correct|is\s+the\s+answer)", "is_correct"),
    ]
    for pat, rule in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return ExtractResult(m.group(1).upper(), rule, False)

    # 3. Last non-empty line containing a stand-alone letter
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        found = re.findall(rf"\b({cls})\b", lines[-1].upper())
        if found:
            return ExtractResult(found[-1], "last_line", False)

    return ExtractResult(UNKNOWN, "no_match", True)


# ─── Numeric / free-form ─────────────────────────────────────────────────────

_NUM = r"-?\$?\d[\d,]*\.?\d*"


def normalize_number(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def extract_numeric(text: str) -> ExtractResult:
    """Extract a final numeric answer (GSM8K / MATH style)."""
    if not text or not text.strip():
        return ExtractResult(UNKNOWN, "empty", True)

    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        m = re.search(_NUM, boxed[-1])
        if m:
            return ExtractResult(normalize_number(m.group(0)), "boxed", False)
        return ExtractResult(boxed[-1].strip(), "boxed_text", False)

    m = re.search(rf"(?:final\s+)?answer\s*(?:is|:)\s*\**\s*({_NUM})", text, re.IGNORECASE)
    if m:
        return ExtractResult(normalize_number(m.group(1)), "answer_is", False)

    # GSM8K gold format
    m = re.search(rf"####\s*({_NUM})", text)
    if m:
        return ExtractResult(normalize_number(m.group(1)), "hash", False)

    nums = re.findall(_NUM, text)
    if nums:
        return ExtractResult(normalize_number(nums[-1]), "last_number", False)

    return ExtractResult(UNKNOWN, "no_match", True)


def extract_answer(text: str, task_type: str = "mcq", num_choices: int = 4) -> ExtractResult:
    """Dispatch on task type."""
    if task_type == "numeric":
        return extract_numeric(text)
    if task_type == "mcq":
        return extract_mcq(text, num_choices)
    # free-form text: exact-match on the stripped output
    return ExtractResult((text or "").strip(), "raw", not bool((text or "").strip()))


def is_correct(pred: str, gold: str, task_type: str = "mcq") -> bool:
    if pred == UNKNOWN:
        return False
    if task_type == "numeric":
        return normalize_number(pred) == normalize_number(gold)
    if task_type == "mcq":
        return pred.strip().upper() == gold.strip().upper()
    return pred.strip().lower() == gold.strip().lower()


def majority_vote(votes: List[str]) -> tuple:
    """Majority vote ignoring UNKNOWN unless everything is UNKNOWN.

    Returns (winner, counts_dict, agreement_fraction).
    """
    from collections import Counter

    valid = [v for v in votes if v != UNKNOWN]
    pool = valid or list(votes)
    counts = Counter(pool)
    if not counts:
        return UNKNOWN, {}, 0.0
    winner, n = counts.most_common(1)[0]
    return winner, dict(Counter(votes)), n / len(votes)
