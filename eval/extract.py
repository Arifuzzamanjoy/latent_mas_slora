"""
Answer extraction.

Extraction is part of the measurement, so it is reported rather than hidden:
every prediction carries the rule that produced it, whether it failed, and
whether the model actually emitted the format the prompt asked for.

Two accuracies fall out of that last flag, following the convention
lm-evaluation-harness uses for GSM8K:

  strict   - only answers the model boxed in the requested format count
  flexible - fall back to the answer stated in prose when the box is malformed

A large gap between them is a prompt bug, not a reasoning result: it means the
model solved the problem and then wrote the answer down wrong. `\\boxed{A}` on a
numeric item is the canonical case, and it used to be scored as a confident
wrong answer with `parse fail = 0%`.
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
    strict: bool = True  # True when the model emitted the requested \boxed{} format

    @property
    def format_violation(self) -> bool:
        """The model produced an answer, but not in the format it was asked for."""
        return not self.strict and not self.failed


# ─── Multiple choice ─────────────────────────────────────────────────────────


def _letter_class(num_choices: int) -> str:
    return _LETTERS[: max(2, min(num_choices, len(_LETTERS)))]


def extract_mcq(text: str, num_choices: int = 4) -> ExtractResult:
    """Extract a choice letter from free-form model output."""
    if not text or not text.strip():
        return ExtractResult(UNKNOWN, "empty", True, False)

    L = _letter_class(num_choices)
    cls = f"[{L}]"

    # 0. Self-consistency header written by src/pipelines/hierarchical.py
    m = re.match(rf"\[Self-Consistency Vote:\s*({cls})\]", text)
    if m:
        return ExtractResult(m.group(1).upper(), "sc_header", False)

    # 1. \boxed{X} - the format every prompt in this repo asks for.
    # Scan boxes last-to-first: a later box that holds no letter (a bare number,
    # say) should not mask an earlier well-formed one.
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    for b in reversed(boxed):
        m = re.search(cls, b.strip().upper())
        if m:
            return ExtractResult(m.group(0), "boxed", False, True)

    # Past here the model did not box a valid option letter. Anything we recover
    # is a rescue, so it is reported as non-strict.
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
            return ExtractResult(m.group(1).upper(), rule, False, False)

    # 3. Last non-empty line containing a stand-alone letter
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        found = re.findall(rf"\b({cls})\b", lines[-1].upper())
        if found:
            return ExtractResult(found[-1], "last_line", False, False)

    rule = "boxed_no_letter" if boxed else "no_match"
    return ExtractResult(UNKNOWN, rule, True, False)


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
        return ExtractResult(UNKNOWN, "empty", True, False)

    # 1. \boxed{N}, scanned last-to-first so a trailing malformed box does not
    # mask a well-formed one earlier in the response.
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    for b in reversed(boxed):
        m = re.search(_NUM, b)
        if m:
            return ExtractResult(normalize_number(m.group(0)), "boxed", False, True)

    # A box exists but holds no number - e.g. \boxed{A}, an option letter emitted
    # on a numeric item because the prompt asked for one. The model answered in
    # the wrong format. Fall through to the flexible rules, but never report the
    # result as strict: the format was violated whether or not we recover it.
    # 2. Explicit natural-language commitment
    m = re.search(rf"(?:final\s+)?answer\s*(?:is|:)\s*\**\s*({_NUM})", text, re.IGNORECASE)
    if m:
        return ExtractResult(normalize_number(m.group(1)), "answer_is", False, False)

    # 3. GSM8K gold format
    m = re.search(rf"####\s*({_NUM})", text)
    if m:
        return ExtractResult(normalize_number(m.group(1)), "hash", False, False)

    # 4. Last number anywhere (lm-eval-harness "flexible-extract"). Weak: on
    # "3 loaves cost $4 more than 2 bagels" it returns 2. Recorded as its own
    # rule so its contribution stays visible in the rule histogram.
    nums = re.findall(_NUM, text)
    if nums:
        return ExtractResult(normalize_number(nums[-1]), "last_number", False, False)

    rule = "boxed_no_number" if boxed else "no_match"
    return ExtractResult(UNKNOWN, rule, True, False)


def extract_answer(text: str, task_type: str = "mcq", num_choices: int = 4) -> ExtractResult:
    """Dispatch on task type."""
    if task_type == "numeric":
        return extract_numeric(text)
    if task_type == "mcq":
        return extract_mcq(text, num_choices)
    # free-form text: exact-match on the stripped output
    stripped = (text or "").strip()
    return ExtractResult(stripped, "raw", not bool(stripped), bool(stripped))


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
