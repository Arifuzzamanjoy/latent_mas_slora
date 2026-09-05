"""
Dataset loading and deterministic segmentation.

Every dataset is normalized to a list of EvalItem so that methods, metrics and
reports never need to know where the data came from.

Segmentation is the answer to "run on 10% / 50% / 100%": it is a pure function
of (dataset, seed, fraction, offset, limit, stratify), so the 10% slice is
always a subset of the 50% slice for the same seed.
"""

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ABCDEFGHIJ"


@dataclass
class EvalItem:
    id: str
    question: str  # fully rendered prompt body (with choices)
    gold: str  # letter for mcq, string for numeric/text
    choices: List[str] = field(default_factory=list)  # choice texts, no letters
    domain: str = "general"
    source: str = "local"
    task_type: str = "mcq"  # mcq | numeric | text
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_choices(self) -> int:
        return len(self.choices) if self.choices else 4

    def rendered(self, choice_order: Optional[List[int]] = None) -> "EvalItem":
        """Return a copy with choices permuted (for position-bias testing)."""
        if not choice_order or not self.choices:
            return self
        new_choices = [self.choices[i] for i in choice_order]
        gold_idx = LETTERS.index(self.gold) if self.gold in LETTERS else -1
        new_gold = self.gold
        if gold_idx >= 0:
            new_gold = LETTERS[choice_order.index(gold_idx)]
        stem = self.metadata.get("stem", self.question.split("\n" + LETTERS[0] + ".")[0])
        body = (
            stem.rstrip()
            + "\n"
            + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(new_choices))
        )
        return EvalItem(
            id=self.id,
            question=body,
            gold=new_gold,
            choices=new_choices,
            domain=self.domain,
            source=self.source,
            task_type=self.task_type,
            metadata={**self.metadata, "permutation": choice_order, "orig_gold": self.gold},
        )


def _render(stem: str, choices: List[str]) -> str:
    if not choices:
        return stem.strip()
    return stem.strip() + "\n" + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


# ─── Local files ─────────────────────────────────────────────────────────────


def _parse_local_record(rec: Dict[str, Any], idx: int, source: str) -> EvalItem:
    q = rec.get("question") or rec.get("prompt") or ""
    choices = rec.get("choices") or rec.get("options") or []
    if isinstance(choices, dict):  # {"A": "...", "B": "..."}
        choices = [choices[k] for k in sorted(choices)]

    gold = str(rec.get("gold_letter") or rec.get("answer") or rec.get("gold") or "").strip()
    task_type = rec.get("task_type") or (
        "mcq" if re.fullmatch(r"[A-J]", gold.upper()) else "numeric"
    )

    stem = q
    if not choices and "\n" in q:  # choices embedded in the question text
        lines = q.split("\n")
        opt_idx = next(
            (i for i, line in enumerate(lines) if re.match(r"^\s*[A-J][.)]\s", line)), None
        )
        if opt_idx is not None:
            stem = "\n".join(lines[:opt_idx])
            choices = [re.sub(r"^\s*[A-J][.)]\s*", "", ln) for ln in lines[opt_idx:] if ln.strip()]

    if task_type == "mcq":
        m = re.search(r"([A-J])", gold.upper())
        gold = m.group(1) if m else gold.upper()

    return EvalItem(
        id=str(rec.get("id", idx + 1)),
        question=_render(stem, choices) if choices else q,
        gold=gold,
        choices=choices,
        domain=rec.get("domain", "general"),
        source=source,
        task_type=task_type,
        metadata={"stem": stem},
    )


def load_local(path: str) -> List[EvalItem]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"dataset file not found: {p}")

    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        records = json.loads(p.read_text())
        if isinstance(records, dict):
            records = records.get("questions") or records.get("data") or []

    return [_parse_local_record(r, i, f"local:{p.name}") for i, r in enumerate(records)]


# ─── HuggingFace datasets ────────────────────────────────────────────────────


def _hf(path: str, name: Optional[str], split: str, cache_dir: Optional[str]):
    from datasets import load_dataset as hf_load

    return hf_load(path, name, split=split, cache_dir=cache_dir)


def load_hf(
    spec: str, split: Optional[str], cache_dir: Optional[str], max_items: Optional[int] = None
) -> List[EvalItem]:
    """Load one of the registered HF benchmarks. `spec` may carry a suffix,
    e.g. mmlu:anatomy."""
    base, _, arg = spec.partition(":")
    items: List[EvalItem] = []

    if base == "medqa":
        ds = _hf("GBaker/MedQA-USMLE-4-options", None, split or "test", cache_dir)
        for i, r in enumerate(ds):
            opts = r["options"]
            choices = [opts[k] for k in sorted(opts)]
            gold = r.get("answer_idx") or ""
            items.append(
                EvalItem(
                    f"medqa-{i}",
                    _render(r["question"], choices),
                    str(gold).strip().upper(),
                    choices,
                    "medical",
                    "medqa",
                    "mcq",
                    {"stem": r["question"]},
                )
            )

    elif base == "medmcqa":
        ds = _hf("openlifescienceai/medmcqa", None, split or "validation", cache_dir)
        for i, r in enumerate(ds):
            choices = [r["opa"], r["opb"], r["opc"], r["opd"]]
            items.append(
                EvalItem(
                    f"medmcqa-{i}",
                    _render(r["question"], choices),
                    LETTERS[int(r["cop"])],
                    choices,
                    "medical",
                    "medmcqa",
                    "mcq",
                    {"stem": r["question"], "subject": r.get("subject_name")},
                )
            )

    elif base == "pubmedqa":
        ds = _hf("qiaojin/PubMedQA", arg or "pqa_labeled", split or "train", cache_dir)
        choices = ["yes", "no", "maybe"]
        for i, r in enumerate(ds):
            ctx = " ".join(r["context"]["contexts"]) if isinstance(r.get("context"), dict) else ""
            stem = f"{ctx}\n\nQuestion: {r['question']}"
            gold = LETTERS[choices.index(r["final_decision"])]
            items.append(
                EvalItem(
                    f"pubmedqa-{i}",
                    _render(stem, choices),
                    gold,
                    choices,
                    "medical",
                    "pubmedqa",
                    "mcq",
                    {"stem": stem},
                )
            )

    elif base == "mmlu":
        ds = _hf("cais/mmlu", arg or "all", split or "test", cache_dir)
        for i, r in enumerate(ds):
            choices = list(r["choices"])
            items.append(
                EvalItem(
                    f"mmlu-{i}",
                    _render(r["question"], choices),
                    LETTERS[int(r["answer"])],
                    choices,
                    r.get("subject", "general"),
                    "mmlu",
                    "mcq",
                    {"stem": r["question"], "subject": r.get("subject")},
                )
            )

    elif base == "mmlu_pro":
        ds = _hf("TIGER-Lab/MMLU-Pro", None, split or "test", cache_dir)
        for i, r in enumerate(ds):
            choices = list(r["options"])
            items.append(
                EvalItem(
                    f"mmlupro-{i}",
                    _render(r["question"], choices),
                    str(r["answer"]).strip().upper(),
                    choices,
                    r.get("category", "general"),
                    "mmlu_pro",
                    "mcq",
                    {"stem": r["question"]},
                )
            )

    elif base == "arc":
        ds = _hf("allenai/ai2_arc", arg or "ARC-Challenge", split or "test", cache_dir)
        for i, r in enumerate(ds):
            choices = list(r["choices"]["text"])
            labels = list(r["choices"]["label"])
            key = str(r["answerKey"])
            gold = LETTERS[labels.index(key)] if key in labels else key
            items.append(
                EvalItem(
                    f"arc-{i}",
                    _render(r["question"], choices),
                    gold,
                    choices,
                    "reasoning",
                    "arc",
                    "mcq",
                    {"stem": r["question"]},
                )
            )

    elif base == "gsm8k":
        ds = _hf("openai/gsm8k", arg or "main", split or "test", cache_dir)
        for i, r in enumerate(ds):
            gold = r["answer"].split("####")[-1].strip()
            items.append(
                EvalItem(
                    f"gsm8k-{i}",
                    r["question"],
                    gold,
                    [],
                    "math",
                    "gsm8k",
                    "numeric",
                    {"stem": r["question"], "rationale": r["answer"]},
                )
            )

    elif base == "math500":
        ds = _hf("HuggingFaceH4/MATH-500", None, split or "test", cache_dir)
        for i, r in enumerate(ds):
            items.append(
                EvalItem(
                    f"math500-{i}",
                    r["problem"],
                    str(r["answer"]),
                    [],
                    "math",
                    "math500",
                    "numeric",
                    {"stem": r["problem"], "level": r.get("level")},
                )
            )

    elif base == "gpqa":
        ds = _hf("Idavidrein/gpqa", arg or "gpqa_diamond", split or "train", cache_dir)
        for i, r in enumerate(ds):
            choices = [
                r["Correct Answer"],
                r["Incorrect Answer 1"],
                r["Incorrect Answer 2"],
                r["Incorrect Answer 3"],
            ]
            rng = random.Random(i)
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [choices[j] for j in order]
            items.append(
                EvalItem(
                    f"gpqa-{i}",
                    _render(r["Question"], shuffled),
                    LETTERS[order.index(0)],
                    shuffled,
                    "reasoning",
                    "gpqa",
                    "mcq",
                    {"stem": r["Question"]},
                )
            )
    else:
        raise ValueError(f"unknown dataset '{spec}'. See --list-datasets.")

    if max_items:
        items = items[:max_items]
    return items


HF_DATASETS = {
    "medqa": "MedQA-USMLE 4-option (medical, test)",
    "medmcqa": "MedMCQA (medical, validation)",
    "pubmedqa": "PubMedQA pqa_labeled, yes/no/maybe (medical)",
    "mmlu": "MMLU, ':subject' selects a config e.g. mmlu:anatomy",
    "mmlu_pro": "MMLU-Pro, 10 options (lower guess floor)",
    "arc": "ARC-Challenge, ':ARC-Easy' for the easy split",
    "gsm8k": "GSM8K grade-school math (numeric)",
    "math500": "MATH-500 (numeric)",
    "gpqa": "GPQA, ':gpqa_diamond' by default (gated dataset)",
}


# ─── Entry point ─────────────────────────────────────────────────────────────


def _slug(spec: str) -> str:
    """Filesystem/id-safe tag for a dataset spec: 'mmlu:college_mathematics' -> 'mmlu-college_mathematics'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", spec.strip()).strip("-")


def _ensure_unique_ids(items: List[EvalItem]) -> List[EvalItem]:
    """
    Guarantee globally unique item ids.

    Ids are the join key for resume, for paired statistics and for permutation
    grouping, so a collision does not merely rename an item - the runner treats
    the duplicate as work already done and silently drops it. Disambiguate here,
    loudly, rather than losing items downstream.
    """
    seen: Dict[str, int] = {}
    for it in items:
        n = seen.get(it.id, 0)
        seen[it.id] = n + 1
        if n:
            it.id = f"{it.id}#{n + 1}"
    dupes = sum(v - 1 for v in seen.values() if v > 1)
    if dupes:
        print(f"[data] warning: {dupes} duplicate item id(s) disambiguated with a '#n' suffix")
    return items


def load_dataset(
    spec: str, split: Optional[str] = None, cache_dir: Optional[str] = None
) -> List[EvalItem]:
    """
    spec forms:
      sample                  -> data/sample_data.json (the repo's 5 questions)
      local:path/to/file.json -> any local json/jsonl
      medqa, gsm8k, mmlu:anatomy, ... -> HuggingFace benchmarks
      mix:medqa,gsm8k         -> concatenation of several of the above
    """
    spec = spec.strip()
    if spec == "sample":
        return _ensure_unique_ids(load_local("data/sample_data.json"))
    if spec.startswith("local:"):
        return _ensure_unique_ids(load_local(spec[len("local:") :]))
    if spec.startswith("mix:"):
        # Sub-datasets number their items independently (two MMLU subjects both
        # emit mmlu-0, mmlu-1, ...), so every part is namespaced by its spec.
        out: List[EvalItem] = []
        for part in spec[len("mix:") :].split(","):
            part = part.strip()
            tag = _slug(part)
            for it in load_dataset(part, split, cache_dir):
                it.id = f"{tag}/{it.id}"
                out.append(it)
        return _ensure_unique_ids(out)
    return _ensure_unique_ids(load_hf(spec, split, cache_dir))


# ─── Segmentation ────────────────────────────────────────────────────────────


def segment(
    items: List[EvalItem],
    fraction: float = 1.0,
    limit: Optional[int] = None,
    offset: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    stratify_by: Optional[str] = "domain",
    min_per_group: int = 1,
) -> List[EvalItem]:
    """
    Deterministically select a subset.

    Order of operations: shuffle (seeded) -> offset -> fraction -> limit.

    With shuffle=True and a fixed seed, the fraction slices are nested:
    the 10% slice is contained in the 50% slice, which is contained in 100%.
    That is what makes a data-scaling curve interpretable - each larger slice
    only adds items, it never swaps them.

    stratify_by keeps the per-group proportions of a field (default: domain)
    so a 10% slice of a mixed medical/math/code set stays mixed.

    Very small fractions are supported: 0.001 of MedQA is one item. min_per_group
    is the floor each stratum keeps (1 by default, so every domain stays
    represented); set it to 0 to let tiny fractions drop whole strata instead.
    """
    if not items:
        return []

    pool = list(items)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pool)

    if offset:
        pool = pool[offset:]

    fraction = max(0.0, min(1.0, fraction))

    if fraction < 1.0:
        if stratify_by:
            groups: Dict[str, List[EvalItem]] = {}
            for it in pool:
                groups.setdefault(str(getattr(it, stratify_by, "?")), []).append(it)
            keep: List[EvalItem] = []
            for _, g in sorted(groups.items()):
                n = max(min_per_group, round(len(g) * fraction)) if g else 0
                keep.extend(g[:n])  # prefix -> nested across fractions
            order = {id(it): i for i, it in enumerate(pool)}
            pool = sorted(keep, key=lambda it: order[id(it)])
        else:
            n = max(1, round(len(pool) * fraction))
            pool = pool[:n]  # never empty: a sub-item fraction still yields one

    if limit is not None:
        pool = pool[:limit]

    return pool


def describe_segment(items: List[EvalItem]) -> Dict[str, Any]:
    from collections import Counter

    return {
        "n": len(items),
        "by_domain": dict(Counter(i.domain for i in items)),
        "by_source": dict(Counter(i.source for i in items)),
        "by_task_type": dict(Counter(i.task_type for i in items)),
    }


def item_to_dict(item: EvalItem) -> Dict[str, Any]:
    d = asdict(item)
    d.pop("metadata", None)
    return d
