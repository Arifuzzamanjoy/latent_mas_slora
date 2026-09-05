"""
The runner.

Responsibilities that are deliberately kept here rather than in methods, so
that they are applied identically to every method:

  - dataset segmentation and option permutation
  - self-consistency sampling and voting
  - seeding and repeats
  - scoring, record writing, resume
  - loading exactly one backend at a time
"""

import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .backends import build_backend
from .config import EvalConfig
from .data import EvalItem, describe_segment, load_dataset, segment
from .extract import UNKNOWN, is_correct, majority_vote
from .methods import backend_for, build_method, expand_methods


# ─── Option permutation ──────────────────────────────────────────────────────

def permutations_for(item: EvalItem, mode: str) -> List[Optional[List[int]]]:
    """
    Answer-position variants of one item.

    A 7B model on 4-choice questions has a measurable preference for certain
    positions; running the cyclic shifts turns that bias from an unknown into a
    reported number (position_consistency).
    """
    n = len(item.choices)
    if mode == "none" or n < 2:
        return [None]
    if mode == "cyclic":
        return [[(i + s) % n for i in range(n)] for s in range(n)]
    if mode == "all":
        from itertools import permutations as iperm
        return [list(p) for p in iperm(range(n))]
    raise ValueError(f"unknown --permute-options mode: {mode}")


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


# ─── Runner ──────────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, cfg: EvalConfig, dry_run: bool = False, verbose: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.verbose = verbose

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_name = cfg.run_name or f"{stamp}-{cfg.fingerprint()}"
        self.run_dir = Path(cfg.out_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.run_dir / "records.jsonl"

        self.methods = expand_methods(cfg.methods)
        self.records: List[Dict[str, Any]] = []
        self._done_keys = set()
        self._n_items = 0
        self._since_live = 0

        prior_cfg = self.run_dir / "config.json"
        if cfg.resume and prior_cfg.exists() and self.records_path.exists():
            prior = json.loads(prior_cfg.read_text())
            if prior.get("fingerprint") and prior["fingerprint"] != cfg.fingerprint():
                now = cfg.to_dict()
                changed = [k for k in now
                           if k in prior and k != "fingerprint" and prior[k] != now[k]]
                raise SystemExit(
                    f"[runner] refusing to resume '{self.run_name}': the settings "
                    f"changed since those records were written "
                    f"({', '.join(changed) or 'unknown keys'}).\n"
                    f"          Resuming would mix results from two configurations. "
                    f"Use a new --run-name, or --no-resume to start this one over."
                )

        if cfg.resume and self.records_path.exists():
            for line in self.records_path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                self.records.append(r)
                self._done_keys.add(self._key(r["method"], r["seed"], r["record_id"]))
            print(f"[runner] resuming: {len(self.records)} existing records in {self.records_path}")

    @staticmethod
    def _key(method: str, seed: int, record_id: str) -> str:
        return f"{method}|{seed}|{record_id}"

    # -- data ---------------------------------------------------------------
    def load_items(self) -> List[EvalItem]:
        cfg = self.cfg
        full = load_dataset(cfg.dataset, cfg.split, cfg.cache_dir)
        items = segment(full, fraction=cfg.fraction, limit=cfg.limit, offset=cfg.offset,
                        seed=cfg.data_seed, shuffle=cfg.shuffle, stratify_by=cfg.stratify_by,
                        min_per_group=cfg.min_per_group)
        print(f"[data] {cfg.dataset}: {len(full)} items -> segment "
              f"fraction={cfg.fraction} offset={cfg.offset} limit={cfg.limit} "
              f"seed={cfg.data_seed} -> {len(items)} items")
        print(f"[data] {describe_segment(items)}")

        ids = [i.id for i in items]
        if len(set(ids)) != len(ids):
            from collections import Counter
            dupes = [k for k, v in Counter(ids).items() if v > 1]
            raise SystemExit(
                f"[data] {len(ids) - len(set(ids))} duplicate item id(s) in the segment "
                f"(e.g. {dupes[:3]}). Ids are the join key for resume and for paired "
                f"statistics, so duplicates would be silently dropped. This is a loader "
                f"bug - please report the --dataset spec."
            )
        return items

    # -- execution ----------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        items = self.load_items()
        self._n_items = len(items)
        if not items:
            raise SystemExit("segment is empty - check --fraction/--limit/--offset")

        seg_path = self.run_dir / "segment.json"
        if self.cfg.resume and self.records and seg_path.exists():
            prior_ids = json.loads(seg_path.read_text()).get("item_ids", [])
            if prior_ids and prior_ids != [i.id for i in items]:
                raise SystemExit(
                    f"[runner] refusing to resume '{self.run_name}': the segment no longer "
                    f"matches the one those records were written against "
                    f"({len(prior_ids)} items then, {len(items)} now).\n"
                    f"          Appending would double-count. Use a new --run-name."
                )

        self.cfg.save(self.run_dir / "config.json")
        seg_path.write_text(json.dumps(
            {"spec": self.cfg.dataset, **describe_segment(items),
             "item_ids": [i.id for i in items]}, indent=2))

        # Group by backend so a 7B model is loaded once per backend, not per method.
        groups: Dict[str, List[str]] = {}
        for m in self.methods:
            groups.setdefault(backend_for(m), []).append(m)

        t_start = time.time()
        for kind, method_names in groups.items():
            backend = None
            if kind != "none":
                backend = build_backend(kind, self.cfg, self.dry_run)
            try:
                for name in method_names:
                    t0 = time.time()
                    self._run_method(name, backend, items)
                    print(f"[method] {name} finished in {_fmt_dur(time.time() - t0)}")
            finally:
                if backend is not None:
                    backend.free()
                    print(f"[runner] freed backend '{kind}'")

        summary = self.finalize(items, time.time() - t_start)
        return summary

    def _run_method(self, name: str, backend, items: List[EvalItem]) -> None:
        print(f"\n{'=' * 72}\n[method] {name}   args={self.cfg.args_for(name)}\n{'=' * 72}")
        n_variants = len(permutations_for(items[0], self.cfg.permute_options)) if items else 1
        planned = len(items) * n_variants * len(self.cfg.seeds)
        done = correct = 0
        t_method = time.time()
        try:
            method = build_method(name, backend, self.cfg)
        except Exception as e:
            print(f"[method] {name} failed to build: {e}")
            traceback.print_exc()
            return

        for seed in self.cfg.seeds:
            for idx, item in enumerate(items):
                for perm in permutations_for(item, self.cfg.permute_options):
                    variant = item.rendered(perm) if perm else item
                    record_id = item.id if perm is None else f"{item.id}#p{''.join(map(str, perm))}"

                    if self._key(name, seed, record_id) in self._done_keys:
                        continue

                    rec = self._score_one(method, name, seed, variant, item, record_id, perm)
                    self._append(rec)

                    done += 1
                    correct += int(rec["correct"])
                    self._progress(name, seed, rec, done, planned, correct, t_method)

    def _progress(self, name: str, seed: int, rec: Dict[str, Any], done: int,
                  planned: int, correct: int, t_method: float) -> None:
        """
        One line per item by default.

        A multi-agent item takes 10-30s, so a harness that prints every 25 items
        is indistinguishable from a hung process. --progress controls the
        interval; 0 silences it.
        """
        every = self.cfg.progress_every
        if every <= 0:
            return
        if not (self.verbose or done % every == 0 or done == planned):
            return

        elapsed = time.time() - t_method
        per_item = elapsed / max(1, done)
        remaining = max(0, planned - done) * per_item
        mark = "PASS" if rec["correct"] else "FAIL"
        flag = ""
        if rec.get("error"):
            flag = "  ERR"
        elif rec.get("extract_failed"):
            flag = "  no-parse"

        print(f"  [{name} s{seed}] {done:>4}/{planned}  {rec['record_id']:<14} "
              f"{rec['pred']:>7} vs {rec['gold']:<7} {mark}  "
              f"acc {100 * correct / done:5.1f}%  "
              f"{rec['latency_ms'] / 1000:5.1f}s  {rec['total_tokens']:>5} tok  "
              f"eta {_fmt_dur(remaining)}{flag}", flush=True)

    def _score_one(self, method, name: str, seed: int, variant: EvalItem,
                   original: EvalItem, record_id: str, perm) -> Dict[str, Any]:
        gen_base = self.cfg.gen(seed)
        k = max(1, self.cfg.self_consistency)

        votes: List[str] = []
        texts: List[str] = []
        p_tok = c_tok = lat = 0
        rules: List[str] = []
        failed = 0
        extras: List[Dict[str, Any]] = []
        error = None

        for s in range(k):
            try:
                smp = method.sample(variant, gen_base.for_sample(s))
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                if self.verbose:
                    traceback.print_exc()
                smp = None
            if smp is None:
                votes.append(UNKNOWN)
                rules.append("error")
                failed += 1
                continue
            votes.append(smp.pred)
            texts.append(smp.text)
            rules.append(smp.extract_rule)
            failed += int(smp.extract_failed)
            p_tok += smp.prompt_tokens
            c_tok += smp.completion_tokens
            lat += smp.latency_ms
            extras.append(smp.extra)

        pred, counts, agreement = majority_vote(votes)
        gold = method.gold_of(variant)
        task_type = method.task_type_of(variant)
        ok = is_correct(pred, gold, task_type)

        rec: Dict[str, Any] = {
            "run": self.run_name,
            "method": name,
            "seed": seed,
            "record_id": record_id,
            "item_id": original.id,
            "domain": original.domain,
            "source": original.source,
            "task_type": task_type,
            "gold": gold,
            "pred": pred,
            "correct": ok,
            "votes": votes,
            "vote_counts": counts,
            "vote_confidence": round(agreement, 4),
            "self_consistency": k,
            "extract_rule": rules[0] if rules else "?",
            "extract_failed": failed == k,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": p_tok + c_tok,
            "latency_ms": lat,
            "permutation": perm,
            "error": error,
            "extra": extras[0] if extras else {},
        }
        if self.cfg.save_generations:
            rec["generations"] = texts if k > 1 else (texts[0] if texts else "")
        return rec

    def _append(self, rec: Dict[str, Any]) -> None:
        self.records.append(rec)
        self._done_keys.add(self._key(rec["method"], rec["seed"], rec["record_id"]))
        with self.records_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self._maybe_live_plot()

    def _maybe_live_plot(self) -> None:
        """Refresh live.png every --live-every records while the eval runs."""
        if not self.cfg.live_plot:
            return
        self._since_live += 1
        if self._since_live < max(1, self.cfg.live_every):
            return
        self._since_live = 0
        try:
            from .plots import plot_live
            plot_live(self.records, self.run_dir, total=self._n_items or None)
        except Exception as e:
            print(f"[plots] live update failed: {type(e).__name__}: {e}")

    # -- analysis -----------------------------------------------------------
    def finalize(self, items: List[EvalItem], wall_s: float) -> Dict[str, Any]:
        from .report import build_summary, write_reports
        summary = build_summary(self.cfg, self.records, items, wall_s, self.run_name)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        write_reports(summary, self.run_dir, markdown=self.cfg.report_markdown)

        if self.cfg.plots:
            from .plots import make_all, plot_live
            made = make_all(summary, self.run_dir)
            if self.cfg.live_plot:
                plot_live(self.records, self.run_dir, total=self._n_items or None,
                          title="Running accuracy (final)")
                made.append(str(self.run_dir / "live.png"))
            if made:
                print(f"[plots] {len(made)} figures -> {self.run_dir}")
        print(f"\n[runner] results -> {self.run_dir}")
        return summary
