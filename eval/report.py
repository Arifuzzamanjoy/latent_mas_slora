"""
Summaries and reports.

Console output is a table; the markdown report is the artifact you keep. Both
are generated from records.jsonl, so a report can be regenerated for an old run
without re-running the model (`--report-only <run_dir>`).
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EvalConfig
from .data import EvalItem, describe_segment
from .metrics import (
    classification_report, mcnemar, paired_bootstrap_delta, summarize_records,
)


def _position_consistency(records: List[Dict[str, Any]]) -> Optional[float]:
    """Fraction of items whose prediction is invariant under option permutation."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if r.get("permutation"):
            groups[r["item_id"]].append(r)
    if not groups:
        return None
    stable = 0
    for _, rows in groups.items():
        # map each prediction back to the original option index
        mapped = set()
        for r in rows:
            perm = r["permutation"]
            pred = r["pred"]
            if pred == "UNKNOWN" or not perm:
                mapped.add(pred)
                continue
            try:
                letter_idx = "ABCDEFGHIJ".index(pred)
                mapped.add(perm[letter_idx])
            except (ValueError, IndexError):
                mapped.add(pred)
        stable += int(len(mapped) == 1)
    return round(stable / len(groups), 4)


def build_summary(cfg: EvalConfig, records: List[Dict[str, Any]],
                  items: List[EvalItem], wall_s: float, run_name: str) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_method[r["method"]].append(r)

    methods: Dict[str, Any] = {}
    for name, recs in by_method.items():
        if name == "router-only":
            preds = [r["pred"] for r in recs]
            golds = [r["gold"] for r in recs]
            methods[name] = {
                "n": len(recs),
                "routing": classification_report(preds, golds),
                "latency_ms": {"mean": round(sum(r["latency_ms"] for r in recs) / max(1, len(recs)), 2)},
                "accuracy": classification_report(preds, golds)["accuracy"],
            }
            continue

        block = summarize_records(recs, cfg.bootstrap, cfg.ci, cfg.data_seed)
        block["position_consistency"] = _position_consistency(recs)
        # per-seed accuracy shows run-to-run variance, which is often larger
        # than the difference being claimed
        per_seed = {}
        for seed in sorted({r["seed"] for r in recs}):
            rows = [r for r in recs if r["seed"] == seed]
            per_seed[str(seed)] = round(sum(r["correct"] for r in rows) / max(1, len(rows)), 4)
        block["per_seed_accuracy"] = per_seed
        block["errors"] = sum(1 for r in recs if r.get("error"))
        methods[name] = block

    # Paired comparisons against the reference method
    ref = cfg.compare_to
    if ref is None:
        for candidate in ("baseline-cot", "baseline-judger", "baseline-direct"):
            if candidate in by_method:
                ref = candidate
                break
    comparisons = {}
    if ref and ref in by_method:
        ref_map = {(r["seed"], r["record_id"]): bool(r["correct"]) for r in by_method[ref]}
        for name, recs in by_method.items():
            if name in (ref, "router-only"):
                continue
            pairs = [(bool(r["correct"]), ref_map[(r["seed"], r["record_id"])])
                     for r in recs if (r["seed"], r["record_id"]) in ref_map]
            if not pairs:
                continue
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs]
            comparisons[name] = {
                "vs": ref,
                "n_paired": len(pairs),
                "accuracy": round(sum(a) / len(a), 4),
                "reference_accuracy": round(sum(b) / len(b), 4),
                **paired_bootstrap_delta(a, b, cfg.bootstrap, cfg.ci, cfg.data_seed),
                "mcnemar": mcnemar(a, b),
            }

    return {
        "run": run_name,
        "config": cfg.to_dict(),
        "segment": {"spec": cfg.dataset, **describe_segment(items)},
        "wall_clock_s": round(wall_s, 1),
        "methods": methods,
        "reference_method": ref,
        "comparisons": comparisons,
    }


# ─── Rendering ───────────────────────────────────────────────────────────────

def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render_console(summary: Dict[str, Any]) -> str:
    seg = summary["segment"]
    cfg = summary["config"]
    lines = [
        "",
        "=" * 96,
        f"EVAL {summary['run']}",
        "=" * 96,
        f"model      : {cfg['model']} ({cfg['dtype']})",
        f"dataset    : {seg['spec']}  n={seg['n']}  fraction={cfg['fraction']}  "
        f"limit={cfg['limit']}  offset={cfg['offset']}  data_seed={cfg['data_seed']}",
        f"domains    : {seg['by_domain']}",
        f"decoding   : temp={cfg['temperature']} top_p={cfg['top_p']} "
        f"max_new_tokens={cfg['max_new_tokens']} seeds={cfg['seeds']} sc={cfg['self_consistency']}",
        f"protocol   : scoring={cfg['scoring']} permute={cfg['permute_options']} "
        f"latent_steps={cfg['latent_steps']} router={cfg['use_router']}",
        f"wall clock : {summary['wall_clock_s']}s",
        "",
        f"{'method':<18} {'n':>5} {'acc':>7} {'95% CI':>16} {'parse fail':>11} "
        f"{'tok/item':>9} {'tok/correct':>12} {'p50 ms':>8} {'ECE':>6}",
        "-" * 96,
    ]

    for name, m in summary["methods"].items():
        if name == "router-only":
            r = m["routing"]
            lines.append(f"{name:<18} {m['n']:>5} {_pct(r['accuracy']):>7} "
                         f"{'macro-F1 ' + str(r['macro_f1']):>16}")
            continue
        ci = f"[{_pct(m['ci']['low'])}, {_pct(m['ci']['high'])}]"
        ece = m["calibration"].get("ece")
        lines.append(
            f"{name:<18} {m['n']:>5} {_pct(m['accuracy']):>7} {ci:>16} "
            f"{_pct(m['parse_failure_rate']):>11} {m['tokens']['total_mean']:>9.0f} "
            f"{str(m['tokens']['total_per_correct']):>12} {m['latency_ms']['p50']:>8.0f} "
            f"{(f'{ece:.3f}' if ece is not None else '-'):>6}"
        )

    if summary["comparisons"]:
        ref = summary["reference_method"]
        lines += ["", f"Paired comparisons vs {ref} (McNemar, same items):",
                  "-" * 96,
                  f"{'method':<18} {'Δacc':>8} {'95% CI of Δ':>20} {'wins':>6} {'losses':>7} {'p':>10}"]
        for name, c in summary["comparisons"].items():
            d = f"{100 * c['delta']:+.1f}%"
            ci = f"[{100 * c['ci_low']:+.1f}%, {100 * c['ci_high']:+.1f}%]"
            mc = c["mcnemar"]
            lines.append(f"{name:<18} {d:>8} {ci:>20} {mc['a_only']:>6} {mc['b_only']:>7} "
                         f"{mc['p_value']:>10.4f}")

    lines.append("=" * 96)
    return "\n".join(lines)


def render_markdown(summary: Dict[str, Any]) -> str:
    cfg = summary["config"]
    seg = summary["segment"]
    L: List[str] = []
    L.append(f"# Eval report — `{summary['run']}`\n")
    L.append("## Configuration\n")
    L.append("| setting | value |")
    L.append("|---|---|")
    for k in ("model", "dtype", "device", "dataset", "split", "fraction", "limit", "offset",
              "data_seed", "shuffle", "stratify_by", "temperature", "top_p", "max_new_tokens",
              "seeds", "self_consistency", "scoring", "permute_options", "latent_steps",
              "agents", "use_router", "adaptive_latent_steps", "kv_handoff", "loras",
              "bootstrap", "ci", "fingerprint"):
        if k in cfg:
            L.append(f"| `{k}` | `{cfg[k]}` |")
    L.append(f"\nSegment: **n={seg['n']}**, domains `{seg['by_domain']}`, "
             f"sources `{seg['by_source']}`. Wall clock {summary['wall_clock_s']}s.\n")

    L.append("## Results\n")
    L.append("| method | n | accuracy | 95% CI | parse fail | tokens/item | tokens/correct | "
             "latency p50 | latency p95 | ECE |")
    L.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for name, m in summary["methods"].items():
        if name == "router-only":
            r = m["routing"]
            L.append(f"| `{name}` | {m['n']} | {_pct(r['accuracy'])} | macro-F1 {r['macro_f1']} "
                     f"| - | - | - | {m['latency_ms']['mean']:.0f} | - | - |")
            continue
        ece = m["calibration"].get("ece")
        L.append(
            f"| `{name}` | {m['n']} | **{_pct(m['accuracy'])}** | "
            f"[{_pct(m['ci']['low'])}, {_pct(m['ci']['high'])}] | {_pct(m['parse_failure_rate'])} | "
            f"{m['tokens']['total_mean']:.0f} | {m['tokens']['total_per_correct']} | "
            f"{m['latency_ms']['p50']:.0f} | {m['latency_ms']['p95']:.0f} | "
            f"{ece if ece is not None else '-'} |"
        )

    if summary["comparisons"]:
        L.append(f"\n## Paired comparisons vs `{summary['reference_method']}`\n")
        L.append("McNemar's exact test on the items both methods answered. "
                 "`wins` = correct here and wrong for the reference.\n")
        L.append("| method | Δ accuracy | 95% CI of Δ | wins | losses | discordant | p |")
        L.append("|---|---:|---|---:|---:|---:|---:|")
        for name, c in summary["comparisons"].items():
            mc = c["mcnemar"]
            L.append(f"| `{name}` | {100 * c['delta']:+.1f}% | "
                     f"[{100 * c['ci_low']:+.1f}%, {100 * c['ci_high']:+.1f}%] | "
                     f"{mc['a_only']} | {mc['b_only']} | {mc['discordant']} | {mc['p_value']} |")

    L.append("\n## Per-domain accuracy\n")
    L.append("| method | " + " | ".join(sorted(seg["by_domain"])) + " |")
    L.append("|---|" + "---:|" * len(seg["by_domain"]))
    for name, m in summary["methods"].items():
        if "by_domain" not in m:
            continue
        cells = []
        for dom in sorted(seg["by_domain"]):
            d = m["by_domain"].get(dom)
            cells.append(f"{_pct(d['accuracy'])} ({d['n']})" if d else "-")
        L.append(f"| `{name}` | " + " | ".join(cells) + " |")

    L.append("\n## Robustness\n")
    L.append("| method | per-seed accuracy | position consistency | unknown rate | errors |")
    L.append("|---|---|---:|---:|---:|")
    for name, m in summary["methods"].items():
        if "per_seed_accuracy" not in m:
            continue
        L.append(f"| `{name}` | `{m['per_seed_accuracy']}` | "
                 f"{_pct(m.get('position_consistency'))} | {_pct(m.get('unknown_rate'))} | "
                 f"{m.get('errors', 0)} |")

    if "router-only" in summary["methods"]:
        r = summary["methods"]["router-only"]["routing"]
        L.append("\n## Router confusion matrix\n")
        L.append("| gold \\ pred | " + " | ".join(r["labels"]) + " |")
        L.append("|---|" + "---:|" * len(r["labels"]))
        for g in r["labels"]:
            L.append(f"| **{g}** | " + " | ".join(str(r["confusion"][g][p]) for p in r["labels"]) + " |")

    L.append("\n---\n")
    L.append("Records: `records.jsonl` (one row per item × method × seed × permutation). "
             "Config: `config.json`. Regenerate this report with "
             "`python run_eval.py --report-only <run_dir>`.\n")
    return "\n".join(L)


def write_reports(summary: Dict[str, Any], run_dir: Path, markdown: bool = True,
                  plots: bool = False) -> None:
    console = render_console(summary)
    print(console)
    (run_dir / "report.txt").write_text(console)
    if markdown:
        (run_dir / "report.md").write_text(render_markdown(summary))
    if plots:
        from .plots import make_all
        made = make_all(summary, run_dir)
        if made:
            print(f"[plots] {len(made)} figures -> {run_dir}")


def report_only(run_dir: str) -> Dict[str, Any]:
    """Rebuild the summary and reports from an existing run directory."""
    d = Path(run_dir)
    cfg = EvalConfig.from_file(str(d / "config.json"))
    records = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if l.strip()]
    seg = json.loads((d / "segment.json").read_text())
    items = [EvalItem(id=i, question="", gold="", domain=dom)
             for dom, n in seg["by_domain"].items() for i in [f"{dom}-{k}" for k in range(n)]]
    summary = build_summary(cfg, records, items, 0.0, d.name)
    summary["segment"] = {"spec": seg.get("spec", cfg.dataset), **{k: v for k, v in seg.items() if k != "item_ids"}}
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # --report-only regenerates everything derived from records.jsonl,
    # figures included - that is what "derived" is supposed to mean
    write_reports(summary, d, markdown=True, plots=True)
    return summary
