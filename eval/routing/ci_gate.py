#!/usr/bin/env python3
"""
CI gate: fail the build if routing quality regresses.

Reads a run_eval.py results.json and enforces two bounds:
  - top-1 accuracy must be >= --min-accuracy
  - confident-and-wrong rate must be <= --max-confident-wrong

Exits non-zero (fails CI) if either bound is violated. Prints a legible report.
Thresholds are supplied by the workflow so they are visible in the CI config.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/routing/results/staged/results.json")
    ap.add_argument("--min-accuracy", type=float, required=True)
    ap.add_argument("--max-confident-wrong", type=float, required=True)
    args = ap.parse_args()

    p = Path(args.results)
    if not p.exists():
        print(f"[GATE] FAIL: results file not found: {p}")
        sys.exit(2)

    m = json.loads(p.read_text())["metrics"]
    acc = m["overall_accuracy"]
    cw = m["confident_and_wrong_rate"]

    acc_ok = acc >= args.min_accuracy
    cw_ok = cw <= args.max_confident_wrong

    print("=" * 60)
    print("ROUTING CI GATE")
    print("=" * 60)
    print(f"  top-1 accuracy      : {acc:.3f}  (floor {args.min_accuracy:.3f})   "
          f"{'OK' if acc_ok else 'FAIL'}")
    print(f"  confident-and-wrong : {cw:.3f}  (ceiling {args.max_confident_wrong:.3f})  "
          f"{'OK' if cw_ok else 'FAIL'}")
    print("=" * 60)

    if acc_ok and cw_ok:
        print("[GATE] PASS")
        sys.exit(0)
    if not acc_ok:
        print(f"[GATE] FAIL: accuracy {acc:.3f} dropped below floor {args.min_accuracy:.3f}")
    if not cw_ok:
        print(f"[GATE] FAIL: confident-and-wrong {cw:.3f} rose above ceiling "
              f"{args.max_confident_wrong:.3f}")
    sys.exit(1)


if __name__ == "__main__":
    main()
