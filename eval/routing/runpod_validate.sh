#!/usr/bin/env bash
#
# RunPod full-stack routing validation (GPU REQUIRED).
#
# Proves the real path works on real hardware: route -> load/swap LoRA adapter ->
# generate, with the actual Qwen2.5-VL-7B base and the four registry adapters.
# This is the GPU counterpart to the CPU-only CI routing eval; it is NOT run in CI.
#
# It NEVER fabricates results. It writes a timestamped evidence pack and fails
# loudly (non-zero exit) on OOM or adapter-load failure.
#
# Usage (from the repo root on a RunPod GPU pod):
#     bash eval/routing/runpod_validate.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
EVID="eval/routing/results/runpod/${TS}"
mkdir -p "$EVID"
RAW="$EVID/raw.log"

echo "== RunPod full-stack validation =="       | tee -a "$RAW"
echo "repo   : $REPO_ROOT"                        | tee -a "$RAW"
echo "evidence: $EVID"                            | tee -a "$RAW"
echo "utc    : $TS"                               | tee -a "$RAW"

fail() { echo "[VALIDATION FAILED] $*" | tee -a "$RAW" >&2; exit 1; }
trap 'echo "[trap] non-zero exit ($?). Evidence preserved in $EVID" | tee -a "$RAW" >&2' ERR

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found -- is this a GPU pod?"

# environment snapshot
{
  echo "### uname"; uname -a
  echo "### python"; python3 --version
  echo "### pip (torch/transformers/peft)"; pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|accelerate|qwen)' || true
} > "$EVID/environment.txt" 2>&1

# nvidia-smi BEFORE
nvidia-smi > "$EVID/nvidia-smi.before.txt" 2>&1 || fail "nvidia-smi (before) failed"
echo "[ok] captured nvidia-smi.before.txt" | tee -a "$RAW"

# run the driver (captures nvidia-smi.during, metrics.json, summary.md itself).
# tee the driver output into the raw log; preserve the driver's exit code.
set +e
python3 eval/routing/runpod_driver.py "$EVID" 2>&1 | tee -a "$RAW"
CODE=${PIPESTATUS[0]}
set -e

# nvidia-smi AFTER (always, even on failure)
nvidia-smi > "$EVID/nvidia-smi.after.txt" 2>&1 || true
echo "[ok] captured nvidia-smi.after.txt" | tee -a "$RAW"

case "$CODE" in
  0) echo "[VALIDATION PASSED] evidence: $EVID" | tee -a "$RAW" ;;
  1) echo "[VALIDATION WARN] ran, but not every query produced output. See $EVID/metrics.json" | tee -a "$RAW" ;;
  3) fail "CUDA OUT OF MEMORY. Use a larger-VRAM GPU or reduce load. See $EVID/raw.log" ;;
  4) fail "ADAPTER LOAD/SWAP FAILURE. See $EVID/raw.log and metrics.json" ;;
  5) fail "BASE MODEL / ENVIRONMENT FAILURE. See $EVID/raw.log" ;;
  6) fail "NO GPU AVAILABLE." ;;
  *) fail "driver exited with unexpected code $CODE" ;;
esac

exit "$CODE"
