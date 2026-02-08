#!/bin/bash
#
# Run full evaluation comparison between Traditional 4-Agent RAG and LatentMAS-SLoRA
#
# Usage:
#   ./run_evaluation.sh                    # Run MIRAGE benchmark (recommended)
#   ./run_evaluation.sh mirage             # Run MIRAGE medical QA benchmark
#   ./run_evaluation.sh compare            # Run simple comparison
#   ./run_evaluation.sh traditional        # Run only traditional 4-agent RAG
#   ./run_evaluation.sh latentmas          # Run only LatentMAS
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Activate virtual environment if exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

cd "$SCRIPT_DIR"

# Default model
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DEVICE="${DEVICE:-cuda}"
NUM_QUESTIONS="${NUM_QUESTIONS:-25}"

echo "========================================"
echo "LatentMAS-SLoRA Evaluation Suite"
echo "========================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Working dir: $SCRIPT_DIR"
echo "========================================"
echo ""

# Install required packages
install_deps() {
    echo "Installing evaluation dependencies..."
    pip install rouge-score bert-score matplotlib scikit-learn -q
    echo "✓ Dependencies installed"
}

run_mirage() {
    echo "Running MIRAGE Medical QA Benchmark..."
    echo "Evaluation: Traditional 4-Agent RAG vs LatentMAS-SLoRA"
    echo "Metrics: ROUGE + BERTScore + Accuracy"
    echo ""
    python run_mirage_evaluation.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --num-questions "$NUM_QUESTIONS" \
        --output "mirage_evaluation_results.json"
}

run_mirage_fast() {
    echo "Running MIRAGE Benchmark (Fast mode - no BERTScore)..."
    python run_mirage_evaluation.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --num-questions "$NUM_QUESTIONS" \
        --skip-bertscore \
        --output "mirage_evaluation_results.json"
}

run_traditional() {
    echo "Running Traditional 4-Agent RAG Evaluation..."
    python traditional_4agent_rag.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output "traditional_4agent_results.json"
}

run_latentmas() {
    echo "Running Full LatentMAS-SLoRA Evaluation..."
    python full_latent_mas.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --latent-steps 10 \
        --output "latent_mas_results.json"
}

run_compare() {
    echo "Running Simple Comparison..."
    python compare_systems.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output "comparison_results.json"
}

# Parse command
case "${1:-mirage}" in
    deps|install)
        install_deps
        ;;
    mirage)
        install_deps
        run_mirage
        ;;
    mirage-fast)
        install_deps
        run_mirage_fast
        ;;
    traditional)
        run_traditional
        ;;
    latentmas)
        run_latentmas
        ;;
    compare)
        run_compare
        ;;
    all)
        install_deps
        run_mirage
        ;;
    *)
        echo "Usage: $0 [mirage|mirage-fast|traditional|latentmas|compare|deps]"
        echo ""
        echo "Commands:"
        echo "  mirage       - Run MIRAGE medical benchmark with ROUGE+BERTScore (recommended)"
        echo "  mirage-fast  - Run MIRAGE benchmark without BERTScore (faster)"
        echo "  traditional  - Run only traditional 4-agent RAG"
        echo "  latentmas    - Run only LatentMAS-SLoRA system"
        echo "  compare      - Run simple comparison"
        echo "  deps         - Install required packages only"
        echo ""
        echo "Environment variables:"
        echo "  MODEL         - Model name (default: Qwen/Qwen2.5-3B-Instruct)"
        echo "  DEVICE        - Device (default: cuda)"
        echo "  NUM_QUESTIONS - Number of questions (default: 25)"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - mirage_evaluation_results.json (full results)"
echo "  - evaluation_comparison.png (plots)"
echo "  - evaluation_comparison.pdf (high-quality plots)"
echo "========================================"
