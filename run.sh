#!/bin/bash
# Run LatentMAS + S-LoRA System
# Optimized for 24-48GB VRAM

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "/workspace/LatentMAS/venv" ]; then
    source /workspace/LatentMAS/venv/bin/activate
fi

# Check dependencies
python -c "import torch; import transformers; import peft" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Parse arguments
EXAMPLE="${1:-quickstart}"

echo "=============================================="
echo "LatentMAS + S-LoRA Multi-Agent System"
echo "=============================================="
echo "Example: $EXAMPLE"
echo ""

case "$EXAMPLE" in
    quickstart)
        python examples/quickstart.py
        ;;
    medical)
        python examples/medical_qa.py
        ;;
    custom)
        python examples/custom_pipeline.py
        ;;
    *)
        echo "Usage: ./run.sh [quickstart|medical|custom]"
        exit 1
        ;;
esac
