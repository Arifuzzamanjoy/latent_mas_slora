# LatentMAS + S-LoRA RunPod Serverless Worker
# Multi-agent reasoning with domain-specific LoRA adapters
# hadolint ignore=DL3006
FROM runpod/base:0.6.3-cuda12.1.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/runpod-volume/.cache/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface \
    HF_HUB_CACHE=/runpod-volume/.cache/huggingface/hub \
    TORCH_HOME=/runpod-volume/.cache/torch \
    SENTENCE_TRANSFORMERS_HOME=/runpod-volume/.cache/sentence_transformers

WORKDIR /app

# Install system dependencies
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for Python 3.11
# hadolint ignore=DL3013,DL3042
RUN python3.11 -m pip install --no-cache-dir --upgrade pip==24.0 setuptools==69.0.3 wheel==0.42.0

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# hadolint ignore=DL3013
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt && \
    python3.11 -m pip install --no-cache-dir runpod==1.7.0 requests==2.31.0 "qwen-vl-utils[decord]"

# Copy source code
COPY src/ /app/src/
COPY examples/ /app/examples/
COPY data/ /app/data/
COPY handler.py /app/handler.py

# Create cache directories
RUN mkdir -p /home/caches/huggingface/hub /home/caches/torch /home/caches/sentence_transformers

# Set local cache fallback (for when volume not mounted)
ENV HF_HOME_LOCAL=/home/caches/huggingface
ENV TORCH_HOME_LOCAL=/home/caches/torch

# Pre-download sentence-transformers embedding model (small, ~90MB)
RUN python3.11 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/home/caches/sentence_transformers')"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3.11 -c "import torch; print(torch.cuda.is_available())" || exit 1

# Start the serverless handler
CMD ["python3.11", "-u", "/app/handler.py"]
