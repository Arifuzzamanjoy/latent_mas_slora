# GitHub Actions Workflows

This directory contains CI/CD workflows for automated testing and deployment.

## Workflows

### 1. CI - Test & Lint (`ci.yml`)
**Trigger:** Push to `main`/`develop`, Pull Requests to `main`

Runs:
- Python syntax validation
- Flake8 linting
- Import tests
- Dockerfile linting

### 2. CD - Docker Build & Push (`cd-docker.yml`)
**Trigger:** Push to `main`, Version tags (`v*`), Manual dispatch

Builds and pushes Docker image to:
- **Docker Hub:** `s1710374103/latent-mas-slora:latest`
- **GitHub Container Registry:** `ghcr.io/arifuzzamanjoy/latent-mas-slora:latest`

## Setup Instructions

### 1. Fork the Repository
Click "Fork" at the top right of the GitHub page.

### 2. Configure Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `DOCKERHUB_TOKEN` | Docker Hub access token | Yes |
| `HF_TOKEN` | Hugging Face token (for gated models) | Optional |

#### How to get Docker Hub token:
1. Go to https://hub.docker.com/settings/security
2. Click "New Access Token"
3. Name it "GitHub Actions"
4. Copy the token and add as `DOCKERHUB_TOKEN` secret

### 3. Enable GitHub Actions
1. Go to **Actions** tab in your repository
2. Click "I understand my workflows, go ahead and enable them"

### 4. Trigger a Build

**Option A: Push to main**
```bash
git add .
git commit -m "Trigger CI/CD"
git push origin main
```

**Option B: Create a release tag**
```bash
git tag v1.0.0
git push origin v1.0.0
```

**Option C: Manual trigger**
1. Go to **Actions** tab
2. Select "CD - Docker Build & Push"
3. Click "Run workflow"

## Docker Image URLs

After successful build:

```
# Docker Hub
docker.io/s1710374103/latent-mas-slora:latest

# GitHub Container Registry
ghcr.io/arifuzzamanjoy/latent-mas-slora:latest
```

## Deploying to RunPod

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **"New Endpoint"**
3. Choose **"Custom"** template
4. Enter Docker image URL: `docker.io/s1710374103/latent-mas-slora:latest`
5. Configure:
   - **GPU Type:** A5000/A6000/A100 (24GB+ VRAM recommended)
   - **Container Disk:** 30GB
   - **Idle Timeout:** 30 seconds
   - **Max Workers:** 3
6. Add environment variables:
   - `HF_TOKEN`: Your Hugging Face token (optional)
7. Click **"Deploy"**

## Testing Locally

```bash
# Build image locally
docker build -t latent-mas-slora .

# Run with GPU
docker run --gpus all -p 8000:8000 latent-mas-slora

# Test the handler
python handler.py --test_input test_input.json
```

## API Usage

### Input Schema
```json
{
    "input": {
        "prompt": "What is the treatment for hypertension?",
        "rag_data": "https://example.com/data.json",
        "rag_documents": ["doc1 content", "doc2 content"],
        "system_prompt": "You are a medical expert",
        "conversation_id": "uuid-to-continue",
        "max_tokens": 800,
        "temperature": 0.7,
        "enable_tools": false,
        "model": "Qwen/Qwen2.5-3B-Instruct"
    }
}
```

### Output Schema
```json
{
    "response": "AI response text",
    "conversation_id": "uuid",
    "domain": "medical",
    "domain_confidence": 0.85,
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "rag_enabled": true,
    "tools_enabled": false
}
```

### Example cURL Request
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
        "prompt": "What is hypertension?"
    }
  }'
```
