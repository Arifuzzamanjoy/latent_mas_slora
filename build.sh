#!/bin/bash
#
# Build and push Docker image to Docker Hub
# Usage: ./build.sh [dockerhub_username] [tag]
#

set -e

DOCKERHUB_USER=${1:-"s1710374103"}
TAG=${2:-"latest"}
IMAGE_NAME="latent-mas-slora"

echo "========================================"
echo "Building LatentMAS + S-LoRA Serverless Worker"
echo "========================================"
echo "Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo ""

# Build the image
echo "Building Docker image..."
docker build --platform linux/amd64 -t ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG} .

echo ""
echo "Build complete!"
echo ""
echo "Verifying image..."
docker images | grep ${IMAGE_NAME}
echo ""

# Ask to push
read -p "Push to Docker Hub? (y/n): " PUSH

if [ "$PUSH" = "y" ] || [ "$PUSH" = "Y" ]; then
    echo "Logging in to Docker Hub..."
    docker login
    
    echo "Pushing to Docker Hub..."
    docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}
    
    echo ""
    echo "Push complete!"
    echo ""
    echo "========================================"
    echo "Your image is available at:"
    echo "  docker.io/${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
    echo ""
    echo "Use this URL when creating your RunPod endpoint."
    echo "========================================"
else
    echo ""
    echo "Skipping push. To push later, run:"
    echo "  docker login"
    echo "  docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
fi
