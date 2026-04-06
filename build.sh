#!/bin/bash
set -e

# Build local images identical to GitHub Actions setup

echo "🚀 开始构建 GPU 镜像 (基于 TensorRT & CUDA)..."
docker build -t ghcr.io/yorkane/rapidocr-api:latest -f Dockerfile .

echo "--------------------------------------------------------"

echo "🚀 开始构建 CPU 镜像 (基于 onnxruntime + Python Slim)..."
docker build -t ghcr.io/yorkane/rapidocr-api:cpu -f Dockerfile.cpu .

echo "✅ 本地全套镜像构建完成！"
echo ""
echo "如需推送到远端镜像仓库 (ghcr.io)，请运行:"
echo "docker push ghcr.io/yorkane/rapidocr-api:latest"
echo "docker push ghcr.io/yorkane/rapidocr-api:cpu"
