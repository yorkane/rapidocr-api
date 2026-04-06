# ═══════════════════════════════════════════════════════════════════════
#  RapidOCR TensorRT HTTP Service
#  极简压缩版 - 替换了庞大的 DeepStream 基础镜像 (~30GB -> ~2.5GB)
#  基于 nvidia/cuda:12.2.2-runtime-ubuntu22.04 + Gunicorn 持续并发
#  模型: ch_PP-OCRv4 (det + rec + cls) via TensorRT
# ═══════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# ── 1. 合并执行 apt 和 pip，大幅度压缩镜像层和无用缓存 ──
COPY RapidOCR/python/requirements.txt /tmp/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip uninstall -y opencv-python 2>/dev/null || true \
    && pip install --no-cache-dir opencv-python-headless fastapi uvicorn[standard] \
        gunicorn python-multipart onnxruntime "tensorrt>=8.6,<8.7" "cuda-python>=12.0,<13.0" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/requirements.txt ~/.cache/pip

# ── 2. 复制源码与代码 ────────────────────────────────────────────────
COPY RapidOCR/python/ /app/
COPY config_tensorrt.yaml /app/rapidocr/config_tensorrt.yaml
COPY download_models.py /app/download_models.py
COPY app/ /app/app/
COPY gunicorn_conf.py /app/gunicorn_conf.py

# ── 3. 构建阶段预下载 ONNX 模型 ────────────────────────────────────
RUN python3 download_models.py

# ── 4. 服务配置 ────────────────────────────────────────────────────────
ENV OCR_SERVICE_PORT=8089
EXPOSE 8089

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8089/health')" || exit 1

# ── Gunicorn + UvicornWorker 高并发 ──────────────────────────────────
CMD ["gunicorn", "app.main:app", "--config", "gunicorn_conf.py"]