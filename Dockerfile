# ═══════════════════════════════════════════════════════════════════════
#  RapidOCR TensorRT HTTP Service
#  基于 RapidAI/RapidOCR Dockerfile.tensorrt + Gunicorn 高并发
#  模型: ch_PP-OCRv4 (det + rec + cls) via TensorRT
# ═══════════════════════════════════════════════════════════════════════

FROM nvcr.io/nvidia/deepstream:7.0-gc-triton-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# ── 系统依赖（沿用 RapidOCR 原始构建） ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    g++ \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ── RapidOCR 核心依赖 ────────────────────────────────────────────────
COPY RapidOCR/python/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip uninstall -y opencv-python 2>/dev/null; \
    pip install --no-cache-dir opencv-python-headless \
        -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    rm /tmp/requirements.txt

# ── TensorRT Python bindings (匹配 DeepStream 7.0 内置 TRT 8.6.1) ──
RUN pip install --no-cache-dir "tensorrt>=8.6,<8.7" "cuda-python>=12.0,<13.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── HTTP 服务额外依赖 ────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] gunicorn python-multipart onnxruntime \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 复制 RapidOCR 源码 ──────────────────────────────────────────────
COPY RapidOCR/python/ /app/

# ── TensorRT 配置（覆盖默认 onnxruntime 引擎） ──────────────────────
COPY config_tensorrt.yaml /app/rapidocr/config_tensorrt.yaml

# ── 构建阶段预下载 ONNX 模型（TensorRT 从 ONNX 构建引擎） ──────────
COPY download_models.py /app/download_models.py
RUN python3 download_models.py

# ── HTTP 服务代码 ────────────────────────────────────────────────────
COPY app/ /app/app/
COPY gunicorn_conf.py /app/gunicorn_conf.py

# ── 服务配置 ──────────────────────────────────────────────────────────
ENV OCR_SERVICE_PORT=8089
EXPOSE 8089

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8089/health')" || exit 1

# ── Gunicorn + UvicornWorker 高并发 ──────────────────────────────────
CMD ["gunicorn", "app.main:app", "--config", "gunicorn_conf.py"]