"""Gunicorn 生产配置 — Uvicorn Worker 高并发

TensorRT GPU 推理的并发策略：
  - Worker 数量受限于 GPU 显存（每个 worker 加载一份 TRT engine）
  - TensorRT engine 占用显存约 200-500MB
  - RTX 4090 24GB → 建议 2-4 workers
"""

import os

# ── 绑定地址 ──────────────────────────────────────────────────────────
bind = f"0.0.0.0:{os.environ.get('OCR_SERVICE_PORT', '8089')}"

# ── Worker 配置 ───────────────────────────────────────────────────────
workers = int(os.environ.get("OCR_WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"
threads = int(os.environ.get("OCR_THREADS", "4"))

# ── 超时 ──────────────────────────────────────────────────────────────
# TensorRT 首次构建 engine 非常慢（可达数分钟），超时设长
timeout = 600
graceful_timeout = 30
keepalive = 5

# ── 内存管理 ──────────────────────────────────────────────────────────
max_requests = int(os.environ.get("OCR_MAX_REQUESTS", "1000"))
max_requests_jitter = 50

# ── 日志 ──────────────────────────────────────────────────────────────
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# ── 应用加载 ──────────────────────────────────────────────────────────
preload_app = False
