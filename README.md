# RapidOCR HTTP Service (v2)

基于 **RapidOCR 源码 + DeepStream 7.0 基础镜像** 的高并发 OCR HTTP 服务。

## 架构

| 项目 | 说明 |
|------|------|
| OCR 引擎 | RapidOCR (ONNX Runtime) |
| 模型 | **ch_PP-OCRv4** (det + cls + rec) |
| 基础镜像 | `nvcr.io/nvidia/deepstream:7.0-gc-triton-devel` |
| 构建参考 | `RapidOCR/docker/Dockerfile.tensorrt` |
| HTTP 服务 | **Gunicorn + UvicornWorker** 多进程高并发 |
| API 框架 | FastAPI |

## 特性

- 🚀 **Gunicorn 多 Worker 进程** — 并发处理 OCR 请求
- 📦 **开箱即用** — ONNX 模型在构建阶段预下载到镜像
- 📖 **Swagger 文档** — `http://localhost:8089/docs`
- 🎮 **GPU 就绪** — DeepStream + CUDA + TensorRT 基础设施

## 快速开始

```bash
# 构建镜像
sudo docker build -t ocr-service:latest .

# 启动（GPU + 2 workers）
sudo docker run -d --name ocr-service --gpus all \
  -p 8089:8089 \
  -e OCR_WORKERS=2 \
  ocr-service:latest

# 或使用 docker-compose
sudo docker compose up -d
```

## API 端点

### `GET /health`

```bash
curl http://localhost:8089/health
```

### `POST /ocr` — 完整 OCR

```bash
curl -X POST http://localhost:8089/ocr -F "file=@test.jpg"
```

返回:
```json
{
  "results": [
    {"text": "识别文本", "confidence": 0.98, "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
  ],
  "total": 1,
  "device": "onnxruntime",
  "elapse": {"det": 0.05, "cls": 0.01, "rec": 0.03}
}
```

### `POST /ocr/region` — 区域 OCR

```bash
curl -X POST http://localhost:8089/ocr/region \
  -F "file=@frame.jpg" \
  -F "top=0.65" -F "bottom=1.0" \
  -F "left=0.125" -F "right=0.875"
```

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OCR_SERVICE_PORT` | `8089` | 服务端口 |
| `OCR_WORKERS` | `2` | Gunicorn Worker 数 |
| `OCR_THREADS` | `4` | 每 Worker 线程数 |
| `OCR_MAX_REQUESTS` | `1000` | Worker 自动回收阈值 |
| `LOG_LEVEL` | `info` | 日志级别 |

## 项目结构

```
ocr-service/
├── Dockerfile           # 基于 Dockerfile.tensorrt 构建方式
├── docker-compose.yml   # GPU 服务编排
├── gunicorn_conf.py     # Gunicorn 高并发配置
├── download_models.py   # 构建阶段模型下载
├── config_tensorrt.yaml # TensorRT 配置（备用）
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 入口
│   └── ocr_engine.py    # OCR 引擎封装
└── RapidOCR/            # RapidOCR 源码 (git clone)
    └── python/          # 核心 Python 模块
```
