# OCR Docker 高并发服务 — 技术文档

> 基于 RapidOCR 源码，使用 ch_PP-OCRv4 模型，Gunicorn 多 Worker 架构处理高并发 OCR 请求。
> 提供 **GPU 版** (DeepStream 7.0) 和 **CPU 版** (python:3.10-slim) 两种镜像。

---

## 1. 技术架构

### 1.1 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Container                       │
│  Base: nvcr.io/nvidia/deepstream:7.0-gc-triton-devel        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Gunicorn Master                     │  │
│  │              (进程管理 + 负载均衡)                     │  │
│  └──────┬─────────────────┬─────────────────┬────────────┘  │
│         │                 │                 │               │
│  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐       │
│  │  Worker #1  │   │  Worker #2  │   │  Worker #N  │       │
│  │ UvicornWorker│  │ UvicornWorker│  │ UvicornWorker│       │
│  │  (FastAPI)  │   │  (FastAPI)  │   │  (FastAPI)  │       │
│  │             │   │             │   │             │       │
│  │ ┌─────────┐ │   │ ┌─────────┐ │   │ ┌─────────┐ │       │
│  │ │RapidOCR │ │   │ │RapidOCR │ │   │ │RapidOCR │ │       │
│  │ │ Engine  │ │   │ │ Engine  │ │   │ │ Engine  │ │       │
│  │ └────┬────┘ │   │ └────┬────┘ │   │ └────┬────┘ │       │
│  └──────┼──────┘   └──────┼──────┘   └──────┼──────┘       │
│         │                 │                 │               │
│  ┌──────▼─────────────────▼─────────────────▼────────────┐  │
│  │              ONNX Runtime (推理引擎)                   │  │
│  │                                                       │  │
│  │  ┌──────────────┐  ┌────────────┐  ┌───────────────┐  │  │
│  │  │  det (4.5MB) │  │ cls (0.6MB)│  │  rec (10.4MB) │  │  │
│  │  │ PP-OCRv4 Det │  │ mobile cls │  │ PP-OCRv4 Rec  │  │  │
│  │  └──────────────┘  └────────────┘  └───────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Port: 8089   │   Models: /app/rapidocr/models/             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 两种镜像版本对比

| 维度 | 🖥️ GPU 版 (`Dockerfile`) | 💻 CPU 版 (`Dockerfile.cpu`) |
|------|--------------------------|------------------------------|
| **基础镜像** | `nvcr.io/nvidia/deepstream:7.0-gc-triton-devel` | `python:3.10-slim-bookworm` |
| **镜像大小** | **~27GB** | **~1GB** |
| **构建耗时** | ~15-20 分钟（首次拉镜像） | ~2-3 分钟 |
| **推理后端** | ONNX Runtime（可切换 TensorRT） | ONNX Runtime (CPU) |
| **硬件需求** | NVIDIA GPU + Driver + Container Toolkit | 无特殊要求 |
| **适用场景** | 高吞吐生产环境 | 开发测试 / 无 GPU 服务器 / 边缘部署 |
| **单次推理速度** | ~0.35s | ~0.39s |
| **推荐 Workers** | 2-4（受限于 GPU 显存） | 4-8（受限于 CPU 核心数） |

### 1.3 组件选型

| 组件 | 选型 | 版本/说明 |
|------|------|-----------|
| **OCR 引擎** | [RapidOCR](https://github.com/RapidAI/RapidOCR) | 源码集成，支持多引擎后端 |
| **推理后端** | ONNX Runtime | CPU/GPU 通用，模型加载快 |
| **检测模型** | `ch_PP-OCRv4_det_infer.onnx` | 4.5MB，PP-OCRv4 中文检测 |
| **分类模型** | `ch_ppocr_mobile_v2.0_cls_infer.onnx` | 0.6MB，文本方向分类 |
| **识别模型** | `ch_PP-OCRv4_rec_infer.onnx` | 10.4MB，PP-OCRv4 中文识别 |
| **HTTP 框架** | FastAPI | 异步，自动生成 OpenAPI 文档 |
| **进程管理** | Gunicorn + UvicornWorker | 多 Worker 高并发 |

### 1.4 Dockerfile 构建参考

| 版本 | 参照 RapidOCR 官方文件 |
|------|------------------------|
| GPU 版 `Dockerfile` | [`docker/Dockerfile.tensorrt`](https://github.com/RapidAI/RapidOCR/blob/main/docker/Dockerfile.tensorrt) |
| CPU 版 `Dockerfile.cpu` | [`docker/Dockerfile.base`](https://github.com/RapidAI/RapidOCR/blob/main/docker/Dockerfile.base) + [`docker/Dockerfile.onnxruntime-cpu`](https://github.com/RapidAI/RapidOCR/blob/main/docker/Dockerfile.onnxruntime-cpu) |

在官方基础上均增加了：

1. HTTP 服务层（FastAPI + Gunicorn）
2. 构建阶段模型预下载（开箱即用）
3. 健康检查

---

## 2. 项目文件结构

```
ocr-service/
├── Dockerfile               # GPU 版构建文件（基于 Dockerfile.tensorrt）
├── Dockerfile.cpu           # CPU 版构建文件（基于 Dockerfile.onnxruntime-cpu）
├── docker-compose.yml       # 双版本服务编排（gpu / cpu profile）
├── .dockerignore            # 排除非必要文件
├── gunicorn_conf.py         # Gunicorn 高并发配置（两版本共用）
├── download_models.py       # 构建阶段模型预下载脚本（两版本共用）
├── config_tensorrt.yaml     # TensorRT 引擎配置（GPU 版备用）
├── ocr-docker.md            # 本文档
│
├── app/                     # HTTP 服务代码（两版本共用）
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口（路由、响应格式化）
│   └── ocr_engine.py        # OCR 引擎封装（懒加载、线程安全）
│
└── RapidOCR/                # RapidOCR 源码（git clone）
    └── python/              # 核心 Python 模块（被 COPY 到镜像的 /app/）
        ├── rapidocr/        # OCR 引擎核心代码
        │   ├── config.yaml          # 默认引擎配置
        │   ├── default_models.yaml  # 模型下载 URL 索引
        │   ├── main.py              # RapidOCR 主入口
        │   ├── inference_engine/    # 多引擎后端
        │   │   ├── onnxruntime/
        │   │   ├── tensorrt/
        │   │   └── ...
        │   ├── ch_ppocr_det/        # 检测模块
        │   ├── ch_ppocr_cls/        # 分类模块
        │   ├── ch_ppocr_rec/        # 识别模块
        │   ├── models/              # 模型存储目录
        │   └── utils/               # 工具函数
        └── requirements.txt         # Python 依赖清单
```

---

## 3. 配置方案

### 3.1 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_SERVICE_PORT` | `8089` | HTTP 服务端口 |
| `OCR_WORKERS` | `2` | Gunicorn Worker 进程数 |
| `OCR_THREADS` | `4` | 每个 Worker 的线程数 |
| `OCR_MAX_REQUESTS` | `1000` | 每个 Worker 处理多少请求后自动回收（防内存泄漏） |
| `LOG_LEVEL` | `info` | 日志级别 (`debug` / `info` / `warning` / `error`) |
| `NVIDIA_VISIBLE_DEVICES` | `all` | 对容器可见的 GPU 设备 |

### 3.2 Worker 数量调优

每个 Worker 独立加载一份 OCR 模型到内存，约占用 **200-400MB**。根据显存/内存选择合适数量：

| 硬件配置 | 推荐 `OCR_WORKERS` |
|----------|---------------------|
| RTX 4090 (24GB VRAM) | `4` |
| RTX 3090 (24GB VRAM) | `3` |
| RTX 3060 (12GB VRAM) | `2` |
| 8GB VRAM 及以下 | `1` |
| **纯 CPU 模式** | CPU 核心数 |

### 3.3 Gunicorn 关键配置 (`gunicorn_conf.py`)

```python
# Worker 类型：异步 ASGI Worker（处理 FastAPI）
worker_class = "uvicorn.workers.UvicornWorker"

# 超时：600s（首次 TensorRT engine 编译可能需数分钟）
timeout = 600

# Worker 自动回收：处理 1000±50 个请求后重启（防内存泄漏）
max_requests = 1000
max_requests_jitter = 50

# 不预加载应用：每个 Worker 独立初始化 OCR 引擎
preload_app = False
```

### 3.4 OCR 模型配置

模型通过 RapidOCR 的 `config.yaml` 管理，默认使用 **PP-OCRv4 mobile** 版本：

```yaml
Det:
    engine_type: "onnxruntime"    # 推理引擎
    lang_type: "ch"               # 中文
    model_type: "mobile"          # mobile 版（速度优先）
    ocr_version: "PP-OCRv4"       # OCR 版本

Rec:
    engine_type: "onnxruntime"
    lang_type: "ch"
    model_type: "mobile"
    ocr_version: "PP-OCRv4"
    rec_img_shape: [3, 48, 320]   # 输入尺寸
    rec_batch_num: 6              # 批处理大小
```

如需换用 **server 版模型**（更高精度、更慢），修改 `model_type: "server"`。

---

## 4. 构建步骤

### 4.1 前置条件

**通用要求：**

| 依赖 | 版本要求 |
|------|----------|
| Docker | 20.10+ |
| Docker Compose | v2+ |

**GPU 版额外要求：**

| 依赖 | 版本要求 |
|------|----------|
| NVIDIA Driver | 535+ (支持 CUDA 12.x) |
| NVIDIA Container Toolkit | 已安装 |
| 磁盘空间 | ≥ 40GB（DeepStream 基础镜像约 14GB） |

**CPU 版额外要求：**

| 依赖 | 版本要求 |
|------|----------|
| 磁盘空间 | ≥ 3GB |

验证 GPU 可用（仅 GPU 版需要）：
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 4.2 获取 RapidOCR 源码

```bash
cd /ocode/videoParser/ocr-service

# 克隆 RapidOCR（只需 python/ 目录）
git clone https://github.com/RapidAI/RapidOCR.git
```

### 4.3 构建镜像

#### GPU 版（`Dockerfile`）

```bash
# 直接构建
sudo docker build -t ocr-service:latest .

# 或 docker-compose
sudo docker compose --profile gpu build
```

构建过程：
1. **拉取基础镜像** — `deepstream:7.0-gc-triton-devel` (~14GB，首次较慢)
2. **安装系统依赖** — libgl1, gcc, g++ 等
3. **安装 Python 依赖** — RapidOCR core + TensorRT bindings + ONNX Runtime + FastAPI + Gunicorn
4. **复制 RapidOCR 源码** → `/app/`
5. **预下载模型** — 3 个 ONNX 模型文件
6. **复制 HTTP 服务代码**

#### CPU 版（`Dockerfile.cpu`）⚡ 推荐快速上手

```bash
# 直接构建
sudo docker build -t ocr-service-cpu:latest -f Dockerfile.cpu .

# 或 docker-compose
sudo docker compose --profile cpu build
```

构建过程：
1. **拉取基础镜像** — `python:3.10-slim-bookworm` (~50MB，极快)
2. **安装系统依赖** — libgl1, gcc, g++ 等
3. **安装 Python 依赖** — RapidOCR core + ONNX Runtime (CPU) + FastAPI + Gunicorn
4. **复制 RapidOCR 源码** → `/app/`
5. **预下载模型** — 3 个 ONNX 模型文件
6. **复制 HTTP 服务代码**

> ⚠️ 模型从 `modelscope.cn` 下载，国内网络通常很快。如遇下载失败可重试构建。

### 4.4 验证构建

```bash
sudo docker images | grep ocr-service
# 预期输出：
# ocr-service       latest   ...   ~27GB   (GPU 版)
# ocr-service-cpu   latest   ...   ~1GB    (CPU 版)
```

---

## 5. 使用方法

### 5.1 启动服务

#### GPU 版

```bash
# docker run
sudo docker run -d \
  --name ocr-service \
  --gpus all \
  -p 8089:8089 \
  -e OCR_WORKERS=2 \
  ocr-service:latest

# 或 docker-compose
sudo docker compose --profile gpu up -d
```

#### CPU 版

```bash
# docker run（无需 --gpus）
sudo docker run -d \
  --name ocr-service-cpu \
  -p 8089:8089 \
  -e OCR_WORKERS=4 \
  ocr-service-cpu:latest

# 或 docker-compose
sudo docker compose --profile cpu up -d
```

```bash
# 查看启动日志
sudo docker logs -f ocr-service       # GPU 版
sudo docker logs -f ocr-service-cpu   # CPU 版
```

预期启动日志（两版本相同）：
```
[INFO] Starting gunicorn 25.3.0
[INFO] Listening at: http://0.0.0.0:8089
[INFO] Using worker: uvicorn.workers.UvicornWorker
[INFO] Booting worker with pid: 102
🚀 OCR 引擎已初始化: onnxruntime + ch_PP-OCRv4
✅ 引擎预热完成
```

> 💡 **两个版本的 API 完全一致**，以下示例对 GPU 和 CPU 版均适用。

### 5.2 API 端点

#### `GET /health` — 健康检查

```bash
curl http://localhost:8089/health
```

响应：
```json
{
  "status": "ok",
  "device": "onnxruntime",
  "version": "2.0.0",
  "engine": "rapidocr",
  "workers": 2
}
```

#### `POST /ocr` — 完整 OCR

上传图片，返回所有检测到的文本行 + 坐标 + 置信度。

```bash
curl -X POST http://localhost:8089/ocr \
  -F "file=@test.jpg"
```

响应：
```json
{
  "results": [
    {
      "text": "Hello World",
      "confidence": 0.9976,
      "bbox": [[49,32],[226,34],[225,88],[49,85]]
    },
    {
      "text": "RapidOCR Test",
      "confidence": 0.9956,
      "bbox": [[126,122],[480,118],[481,172],[127,177]]
    }
  ],
  "total": 2,
  "device": "onnxruntime",
  "elapse": {
    "det": 0.2772,
    "cls": 0.007,
    "rec": 0.0659
  }
}
```

#### `POST /ocr/region` — 区域 OCR

先按比例裁剪图片，再执行 OCR。适用于提取字幕、标题等固定区域文字。

```bash
# 只识别图片下方 35%（字幕区域）
curl -X POST http://localhost:8089/ocr/region \
  -F "file=@frame.jpg" \
  -F "top=0.65" \
  -F "bottom=1.0" \
  -F "left=0.125" \
  -F "right=0.875"
```

参数说明：
| 参数 | 范围 | 说明 |
|------|------|------|
| `top` | 0.0~1.0 | 顶部起始比例（0 = 图片顶部） |
| `bottom` | 0.0~1.0 | 底部结束比例（1.0 = 图片底部） |
| `left` | 0.0~1.0 | 左侧起始比例 |
| `right` | 0.0~1.0 | 右侧结束比例 |

#### Swagger 在线文档

浏览器访问：
```
http://localhost:8089/docs
```

### 5.3 Python 客户端示例

```python
import requests

def ocr_image(image_path: str, server="http://localhost:8089") -> dict:
    """调用 OCR 服务识别图片"""
    with open(image_path, "rb") as f:
        resp = requests.post(f"{server}/ocr", files={"file": f})
    resp.raise_for_status()
    return resp.json()

def ocr_subtitle(image_path: str, server="http://localhost:8089") -> str:
    """提取字幕区域文字"""
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{server}/ocr/region",
            files={"file": f},
            data={"top": 0.65, "bottom": 1.0, "left": 0.125, "right": 0.875},
        )
    resp.raise_for_status()
    data = resp.json()
    return " ".join(r["text"] for r in data["results"])

# 使用
result = ocr_image("screenshot.png")
for r in result["results"]:
    print(f"[{r['confidence']:.2%}] {r['text']}")

subtitle = ocr_subtitle("video_frame.jpg")
print(f"字幕: {subtitle}")
```

### 5.4 批量并发调用

```python
import asyncio
import aiohttp

async def ocr_batch(image_paths: list, server="http://localhost:8089", concurrency=10):
    """批量并发 OCR"""
    sem = asyncio.Semaphore(concurrency)
    results = {}

    async def _ocr(session, path):
        async with sem:
            with open(path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=path)
                async with session.post(f"{server}/ocr", data=data) as resp:
                    results[path] = await resp.json()

    async with aiohttp.ClientSession() as session:
        tasks = [_ocr(session, p) for p in image_paths]
        await asyncio.gather(*tasks)

    return results

# 使用
import glob
images = glob.glob("frames/*.jpg")
results = asyncio.run(ocr_batch(images, concurrency=20))
```

---

## 6. 运维操作

### 6.1 常用命令

```bash
# 启动 GPU 版
sudo docker compose --profile gpu up -d

# 启动 CPU 版
sudo docker compose --profile cpu up -d

# 停止
sudo docker compose --profile gpu down   # 或 cpu

# 查看日志
sudo docker logs -f ocr-service          # GPU 版
sudo docker logs -f ocr-service-cpu      # CPU 版

# 查看资源占用
sudo docker stats ocr-service
```

### 6.2 调整 Worker 数量（无需重新构建）

```bash
# GPU 版 — 以 4 个 Worker 启动
sudo docker rm -f ocr-service
sudo docker run -d --name ocr-service --gpus all \
  -p 8089:8089 \
  -e OCR_WORKERS=4 \
  ocr-service:latest

# CPU 版 — 以 8 个 Worker 启动
sudo docker rm -f ocr-service-cpu
sudo docker run -d --name ocr-service-cpu \
  -p 8089:8089 \
  -e OCR_WORKERS=8 \
  ocr-service-cpu:latest
```

### 6.3 模型缓存 Volume

`docker-compose.yml` 配置了 `trt-engines` volume 挂载到 `/app/rapidocr/models/`，
用于持久化模型文件和 TensorRT engine 缓存。

```bash
# 查看 volume
docker volume ls | grep ocr

# 清除模型缓存（强制重新下载）
docker volume rm ocr-trt-engines
```

### 6.4 健康检查

容器内置 HEALTHCHECK，每 30 秒检查一次：

```bash
# 查看健康状态
sudo docker inspect ocr-service --format='{{.State.Health.Status}}'
# 预期输出: healthy
```

---

## 7. 关键设计说明

### 7.1 为什么用 Gunicorn 而不是 Uvicorn 单进程？

OCR 推理是 **CPU/GPU 密集型** 任务，Python GIL 限制了单进程的并发能力。
Gunicorn 作为进程管理器，启动多个 Worker 进程，每个 Worker 有独立的 GIL 和 OCR 引擎实例，
实现真正的并行推理。

### 7.2 为什么每个 Worker 独立初始化引擎？

设置 `preload_app = False` 让每个 Worker 在 fork 后独立创建 OCR 引擎实例。
这避免了多进程共享 GPU context 导致的 CUDA 错误，同时确保每个 Worker 有自己的
推理上下文。

### 7.3 GPU 版为什么用 DeepStream？为什么还有 CPU 版？

**GPU 版** 基于 `Dockerfile.tensorrt` 构建，DeepStream 基础镜像内含：
- CUDA 12.x toolkit
- TensorRT 8.6.1（系统级安装）
- cuDNN
- Triton Inference Server

虽然当前使用 ONNX Runtime 作为推理引擎，但已为后续切换到
TensorRT 引擎做好准备（`config_tensorrt.yaml` 已就绪）。

**CPU 版** 基于 `Dockerfile.onnxruntime-cpu` + `Dockerfile.base` 构建，
使用 `python:3.10-slim-bookworm` 作为基础镜像，镜像体积仅 ~1GB（GPU 版的 1/27）。
适用于无 GPU 的服务器、开发测试环境、或不需要极致推理速度的场景。
两个版本共享 **完全相同的 `app/` 代码和 API 接口**，可无缝切换。

### 7.4 模型下载机制

RapidOCR 使用 `default_models.yaml` 管理模型 URL，模型托管在
`modelscope.cn`（国内 CDN）。`download_models.py` 在 Docker 构建阶段
调用 RapidOCR 内置的下载函数，将模型固化到镜像中。

下载的模型文件：
```
/app/rapidocr/models/
├── ch_PP-OCRv4_det_infer.onnx          (4.5MB)  检测模型
├── ch_ppocr_mobile_v2.0_cls_infer.onnx (0.6MB)  分类模型
└── ch_PP-OCRv4_rec_infer.onnx          (10.4MB) 识别模型
```

---

## 8. 故障排查

### 8.1 构建失败：模型下载超时

```
ERROR: Failed to download model from modelscope.cn
```

**解决**：重试构建，或在 `download_models.py` 中配置代理。

### 8.2 启动失败：CUDA 错误

```
CUDA initialization failure with error: 35
```

**解决**：
1. 确认主机 NVIDIA 驱动版本 ≥ 535
2. 确认安装了 NVIDIA Container Toolkit
3. 运行 `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### 8.3 OCR 返回空结果

```json
{"results": [], "total": 0}
```

**原因**：图片中没有可检测的文本，或者图片太小。RapidOCR 默认 `min_height: 30`，
小于 30px 高度的文本区域会被跳过。

### 8.4 Worker 内存持续增长

Gunicorn 配置了 `max_requests=1000`，每个 Worker 处理 1000 个请求后自动重启。
如果仍有问题，可以降低 `OCR_MAX_REQUESTS` 值：

```bash
docker run -d --name ocr-service --gpus all \
  -p 8089:8089 \
  -e OCR_MAX_REQUESTS=500 \
  ocr-service:latest
```

---

## 9. 从零复现完整步骤

### 9.1 GPU 版（需要 NVIDIA GPU）

```bash
# 1. 确保宿主机有 NVIDIA GPU + 驱动 + Container Toolkit
nvidia-smi

# 2. 进入工作目录
cd /ocode/videoParser/ocr-service

# 3. 克隆 RapidOCR 源码（如已存在则跳过）
git clone https://github.com/RapidAI/RapidOCR.git

# 4. 构建镜像（首次约 15-20 分钟，主要是拉基础镜像）
sudo docker build -t ocr-service:latest .

# 5. 启动服务
sudo docker run -d \
  --name ocr-service \
  --gpus all \
  -p 8089:8089 \
  -e OCR_WORKERS=2 \
  ocr-service:latest

# 6. 等待引擎初始化（约 10-15 秒）
sleep 15

# 7. 验证
curl http://localhost:8089/health
curl -X POST http://localhost:8089/ocr -F "file=@test.jpg"
```

### 9.2 CPU 版（任意 Linux 服务器）⚡ 推荐快速上手

```bash
# 1. 进入工作目录
cd /ocode/videoParser/ocr-service

# 2. 克隆 RapidOCR 源码（如已存在则跳过）
git clone https://github.com/RapidAI/RapidOCR.git

# 3. 构建镜像（约 2-3 分钟）
sudo docker build -t ocr-service-cpu:latest -f Dockerfile.cpu .

# 4. 启动服务（无需 --gpus）
sudo docker run -d \
  --name ocr-service-cpu \
  -p 8089:8089 \
  -e OCR_WORKERS=4 \
  ocr-service-cpu:latest

# 5. 等待引擎初始化（约 5-8 秒）
sleep 8

# 6. 验证
curl http://localhost:8089/health
curl -X POST http://localhost:8089/ocr -F "file=@test.jpg"

# 7. Swagger 文档
echo "打开浏览器访问: http://<服务器IP>:8089/docs"
```

---

## 10. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-04-05 | PaddleOCR 3.x + Uvicorn 单进程，基于 `paddlepaddle/paddle:3.3.1-gpu` |
| v2.0 | 2026-04-05 | 迁移至 RapidOCR + DeepStream 7.0 + Gunicorn 多 Worker 高并发 |
| **v2.1** | **2026-04-05** | **新增 CPU 版镜像 (`Dockerfile.cpu`)，基于 `python:3.10-slim`，~1GB 体积** |
