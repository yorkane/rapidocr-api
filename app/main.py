"""RapidOCR HTTP 服务 — FastAPI + Gunicorn 高并发

基于 RapidOCR 源码 (DeepStream 基础镜像) + ch_PP-OCRv4 模型
Gunicorn 多 Worker 进程管理

API 端点:
  POST /ocr           完整 OCR（检测+分类+识别）
  POST /ocr/region    指定区域 OCR
  GET  /health        健康检查
"""

import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.ocr_engine import OCREngine

logger = logging.getLogger("ocr-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


app = FastAPI(
    title="RapidOCR HTTP Service",
    description=(
        "基于 RapidOCR + ONNX Runtime 的高并发 OCR 服务\n"
        "模型: ch_PP-OCRv4 (det + cls + rec)\n"
        "基础镜像: DeepStream 7.0\n"
        "并发: Gunicorn + Uvicorn Worker"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 工具函数 ──────────────────────────────────────────────────────────
async def _read_image(file: UploadFile) -> np.ndarray:
    """读取上传文件为 OpenCV BGR 图像"""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="上传文件为空")
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片，请检查文件格式")
    return img


def _format_result(result) -> dict:
    """将 RapidOCR 输出转为 JSON 可序列化的字典

    RapidOCROutput 包含:
      - boxes: np.ndarray or None
      - txts: tuple of str or None
      - scores: tuple of float or None
      - elapse_list: [det_elapse, cls_elapse, rec_elapse]
    """
    lines = []

    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        txts = result.txts if result.txts else [None] * len(boxes)
        scores = result.scores if result.scores else [0.0] * len(boxes)

        for i in range(len(boxes)):
            bbox = boxes[i]
            if hasattr(bbox, 'tolist'):
                bbox = bbox.tolist()
            elif isinstance(bbox, (list, tuple)):
                bbox = [[float(p) for p in pt] for pt in bbox]

            text = str(txts[i]).strip() if i < len(txts) and txts[i] else ""
            conf = float(scores[i]) if i < len(scores) else 0.0

            lines.append({
                "text": text,
                "confidence": round(conf, 4),
                "bbox": bbox,
            })

    # 解析耗时
    elapse = {}
    if hasattr(result, 'elapse_list') and result.elapse_list:
        keys = ["det", "cls", "rec"]
        for i, v in enumerate(result.elapse_list):
            if i < len(keys) and v is not None:
                elapse[keys[i]] = round(float(v), 4)

    return {
        "results": lines,
        "total": len(lines),
        "device": OCREngine.get_device(),
        "elapse": elapse,
    }


def _crop_region(img: np.ndarray, top: float, bottom: float,
                 left: float, right: float) -> np.ndarray:
    """按比例裁剪图像区域"""
    h, w = img.shape[:2]
    y1, y2 = int(h * top), int(h * bottom)
    x1, x2 = int(w * left), int(w * right)
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        raise HTTPException(status_code=400, detail="裁剪区域为空，请检查参数")
    return cropped


# ── API 端点 ──────────────────────────────────────────────────────────

@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "device": OCREngine.get_device(),
        "version": "2.0.0",
        "engine": "rapidocr",
        "workers": int(os.environ.get("OCR_WORKERS", "2")),
    }


@app.post("/ocr", tags=["OCR"])
async def ocr_full(
    file: UploadFile = File(..., description="待识别的图片文件"),
):
    """
    完整 OCR — 文本检测 + 方向分类 + 文本识别

    使用 ch_PP-OCRv4 模型，支持中英文。
    TensorRT FP16 加速推理。
    """
    img = await _read_image(file)
    result = OCREngine.run_ocr(img)
    return _format_result(result)


@app.post("/ocr/region", tags=["OCR"])
async def ocr_with_region(
    file: UploadFile = File(..., description="待识别的图片文件"),
    top: float = Form(0.0, ge=0.0, le=1.0, description="顶部裁剪比例 (0~1)"),
    bottom: float = Form(1.0, ge=0.0, le=1.0, description="底部裁剪比例 (0~1)"),
    left: float = Form(0.0, ge=0.0, le=1.0, description="左侧裁剪比例 (0~1)"),
    right: float = Form(1.0, ge=0.0, le=1.0, description="右侧裁剪比例 (0~1)"),
):
    """
    指定区域 OCR — 先裁剪图片指定区域，再执行 OCR。

    例如提取字幕区域：top=0.65, bottom=1.0, left=0.125, right=0.875
    """
    img = await _read_image(file)
    cropped = _crop_region(img, top, bottom, left, right)
    result = OCREngine.run_ocr(cropped)
    return _format_result(result)


# ── 全局异常处理 ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"},
    )
