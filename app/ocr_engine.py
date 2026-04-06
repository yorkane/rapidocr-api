"""OCR Engine — RapidOCR ONNX Runtime 封装

使用 RapidOCR 源码 + ONNX Runtime 引擎。
每个 Gunicorn Worker 独立持有实例。
ch_PP-OCRv4 模型已内置到镜像，开箱即用。
"""

import os
import threading
from pathlib import Path

import numpy as np


class OCREngine:
    """每个 Worker 持有一个独立的 RapidOCR 实例（线程安全）"""

    _local = threading.local()

    @classmethod
    def get_instance(cls):
        """获取当前线程的 OCR 实例（懒加载）"""
        if not hasattr(cls._local, "engine"):
            cls._local.engine = cls._create_engine()
        return cls._local.engine

    @staticmethod
    def _create_engine():
        """创建 RapidOCR 实例（使用默认 onnxruntime 引擎）"""
        from rapidocr import RapidOCR

        # 使用默认配置（onnxruntime + ch_PP-OCRv4）
        engine = RapidOCR()
        print("🚀 OCR 引擎已初始化: onnxruntime + ch_PP-OCRv4")

        # 预热
        test_img = np.zeros((100, 300, 3), dtype=np.uint8)
        engine(test_img)
        print("✅ 引擎预热完成")
        return engine

    @classmethod
    def get_device(cls) -> str:
        """返回当前使用的引擎类型"""
        return "onnxruntime"

    @classmethod
    def run_ocr(cls, img: np.ndarray):
        """
        执行完整 OCR

        Returns:
            RapidOCROutput 对象，包含 boxes, txts, scores, elapse_list
        """
        engine = cls.get_instance()
        result = engine(img)
        return result
