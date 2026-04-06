"""模型预下载脚本 — 构建阶段执行

使用 RapidOCR 内置的 download_models 功能，
在 Docker 构建阶段下载 ch_PP-OCRv4 的 ONNX 模型文件。
TensorRT 引擎会在首次推理时从 ONNX 构建优化后的 .engine 文件。
"""

import sys
from pathlib import Path


def download_models():
    """下载 ch_PP-OCRv4 的 ONNX 模型（TensorRT 需要 ONNX 作为输入）"""
    print("📥 正在下载 ch_PP-OCRv4 ONNX 模型...")

    # 使用 RapidOCR 自带的下载机制
    # 先用默认 onnxruntime config 下载模型（TensorRT 共享同一套 ONNX 文件）
    sys.path.insert(0, "/app")

    from rapidocr.utils.download_models import download_models as _download
    from rapidocr.utils.parse_parameters import ParseParams

    # 使用默认 config 下载模型（onnxruntime 引擎的 ONNX 模型）
    config_path = Path("/app/rapidocr/config.yaml")
    _download(config_path)

    # 验证模型目录
    models_dir = Path("/app/rapidocr/models")
    if models_dir.exists():
        print(f"📦 模型目录: {models_dir}")
        for f in sorted(models_dir.rglob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"   - {f.relative_to(models_dir)} ({size_mb:.1f}MB)")
    else:
        print("⚠️ 模型目录不存在，检查下载是否成功")
        sys.exit(1)

    print("✅ 模型下载完成！")


if __name__ == "__main__":
    download_models()
