#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import sys
import argparse
import logging
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.image as image
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import matplotlib.text as text
import matplotlib.font_manager as font_manager

# ============================================
# 项目路径设置
# ============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

# 添加项目路径到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加 src 目录到路径（包含修改后的 diffusers）
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 设置模型路径环境变量（如果未设置）
if "SMARTFREEEDIT_MODEL_PATH" not in os.environ:
    os.environ["SMARTFREEEDIT_MODEL_PATH"] = "/home/liying/Documents/smart_free_edit_huggingface"

# 设置 transformers 环境变量
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

# ============================================
# 编辑类型定义
# ============================================
EDITING_TYPES = {
    "Addition": "添加新对象到图像中，例如：add a bird, add a car in the background",
    "Remove": "删除图像中的对象，例如：remove the car, remove the person",
    "Local": "替换局部对象或改变对象属性，例如：change the cat to a dog, make it smile, replace the red apple with a green apple",
    "Global": "编辑整个图像，例如：let's see it in winter, Change the season from autumn to spring",
    "Background": "改变场景背景，例如：change the background to a beach, make the hedgehog in France",
    "Resize": "调整对象大小，例如：minify the giraffe in the image, make the car bigger"
}

DEFAULT_EDITING_TYPE = "Remove"  # 默认编辑类型

# ============================================
# 数据集路径（原有功能保留）
# ============================================
# dataset path
CHECKPOINTS_DIR = os.path.join(current_dir, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

DATASETS_DIR = os.path.join(current_dir, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# tmp_name = "text_driven_prompts_edit_bench"
# tmp_name = "text_driven_prompts_webpic"
tmp_name = "images_tmp"
IMAGES_DIR = os.path.join(DATASETS_DIR, tmp_name)
os.makedirs(IMAGES_DIR, exist_ok=True)

IMAGES_JSON_FILE = os.path.join(DATASETS_DIR, f"{tmp_name}.json")
# 注意：不在这里检查文件是否存在，因为某些脚本可能不需要这个文件

# result path
OUTPUT_DIR = os.path.join(current_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDITING_RESULTS_DIR = os.path.join(OUTPUT_DIR, tmp_name)
os.makedirs(EDITING_RESULTS_DIR, exist_ok=True)

# ============================================
# 模型加载函数
# ============================================
def load_models():
    """
    加载 BrushNet 和基础模型

    返回:
        pipe: StableDiffusionBrushNetPipeline
    """
    # 导入 torch 并验证
    import torch
    import torch.nn as nn

    # 验证 torch 是否正常工作
    try:
        test_tensor = torch.tensor([1.0])
        test_module = nn.Linear(1, 1)
        del test_tensor, test_module
    except Exception as e:
        raise RuntimeError(
            f"PyTorch 未正确安装或损坏: {e}\n"
            "请运行：\n"
            "pip uninstall torch torchvision torchaudio -y\n"
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
        )

    # 导入 diffusers 组件
    from diffusers.pipelines.brushnet.pipeline_brushnet import StableDiffusionBrushNetPipeline
    from diffusers.models import BrushNetModel
    from diffusers.schedulers import UniPCMultistepScheduler

    print("=" * 60)
    print("正在加载模型...")
    print("=" * 60)

    # 获取模型路径
    try:
        from SmartFreeEdit.config_local import (
            SMARTFREEEDIT_MODEL_PATH,
            DEFAULT_BASE_MODEL_PATH,
            BRUSHNET_PATH,
        )
        model_path = SMARTFREEEDIT_MODEL_PATH
        base_model_path = DEFAULT_BASE_MODEL_PATH
        brushnet_path = BRUSHNET_PATH
    except ImportError:
        model_path = os.getenv("SMARTFREEEDIT_MODEL_PATH", "/home/liying/Documents/smart_free_edit_huggingface")
        base_model_path = os.path.join(model_path, "base_model/realisticVisionV60B1_v51VAE")
        brushnet_path = os.path.join(model_path, "checkpoint-100000/brushnet")

    torch_dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"模型路径: {model_path}")
    print(f"设备: {device}")

    # 加载 BrushNet 和基础模型
    print("\n[1/1] 加载 BrushNet 和基础模型...")
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path, brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    print("✅ 模型加载完成")
    print("=" * 60)

    return pipe


# 导出 SmartFreeEdit_Pipeline 供其他模块使用
def get_smartfreeedit_pipeline():
    """获取 SmartFreeEdit_Pipeline 函数（延迟导入）"""
    from SmartFreeEdit.src.smartfreeedit_all_pipeline import SmartFreeEdit_Pipeline
    return SmartFreeEdit_Pipeline

# 为了向后兼容，提供一个可以直接导入的别名
# 注意：实际使用时会在函数内部导入
SmartFreeEdit_Pipeline = None  # 将在需要时动态导入
