#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单个例子图像编辑 Demo
不需要 OpenAI API 和 GroundingDINO

输入：
- source_image: 源图片路径
- source_mask: mask图片路径（白色区域为编辑区域）
- prompt: 编辑提示词
- editing_type: 编辑类型（可选，有默认值）
- 其他可选参数

输出：
- 编辑后的图片保存在 output/ 目录
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
from datetime import datetime

# 从 project.py 导入公共配置和函数
import project
from project import (
    load_models,
    EDITING_TYPES,
    DEFAULT_EDITING_TYPE,
    SmartFreeEdit_Pipeline
)


def edit_image(
    pipe,
    source_image_path,
    source_mask_path,
    prompt,
    editing_type=DEFAULT_EDITING_TYPE,
    output_dir="output",
    sample_id=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    control_strength=1.0,
    negative_prompt="ugly, low quality, distorted, blurry",
    blending=True,
):
    """
    编辑单个图像

    参数:
        pipe: StableDiffusionBrushNetPipeline
        source_image_path: 源图片路径
        source_mask_path: mask图片路径（白色区域为编辑区域）
        prompt: 编辑提示词
        editing_type: 编辑类型（可选，默认值：Local）
            - "Addition": 添加对象
            - "Remove": 删除对象
            - "Local": 局部编辑/替换（默认）
            - "Global": 全局编辑
            - "Background": 背景替换
            - "Resize": 调整大小
        output_dir: 输出目录（默认：output）
        sample_id: 样本ID（用于命名输出文件，如果为None则使用时间戳）
        num_inference_steps: 推理步数（默认：50）
        guidance_scale: 引导强度（默认：7.5）
        control_strength: 控制强度（默认：1.0）
        negative_prompt: 负面提示词（默认："ugly, low quality, distorted, blurry"）
        blending: 是否混合（默认：True）

    返回:
        编辑后的图片路径
    """
    print("\n" + "=" * 60)
    print("开始编辑图像...")
    print("=" * 60)
    print(f"源图片: {source_image_path}")
    print(f"Mask图片: {source_mask_path}")
    print(f"提示词: {prompt}")
    print(f"编辑类型: {editing_type}")

    # 检查文件是否存在
    if not os.path.exists(source_image_path):
        raise FileNotFoundError(f"源图片不存在: {source_image_path}")
    if not os.path.exists(source_mask_path):
        raise FileNotFoundError(f"Mask图片不存在: {source_mask_path}")

    # 验证编辑类型
    if editing_type not in project.EDITING_TYPES:
        print(f"⚠️  警告：编辑类型 '{editing_type}' 不在支持列表中，使用默认值 '{project.DEFAULT_EDITING_TYPE}'")
        editing_type = project.DEFAULT_EDITING_TYPE

    # 加载图片
    original_image = np.array(Image.open(source_image_path).convert("RGB"))
    original_mask = np.array(Image.open(source_mask_path).convert("RGB"))

    # 如果 mask 是彩色图，转换为灰度图（取第一个通道）
    if original_mask.ndim == 3:
        original_mask = original_mask[:, :, 0]

    print(f"源图片尺寸: {original_image.shape}")
    print(f"Mask尺寸: {original_mask.shape}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名
    if sample_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_id = f"sample_{timestamp}"

    output_path = os.path.join(output_dir, f"{sample_id}_result.png")
    mask_save_path = os.path.join(output_dir, f"{sample_id}_mask.png")

    # 执行编辑
    print("\n正在生成...")
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)

    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        images, mask_image, mask_np, init_image_np = SmartFreeEdit_Pipeline(
            pipe,
            prompt,
            original_mask,
            original_image,
            generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            control_strength=control_strength,
            negative_prompt=negative_prompt,
            num_samples=1,
            blending=blending
        )

    # 保存结果
    images[0].save(output_path)
    Image.fromarray(original_mask).save(mask_save_path)

    print(f"\n✅ 编辑完成！")
    print(f"   结果图片: {output_path}")
    print(f"   Mask图片: {mask_save_path}")
    print("=" * 60)

    return output_path, mask_save_path


if __name__ == "__main__":
    print("=" * 60)
    print("单个例子图像编辑 Demo")
    print("=" * 60)
    print("\n支持的编辑类型：")
    for edit_type, description in project.EDITING_TYPES.items():
        marker = " (默认)" if edit_type == project.DEFAULT_EDITING_TYPE else ""
        print(f"  - {edit_type}{marker}: {description}")

    # ============================================
    # 配置参数（修改这里以运行）
    # ============================================
    source_image_path = "/home/liying/Desktop/IMAGE_EDITE-CVPR-2025/images/ReasonEdit/6-Reasoning/009.png"
    source_mask_path = "/home/liying/Desktop/IMAGE_EDITE-CVPR-2025/images/ReasonEdit/6-Reasoning/009_mask.jpg"  # 如果不存在，需要创建
    prompt = "a carrot"  # 编辑提示词
    editing_type = "Local"  # 编辑类型（可选）
    sample_id = "009_carrot"  # 样本ID（可选，用于命名输出文件）

    # 高级参数（可选，使用默认值即可）
    num_inference_steps = 50
    guidance_scale = 7.5
    control_strength = 1.0
    negative_prompt = "ugly, low quality, distorted, blurry"
    blending = True

    # ============================================
    # 检查配置
    # ============================================
    if not os.path.exists(source_image_path):
        print(f"\n❌ 错误：源图片不存在: {source_image_path}")
        print("   请修改脚本中的 source_image_path 变量")
        sys.exit(1)

    if not os.path.exists(source_mask_path):
        print(f"\n⚠️  警告：Mask图片不存在: {source_mask_path}")
        print("   请创建 mask 图片（白色区域表示要编辑的区域）")
        sys.exit(1)

    # ============================================
    # 加载模型并执行编辑
    # ============================================
    try:
        # 加载模型
        pipe = load_models()

        # 执行编辑
        result_path, mask_path = edit_image(
            pipe=pipe,
            source_image_path=source_image_path,
            source_mask_path=source_mask_path,
            prompt=prompt,
            editing_type=editing_type,
            sample_id=sample_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            control_strength=control_strength,
            negative_prompt=negative_prompt,
            blending=blending,
        )

        print("\n🎉 编辑成功完成！")
        print(f"   结果图片: {result_path}")
        print(f"   Mask图片: {mask_path}")

    except Exception as e:
        print(f"\n❌ 编辑失败: {e}")
        import traceback
        traceback.print_exc()
