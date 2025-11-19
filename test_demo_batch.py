#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SmartFreeEdit 本地推理示例
使用本地模型进行图像编辑
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# 添加 src 目录到路径（包含修改后的 diffusers）
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 设置模型路径（方式1：环境变量）
os.environ["SMARTFREEEDIT_MODEL_PATH"] = "/home/liying/Documents/smart_free_edit_huggingface"

# 或者方式2：修改 SmartFreeEdit/config_local.py 文件

from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
from SmartFreeEdit.src.smartfreeedit_all_pipeline import SmartFreeEdit_Pipeline
from SmartFreeEdit.utils.utils import load_grounding_dino_model
from SmartFreeEdit.utils.utils_lisa import load_lisa_model

def load_models():
    """加载所有需要的模型"""
    print("=" * 50)
    print("正在加载模型...")
    print("=" * 50)

    # 获取模型路径
    try:
        from SmartFreeEdit.config_local import (
            SMARTFREEEDIT_MODEL_PATH,
            DEFAULT_BASE_MODEL_PATH,
            BRUSHNET_PATH,
            LISA_PATH,
            GROUNDINGDINO_PATH
        )
        model_path = SMARTFREEEDIT_MODEL_PATH
        base_model_path = DEFAULT_BASE_MODEL_PATH
        brushnet_path = BRUSHNET_PATH
        lisa_path = LISA_PATH
        groundingdino_path = GROUNDINGDINO_PATH
    except ImportError:
        model_path = os.getenv("SMARTFREEEDIT_MODEL_PATH", "/home/liying/Documents/smart_free_edit_huggingface")
        base_model_path = os.path.join(model_path, "base_model/realisticVisionV60B1_v51VAE")
        brushnet_path = os.path.join(model_path, "checkpoint-100000/brushnet")
        lisa_path = os.path.join(model_path, "LISA-7B-v1-explanatory")
        groundingdino_path = os.path.join(model_path, "grounding_dino/groundingdino_swint_ogc.pth")

    torch_dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"模型路径: {model_path}")
    print(f"设备: {device}")

    # 1. 加载BrushNet和基础模型
    print("\n[1/4] 加载BrushNet和基础模型...")
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path, brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()  # 节省显存
    print("✅ BrushNet和基础模型加载完成")

    # 2. 加载GroundingDINO
    print("\n[2/4] 加载GroundingDINO...")
    config_file = os.path.join(project_root, "SmartFreeEdit/utils/GroundingDINO_SwinT_OGC.py")
    groundingdino_model = load_grounding_dino_model(config_file, groundingdino_path, device=device)
    print("✅ GroundingDINO加载完成")

    # 3. 加载LISA
    print("\n[3/4] 加载LISA模型...")
    lisa_model, tokenizer = load_lisa_model(
        version=lisa_path,
        precision="fp16",
        load_in_8bit=True,
        load_in_4bit=False,
        vision_tower="openai/clip-vit-large-patch14",
        local_rank=0
    )
    print("✅ LISA模型加载完成")

    print("\n" + "=" * 50)
    print("所有模型加载完成！")
    print("=" * 50)

    return pipe, groundingdino_model, lisa_model, tokenizer, device


def simple_inference_with_mask(pipe, input_image_path, mask_image_path, prompt, output_path="output.png"):
    """
    简单推理：需要手动提供mask

    参数:
        pipe: StableDiffusionBrushNetPipeline
        input_image_path: 输入图片路径
        mask_image_path: mask图片路径（白色区域为编辑区域）
        prompt: 编辑提示词
        output_path: 输出路径
    """
    print(f"\n开始推理...")
    print(f"输入图片: {input_image_path}")
    print(f"Mask图片: {mask_image_path}")
    print(f"提示词: {prompt}")

    # 加载图片
    init_image = Image.open(input_image_path).convert("RGB")
    mask_image = Image.open(mask_image_path).convert("RGB")

    # 推理
    generator = torch.Generator("cuda").manual_seed(42)
    images = pipe(
        prompt,
        init_image,
        mask_image,
        num_inference_steps=50,
        guidance_scale=7.5,
        brushnet_conditioning_scale=1.0,
        generator=generator,
        negative_prompt="ugly, low quality"
    ).images

    # 保存结果
    images[0].save(output_path)
    print(f"✅ 结果已保存到: {output_path}")

    return images[0]


def full_pipeline_inference(pipe, groundingdino_model, lisa_model, tokenizer, device,
                            input_image_path, prompt, output_path="output.png",
                            api_key=None, api_version="2024-08-01-preview",
                            end_point=None, engine="4o"):
    """
    完整流水线推理：自动生成mask（需要GPT-4o API）

    参数:
        pipe: StableDiffusionBrushNetPipeline
        groundingdino_model: GroundingDINO模型
        lisa_model: LISA模型
        tokenizer: LISA tokenizer
        device: 设备
        input_image_path: 输入图片路径
        prompt: 编辑指令（自然语言）
        output_path: 输出路径
        api_key: GPT-4o API密钥
        api_version: API版本
        end_point: API端点
        engine: API引擎名称
    """
    from SmartFreeEdit.src.vlm_pipeline import (
        vlm_response_editing_type,
        vlm_response_object_wait_for_edit,
        vlm_response_mask,
        vlm_response_prompt_after_apply_instruction
    )

    if not api_key or not end_point:
        raise ValueError("完整流水线需要GPT-4o API配置。请提供api_key和end_point参数。")

    print(f"\n开始完整流水线推理...")
    print(f"输入图片: {input_image_path}")
    print(f"编辑指令: {prompt}")

    # 加载图片
    original_image = np.array(Image.open(input_image_path).convert("RGB"))

    # 构建API URL
    url = f"{end_point}/openai/deployments/{engine}/chat/completions?api-version={api_version}"

    # 1. 确定编辑类别
    print("\n[步骤1/5] 确定编辑类别...")
    category = vlm_response_editing_type(url, api_key, original_image, prompt, device)
    print(f"编辑类别: {category}")

    # 2. 确定编辑对象
    print("\n[步骤2/5] 确定编辑对象...")
    object_wait_for_edit = vlm_response_object_wait_for_edit(
        url, api_key, original_image, category, prompt, device
    )
    print(f"编辑对象: {object_wait_for_edit}")

    # 3. 生成mask
    print("\n[步骤3/5] 生成mask...")
    original_mask = vlm_response_mask(
        url, api_key, category, original_image, prompt,
        object_wait_for_edit, lisa_model, tokenizer, device
    ).astype(np.uint8)
    print("✅ Mask生成完成")

    # 4. 生成目标提示词
    print("\n[步骤4/5] 生成目标提示词...")
    target_prompt = vlm_response_prompt_after_apply_instruction(
        url, api_key, original_image, prompt, category, device
    )
    print(f"目标提示词: {target_prompt}")

    # 5. 执行编辑
    print("\n[步骤5/5] 执行图像编辑...")
    generator = torch.Generator(device).manual_seed(42)
    with torch.autocast(device):
        images, mask_image, mask_np, init_image_np = SmartFreeEdit_Pipeline(
            pipe,
            target_prompt,
            original_mask,
            original_image,
            generator,
            num_inference_steps=50,
            guidance_scale=7.5,
            control_strength=1.0,
            negative_prompt="ugly, low quality",
            num_samples=1,
            blending=True
        )

    # 保存结果
    images[0].save(output_path)
    print(f"✅ 结果已保存到: {output_path}")

    return images[0]


if __name__ == "__main__":
    print("=" * 60)
    print("SmartFreeEdit 本地推理示例")
    print("=" * 60)

    # 加载模型
    pipe, groundingdino_model, lisa_model, tokenizer, device = load_models()

    # ============================================
    # 示例1：简单推理（需要手动提供mask）
    # ============================================
    print("\n" + "=" * 60)
    print("示例1：简单推理（需要mask）")
    print("=" * 60)

    input_image_path = "/home/liying/Desktop/IMAGE_EDITE-CVPR-2025/images/ReasonEdit/6-Reasoning/009.png"
    mask_image_path = "/home/liying/Desktop/IMAGE_EDITE-CVPR-2025/images/ReasonEdit/6-Reasoning/009_mask.jpg"

    # 取消注释以运行
    simple_inference_with_mask(
        pipe=pipe,
        input_image_path=input_image_path,  # 替换为你的图片路径
        mask_image_path=mask_image_path,    # 替换为你的mask路径（白色=编辑区域）
        prompt="a beautiful landscape",
        output_path="output_simple.png"
    )

    # ============================================
    # 示例2：完整流水线（自动生成mask，需要GPT-4o API）
    # ============================================
    print("\n" + "=" * 60)
    print("示例2：完整流水线（自动生成mask）")
    print("=" * 60)

    # 取消注释并配置API以运行
    # full_pipeline_inference(
    #     pipe=pipe,
    #     groundingdino_model=groundingdino_model,
    #     lisa_model=lisa_model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     input_image_path="input.jpg",  # 替换为你的图片路径
    #     prompt="remove the car",       # 编辑指令
    #     output_path="output_full.png",
    #     api_key="your_api_key",        # 替换为你的API密钥
    #     api_version="2024-08-01-preview",
    #     end_point="https://your-endpoint.openai.azure.com/",  # 替换为你的端点
    #     engine="4o"
    # )

    print("\n" + "=" * 60)
    print("提示：取消注释上面的代码并配置参数以运行推理")
    print("=" * 60)
