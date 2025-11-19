#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对 009.png 的编辑示例
演示如何将青辣椒替换为胡萝卜
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

# 设置模型路径
os.environ["SMARTFREEEDIT_MODEL_PATH"] = "/home/liying/Documents/smart_free_edit_huggingface"

from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
from SmartFreeEdit.src.smartfreeedit_all_pipeline import SmartFreeEdit_Pipeline
from SmartFreeEdit.utils.utils import load_grounding_dino_model
from SmartFreeEdit.utils.utils_lisa import load_lisa_model
from SmartFreeEdit.src.vlm_pipeline import (
    vlm_response_editing_type,
    vlm_response_object_wait_for_edit,
    vlm_response_mask,
    vlm_response_prompt_after_apply_instruction
)

def load_models():
    """加载所有需要的模型"""
    print("=" * 60)
    print("正在加载模型...")
    print("=" * 60)

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
    pipe.enable_model_cpu_offload()
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

    print("\n" + "=" * 60)
    print("所有模型加载完成！")
    print("=" * 60)

    return pipe, groundingdino_model, lisa_model, tokenizer, device


def edit_009_image(pipe, groundingdino_model, lisa_model, tokenizer, device,
                   input_image_path, prompt, output_path,
                   api_key, api_version, end_point, engine):
    """
    编辑 009.png 图片：将青辣椒替换为胡萝卜

    参数:
        prompt: 编辑指令，例如 "replace the green pepper with a carrot"
    """
    print("\n" + "=" * 60)
    print("开始编辑图片...")
    print("=" * 60)
    print(f"输入图片: {input_image_path}")
    print(f"编辑指令: {prompt}")
    print(f"输出路径: {output_path}")

    # 加载图片
    original_image = np.array(Image.open(input_image_path).convert("RGB"))
    print(f"图片尺寸: {original_image.shape}")

    # 构建API URL
    url = f"{end_point}/openai/deployments/{engine}/chat/completions?api-version={api_version}"

    # 1. 确定编辑类别
    print("\n[步骤1/5] 确定编辑类别...")
    category = vlm_response_editing_type(url, api_key, original_image, prompt, device)
    print(f"✅ 编辑类别: {category}")

    # 2. 确定编辑对象
    print("\n[步骤2/5] 确定编辑对象...")
    object_wait_for_edit = vlm_response_object_wait_for_edit(
        url, api_key, original_image, category, prompt, device
    )
    print(f"✅ 编辑对象: {object_wait_for_edit}")

    # 3. 生成mask
    print("\n[步骤3/5] 生成mask...")
    original_mask = vlm_response_mask(
        url, api_key, category, original_image, prompt,
        object_wait_for_edit, lisa_model, tokenizer, device
    ).astype(np.uint8)
    print("✅ Mask生成完成")

    # 保存mask（可选）
    mask_save_path = output_path.replace(".png", "_mask.png")
    Image.fromarray(original_mask.squeeze()).save(mask_save_path)
    print(f"✅ Mask已保存到: {mask_save_path}")

    # 4. 生成目标提示词
    print("\n[步骤4/5] 生成目标提示词...")
    target_prompt = vlm_response_prompt_after_apply_instruction(
        url, api_key, original_image, prompt, category, device
    )
    print(f"✅ 目标提示词: {target_prompt}")

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
            negative_prompt="ugly, low quality, distorted",
            num_samples=1,
            blending=True
        )

    # 保存结果
    images[0].save(output_path)
    print(f"\n✅ 编辑完成！结果已保存到: {output_path}")
    print("=" * 60)

    return images[0], original_mask, target_prompt


if __name__ == "__main__":
    print("=" * 60)
    print("009.png 编辑示例：将青辣椒替换为胡萝卜")
    print("=" * 60)

    # ============================================
    # 配置参数
    # ============================================
    # 图片路径
    input_image_path = "/home/liying/Desktop/IMAGE_EDITE-CVPR-2025/images/ReasonEdit/6-Reasoning/009.png"

    # 编辑指令（你的需求：把青辣椒换成胡萝卜）
    # 方式1：直接替换（推荐）
    prompt = "replace the green pepper with a carrot"

    # 方式2：推理式替换（类似官方风格）
    # prompt = "What is the green spicy vegetable? Please replace it with a carrot."

    # 方式3：详细描述
    # prompt = "replace the green pepper (the spicy vegetable) with a fresh orange carrot"

    # 输出路径
    output_path = "./output_009_carrot.png"

    # GPT-4o API 配置（需要配置）
    api_key = "your_api_key"  # ⚠️ 请替换为你的API密钥
    api_version = "2024-08-01-preview"
    end_point = "https://your-endpoint.openai.azure.com/"  # ⚠️ 请替换为你的端点
    engine = "4o"

    # ============================================
    # 检查配置
    # ============================================
    if api_key == "your_api_key" or end_point == "https://your-endpoint.openai.azure.com/":
        print("\n⚠️  警告：请先配置 GPT-4o API 参数！")
        print("   修改脚本中的 api_key 和 end_point 变量")
        print("\n或者使用简单推理模式（需要手动提供mask）")
        sys.exit(1)

    if not os.path.exists(input_image_path):
        print(f"\n❌ 错误：输入图片不存在: {input_image_path}")
        sys.exit(1)

    # ============================================
    # 加载模型
    # ============================================
    pipe, groundingdino_model, lisa_model, tokenizer, device = load_models()

    # ============================================
    # 执行编辑
    # ============================================
    try:
        result_image, mask, target_prompt = edit_009_image(
            pipe=pipe,
            groundingdino_model=groundingdino_model,
            lisa_model=lisa_model,
            tokenizer=tokenizer,
            device=device,
            input_image_path=input_image_path,
            prompt=prompt,
            output_path=output_path,
            api_key=api_key,
            api_version=api_version,
            end_point=end_point,
            engine=engine
        )

        print("\n🎉 编辑成功完成！")
        print(f"   结果图片: {output_path}")
        print(f"   Mask图片: {output_path.replace('.png', '_mask.png')}")
        print(f"   目标提示词: {target_prompt}")

    except Exception as e:
        print(f"\n❌ 编辑失败: {e}")
        import traceback
        traceback.print_exc()
