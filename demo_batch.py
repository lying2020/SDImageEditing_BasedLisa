#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量图像编辑 Demo
从 JSON 文件读取多个例子并批量处理

JSON 文件格式：
[
    {
        "sample_id": "sample_00001_10_change_simple_action_109",
        "image_path": "images/10_change_simple_action_109.png",
        "image_mask_path": "mask/10_change_simple_action_109.png",
        "instruction": "A woman is holding a bouquet of flowers and smiling.",
        "editing_type": "Local"  // 可选，如果不提供则使用默认值
    },
    ...
]

输出：
- 所有编辑后的图片保存在 output/ 目录
- 每个样本生成 {sample_id}_result.png 和 {sample_id}_mask.png
"""

import os
import sys
import json
import torch
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# 添加 src 目录到路径（包含修改后的 diffusers）
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 设置模型路径
os.environ["SMARTFREEEDIT_MODEL_PATH"] = "/home/liying/Documents/smart_free_edit_huggingface"

# 导入必要的模块
from diffusers.pipelines.brushnet.pipeline_brushnet import StableDiffusionBrushNetPipeline
from diffusers.models import BrushNetModel
from diffusers.schedulers import UniPCMultistepScheduler
from SmartFreeEdit.src.smartfreeedit_all_pipeline import SmartFreeEdit_Pipeline

# 编辑类型定义（与 demo_single.py 保持一致）
EDITING_TYPES = {
    "Addition": "添加新对象到图像中，例如：add a bird, add a car in the background",
    "Remove": "删除图像中的对象，例如：remove the car, remove the person",
    "Local": "替换局部对象或改变对象属性，例如：change the cat to a dog, make it smile, replace the red apple with a green apple",
    "Global": "编辑整个图像，例如：let's see it in winter, Change the season from autumn to spring",
    "Background": "改变场景背景，例如：change the background to a beach, make the hedgehog in France",
    "Resize": "调整对象大小，例如：minify the giraffe in the image, make the car bigger"
}
DEFAULT_EDITING_TYPE = "Local"


def load_models():
    """加载 BrushNet 和基础模型（与 demo_single.py 保持一致）"""
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
    编辑单个图像（与 demo_single.py 保持一致）
    """
    # 检查文件是否存在
    if not os.path.exists(source_image_path):
        raise FileNotFoundError(f"源图片不存在: {source_image_path}")
    if not os.path.exists(source_mask_path):
        raise FileNotFoundError(f"Mask图片不存在: {source_mask_path}")

    # 验证编辑类型
    if editing_type not in EDITING_TYPES:
        editing_type = DEFAULT_EDITING_TYPE

    # 加载图片
    original_image = np.array(Image.open(source_image_path).convert("RGB"))
    original_mask = np.array(Image.open(source_mask_path).convert("RGB"))

    # 如果 mask 是彩色图，转换为灰度图（取第一个通道）
    if original_mask.ndim == 3:
        original_mask = original_mask[:, :, 0]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名
    if sample_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_id = f"sample_{timestamp}"

    output_path = os.path.join(output_dir, f"{sample_id}_result.png")
    mask_save_path = os.path.join(output_dir, f"{sample_id}_mask.png")

    # 执行编辑
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

    return output_path, mask_save_path


def load_json_samples(json_path, base_dir=None):
    """
    从 JSON 文件加载样本列表

    参数:
        json_path: JSON 文件路径
        base_dir: 基础目录（用于解析相对路径，如果为None则使用JSON文件所在目录）

    返回:
        样本列表
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # 如果 base_dir 为 None，使用 JSON 文件所在目录
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(json_path))

    # 处理相对路径
    for sample in samples:
        # 处理 image_path
        if not os.path.isabs(sample.get("image_path", "")):
            sample["image_path"] = os.path.join(base_dir, sample["image_path"])

        # 处理 image_mask_path
        if not os.path.isabs(sample.get("image_mask_path", "")):
            sample["image_mask_path"] = os.path.join(base_dir, sample["image_mask_path"])

    return samples


def process_batch(
    json_path,
    output_dir="output",
    base_dir=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    control_strength=1.0,
    negative_prompt="ugly, low quality, distorted, blurry",
    blending=True,
    skip_existing=False,
):
    """
    批量处理图像编辑

    参数:
        json_path: JSON 文件路径
        output_dir: 输出目录（默认：output）
        base_dir: 基础目录（用于解析相对路径，如果为None则使用JSON文件所在目录）
        num_inference_steps: 推理步数（默认：50）
        guidance_scale: 引导强度（默认：7.5）
        control_strength: 控制强度（默认：1.0）
        negative_prompt: 负面提示词（默认："ugly, low quality, distorted, blurry"）
        blending: 是否混合（默认：True）
        skip_existing: 是否跳过已存在的文件（默认：False）

    返回:
        处理结果统计
    """
    print("=" * 60)
    print("批量图像编辑 Demo")
    print("=" * 60)

    # 加载样本
    print(f"\n正在加载样本列表: {json_path}")
    samples = load_json_samples(json_path, base_dir)
    print(f"✅ 加载了 {len(samples)} 个样本")

    # 加载模型（只加载一次）
    pipe = load_models()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 统计信息
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_samples = []

    # 批量处理
    print(f"\n开始批量处理...")
    print("=" * 60)

    for idx, sample in enumerate(tqdm(samples, desc="处理进度")):
        sample_id = sample.get("sample_id", f"sample_{idx:05d}")
        image_path = sample.get("image_path", "")
        image_mask_path = sample.get("image_mask_path", "")
        instruction = sample.get("instruction", "")
        editing_type = sample.get("editing_type", DEFAULT_EDITING_TYPE)

        # 检查必要字段
        if not image_path or not image_mask_path or not instruction:
            print(f"\n⚠️  样本 {sample_id} 缺少必要字段，跳过")
            failed_count += 1
            failed_samples.append({
                "sample_id": sample_id,
                "reason": "缺少必要字段（image_path, image_mask_path, instruction）"
            })
            continue

        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"\n⚠️  样本 {sample_id} 的图片不存在: {image_path}")
            failed_count += 1
            failed_samples.append({
                "sample_id": sample_id,
                "reason": f"图片不存在: {image_path}"
            })
            continue

        if not os.path.exists(image_mask_path):
            print(f"\n⚠️  样本 {sample_id} 的mask不存在: {image_mask_path}")
            failed_count += 1
            failed_samples.append({
                "sample_id": sample_id,
                "reason": f"Mask不存在: {image_mask_path}"
            })
            continue

        # 检查是否已存在（如果启用跳过）
        output_path = os.path.join(output_dir, f"{sample_id}_result.png")
        if skip_existing and os.path.exists(output_path):
            print(f"\n⏭️  样本 {sample_id} 已存在，跳过")
            skipped_count += 1
            continue

        # 处理样本
        try:
            result_path, mask_path = edit_image(
                pipe=pipe,
                source_image_path=image_path,
                source_mask_path=image_mask_path,
                prompt=instruction,
                editing_type=editing_type,
                output_dir=output_dir,
                sample_id=sample_id,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                control_strength=control_strength,
                negative_prompt=negative_prompt,
                blending=blending,
            )
            success_count += 1

        except Exception as e:
            print(f"\n❌ 样本 {sample_id} 处理失败: {e}")
            failed_count += 1
            failed_samples.append({
                "sample_id": sample_id,
                "reason": str(e)
            })

    # 打印统计信息
    print("\n" + "=" * 60)
    print("批量处理完成！")
    print("=" * 60)
    print(f"总样本数: {len(samples)}")
    print(f"✅ 成功: {success_count}")
    print(f"⏭️  跳过: {skipped_count}")
    print(f"❌ 失败: {failed_count}")

    if failed_samples:
        print("\n失败的样本：")
        for failed in failed_samples:
            print(f"  - {failed['sample_id']}: {failed['reason']}")

    # 保存失败列表到文件
    if failed_samples:
        failed_json_path = os.path.join(output_dir, "failed_samples.json")
        with open(failed_json_path, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, indent=2, ensure_ascii=False)
        print(f"\n失败列表已保存到: {failed_json_path}")

    return {
        "total": len(samples),
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "failed_samples": failed_samples
    }


if __name__ == "__main__":
    print("=" * 60)
    print("批量图像编辑 Demo")
    print("=" * 60)
    print("\n支持的编辑类型：")
    for edit_type, description in EDITING_TYPES.items():
        marker = " (默认)" if edit_type == DEFAULT_EDITING_TYPE else ""
        print(f"  - {edit_type}{marker}: {description}")

    # ============================================
    # 配置参数（修改这里以运行）
    # ============================================
    json_path = "samples.json"  # JSON 文件路径
    output_dir = "output"  # 输出目录
    base_dir = None  # 基础目录（用于解析相对路径，None表示使用JSON文件所在目录）

    # 高级参数（可选，使用默认值即可）
    num_inference_steps = 50
    guidance_scale = 7.5
    control_strength = 1.0
    negative_prompt = "ugly, low quality, distorted, blurry"
    blending = True
    skip_existing = False  # 是否跳过已存在的文件

    # ============================================
    # 检查配置
    # ============================================
    if not os.path.exists(json_path):
        print(f"\n❌ 错误：JSON文件不存在: {json_path}")
        print("   请创建 JSON 文件，格式如下：")
        print("""
[
    {
        "sample_id": "sample_00001_10_change_simple_action_109",
        "image_path": "images/10_change_simple_action_109.png",
        "image_mask_path": "mask/10_change_simple_action_109.png",
        "instruction": "A woman is holding a bouquet of flowers and smiling.",
        "editing_type": "Local"
    }
]
        """)
        sys.exit(1)

    # ============================================
    # 执行批量处理
    # ============================================
    try:
        stats = process_batch(
            json_path=json_path,
            output_dir=output_dir,
            base_dir=base_dir,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            control_strength=control_strength,
            negative_prompt=negative_prompt,
            blending=blending,
            skip_existing=skip_existing,
        )

        print("\n🎉 批量处理完成！")
        print(f"   输出目录: {output_dir}")

    except Exception as e:
        print(f"\n❌ 批量处理失败: {e}")
        import traceback
        traceback.print_exc()
