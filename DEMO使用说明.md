# Demo 使用说明

## 📋 概述

本项目提供了两个简化的图像编辑 Demo：

1. **`demo_single.py`** - 单个例子图像编辑
2. **`demo_batch.py`** - 批量图像编辑

### ✨ 特点

- ✅ **不需要 OpenAI API** - 完全本地运行
- ✅ **不需要 GPT-4o** - 不需要 VLM 模型
- ✅ **不需要 GroundingDINO** - 不需要对象检测模型
- ✅ **只需要 BrushNet 和基础模型** - 最小化依赖

## 🎯 编辑类型

支持的编辑类型（`editing_type` 参数）：

| 类型 | 说明 | 示例 |
|------|------|------|
| **Addition** | 添加新对象 | "add a bird", "add a car in the background" |
| **Remove** | 删除对象 | "remove the car", "remove the person" |
| **Local** (默认) | 局部编辑/替换 | "change the cat to a dog", "replace the green pepper with a carrot" |
| **Global** | 全局编辑 | "let's see it in winter", "Change the season from autumn to spring" |
| **Background** | 背景替换 | "change the background to a beach", "make the hedgehog in France" |
| **Resize** | 调整大小 | "minify the giraffe in the image", "make the car bigger" |

## 📝 Demo 1: 单个例子编辑

### 使用方法

1. **修改脚本中的参数**：

```python
source_image_path = "/path/to/your/image.png"
source_mask_path = "/path/to/your/mask.png"
prompt = "your editing instruction"
editing_type = "Local"  # 可选
sample_id = "my_sample"  # 可选
```

2. **运行脚本**：

```bash
python3 demo_single.py
```

### 输入参数

- **`source_image_path`** (必需): 源图片路径
- **`source_mask_path`** (必需): Mask图片路径（白色区域表示要编辑的区域）
- **`prompt`** (必需): 编辑提示词
- **`editing_type`** (可选): 编辑类型，默认值：`"Local"`
- **`sample_id`** (可选): 样本ID，用于命名输出文件

### 高级参数（可选）

- `num_inference_steps`: 推理步数（默认：50）
- `guidance_scale`: 引导强度（默认：7.5）
- `control_strength`: 控制强度（默认：1.0）
- `negative_prompt`: 负面提示词（默认："ugly, low quality, distorted, blurry"）
- `blending`: 是否混合（默认：True）

### 输出

- 结果图片：`output/{sample_id}_result.png`
- Mask图片：`output/{sample_id}_mask.png`

## 📝 Demo 2: 批量编辑

### 使用方法

1. **创建 JSON 文件**（参考 `samples_example.json`）：

```json
[
    {
        "sample_id": "sample_00001_10_change_simple_action_109",
        "image_path": "images/10_change_simple_action_109.png",
        "image_mask_path": "mask/10_change_simple_action_109.png",
        "instruction": "A woman is holding a bouquet of flowers and smiling.",
        "editing_type": "Local"
    },
    {
        "sample_id": "sample_00002",
        "image_path": "/absolute/path/to/image.png",
        "image_mask_path": "/absolute/path/to/mask.png",
        "instruction": "replace the green pepper with a carrot",
        "editing_type": "Local"
    }
]
```

2. **修改脚本中的参数**：

```python
json_path = "samples.json"  # JSON 文件路径
output_dir = "output"  # 输出目录
base_dir = None  # 基础目录（用于解析相对路径）
```

3. **运行脚本**：

```bash
python3 demo_batch.py
```

### JSON 文件格式

每个样本包含以下字段：

- **`sample_id`** (必需): 样本ID，用于命名输出文件
- **`image_path`** (必需): 图片路径（可以是相对路径或绝对路径）
- **`image_mask_path`** (必需): Mask图片路径（可以是相对路径或绝对路径）
- **`instruction`** (必需): 编辑指令/提示词
- **`editing_type`** (可选): 编辑类型，如果不提供则使用默认值 `"Local"`

### 路径说明

- **相对路径**：相对于 JSON 文件所在目录（如果 `base_dir=None`）或指定的 `base_dir`
- **绝对路径**：直接使用完整路径

### 输出

- 每个样本生成两个文件：
  - `output/{sample_id}_result.png` - 编辑后的图片
  - `output/{sample_id}_mask.png` - Mask图片
- 如果处理失败，会生成 `output/failed_samples.json` 记录失败的样本

### 高级参数（可选）

- `num_inference_steps`: 推理步数（默认：50）
- `guidance_scale`: 引导强度（默认：7.5）
- `control_strength`: 控制强度（默认：1.0）
- `negative_prompt`: 负面提示词（默认："ugly, low quality, distorted, blurry"）
- `blending`: 是否混合（默认：True）
- `skip_existing`: 是否跳过已存在的文件（默认：False）

## 🔧 依赖要求

### 必需依赖

- Python 3.10+
- PyTorch 2.0.1+ (with CUDA)
- diffusers (项目中的修改版本)
- Pillow
- numpy
- tqdm (仅批量处理需要)

### 模型要求

- BrushNet 模型：`/home/liying/Documents/smart_free_edit_huggingface/checkpoint-100000/brushnet`
- 基础模型：`/home/liying/Documents/smart_free_edit_huggingface/base_model/realisticVisionV60B1_v51VAE`

### 不需要的依赖

- ❌ OpenAI API
- ❌ GPT-4o
- ❌ GroundingDINO
- ❌ LISA 模型

## 📌 注意事项

1. **Mask 图片格式**：
   - 白色区域（255）表示要编辑的区域
   - 黑色区域（0）表示保持不变
   - 可以是灰度图或彩色图（会自动转换为灰度图）

2. **图片尺寸**：
   - 建议使用 512x512 或 1024x1024 的图片
   - 系统会自动调整尺寸以适配模型

3. **提示词编写**：
   - 使用简单、清晰的英文提示词
   - 描述你想要的结果，而不是过程
   - 例如："a carrot" 而不是 "replace with a carrot"

4. **编辑类型选择**：
   - 如果不确定，使用默认值 `"Local"`
   - 编辑类型主要用于文档记录，不影响实际编辑效果

## 🐛 常见问题

### Q: 模型加载失败

**A**: 检查模型路径是否正确，确保：
- BrushNet 模型文件已完整下载（不是 Git LFS 指针文件）
- 基础模型路径正确

### Q: CUDA 内存不足

**A**: 尝试：
- 减小图片尺寸
- 减少 `num_inference_steps`
- 使用 `pipe.enable_model_cpu_offload()`（已默认启用）

### Q: 输出结果不理想

**A**: 尝试：
- 调整 `guidance_scale`（增加或减少）
- 调整 `control_strength`（控制 mask 区域的影响强度）
- 改进提示词（更具体、更清晰）
- 检查 mask 是否准确

## 📚 示例

### 示例1：替换对象

```python
source_image_path = "image.png"
source_mask_path = "mask.png"  # 白色区域标记要替换的对象
prompt = "a carrot"
editing_type = "Local"
```

### 示例2：添加对象

```python
source_image_path = "image.png"
source_mask_path = "mask.png"  # 白色区域标记要添加的位置
prompt = "a bird flying in the sky"
editing_type = "Addition"
```

### 示例3：删除对象

```python
source_image_path = "image.png"
source_mask_path = "mask.png"  # 白色区域标记要删除的对象
prompt = "empty background"
editing_type = "Remove"
```

---

**最后更新**：2025年
