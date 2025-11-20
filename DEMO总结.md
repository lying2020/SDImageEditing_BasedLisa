# Demo 脚本总结

## ✅ 已完成的工作

已创建两个简化的图像编辑 Demo 脚本：

### 1. `demo_single.py` - 单个例子编辑

**功能**：
- 输入单个图片、mask 和提示词进行编辑
- 不需要 OpenAI API、GPT-4o、GroundingDINO
- 只需要 BrushNet 和基础模型

**主要参数**：
- `source_image_path`: 源图片路径
- `source_mask_path`: Mask图片路径（白色=编辑区域）
- `prompt`: 编辑提示词
- `editing_type`: 编辑类型（可选，默认："Local"）
- `sample_id`: 样本ID（可选，用于命名输出）

**输出**：
- `output/{sample_id}_result.png` - 编辑后的图片
- `output/{sample_id}_mask.png` - Mask图片

### 2. `demo_batch.py` - 批量编辑

**功能**：
- 从 JSON 文件读取多个样本并批量处理
- 自动处理相对路径和绝对路径
- 提供进度条和错误统计

**JSON 格式**：
```json
[
    {
        "sample_id": "sample_00001",
        "image_path": "images/image.png",
        "image_mask_path": "mask/mask.png",
        "instruction": "your editing instruction",
        "editing_type": "Local"
    }
]
```

**输出**：
- 每个样本生成 `{sample_id}_result.png` 和 `{sample_id}_mask.png`
- 失败样本记录在 `failed_samples.json`

## 📋 编辑类型

支持的编辑类型（`editing_type` 参数）：

1. **Addition** - 添加新对象
2. **Remove** - 删除对象
3. **Local** (默认) - 局部编辑/替换
4. **Global** - 全局编辑
5. **Background** - 背景替换
6. **Resize** - 调整大小

## 🎯 使用步骤

### 单个例子

1. 修改 `demo_single.py` 中的参数：
   ```python
   source_image_path = "/path/to/image.png"
   source_mask_path = "/path/to/mask.png"
   prompt = "your instruction"
   ```

2. 运行：
   ```bash
   python3 demo_single.py
   ```

### 批量处理

1. 创建 JSON 文件（参考 `samples_example.json`）

2. 修改 `demo_batch.py` 中的参数：
   ```python
   json_path = "samples.json"
   output_dir = "output"
   ```

3. 运行：
   ```bash
   python3 demo_batch.py
   ```

## ✨ 特点

- ✅ **无外部 API 依赖** - 完全本地运行
- ✅ **最小化模型依赖** - 只需要 BrushNet 和基础模型
- ✅ **简单易用** - 清晰的参数和文档
- ✅ **灵活配置** - 支持多种编辑类型和高级参数

## 📁 文件清单

- `demo_single.py` - 单个例子编辑脚本
- `demo_batch.py` - 批量编辑脚本
- `samples_example.json` - JSON 格式示例
- `DEMO使用说明.md` - 详细使用文档

## ⚠️ 注意事项

1. **模型路径**：确保模型路径正确配置（通过 `config_local.py` 或环境变量）
2. **Mask 格式**：白色区域（255）表示编辑区域，黑色区域（0）表示保持不变
3. **提示词**：使用简单、清晰的英文提示词，描述想要的结果

---

**创建时间**：2025年
