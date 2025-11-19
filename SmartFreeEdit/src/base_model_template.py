import os
import torch
from huggingface_hub import snapshot_download

from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler



torch_dtype = torch.float16
device = "cpu"

# 优先使用环境变量，否则使用本地配置
SmartFreeEdit_path = os.getenv("SMARTFREEEDIT_MODEL_PATH", None)
if SmartFreeEdit_path is None:
    try:
        from SmartFreeEdit.config_local import SMARTFREEEDIT_MODEL_PATH
        SmartFreeEdit_path = SMARTFREEEDIT_MODEL_PATH
    except ImportError:
        SmartFreeEdit_path = "/home/liying/Documents/smart_free_edit_huggingface"  # 默认路径

brushnet_path = os.path.join(SmartFreeEdit_path, "checkpoint-100000/brushnet")
# 延迟加载brushnet，避免在导入时就加载模型
brushnet = None  # 将在需要时加载


# 延迟加载模型，避免在导入时就加载所有模型
def _get_base_models_list():
    """动态获取base models列表，延迟加载模型"""
    global brushnet
    if brushnet is None:
        brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype)

    base_models_list = [
        {
            "name": "henmixReal (Preload)",
            "local_path": os.path.join(SmartFreeEdit_path, "base_model/henmixReal_v5c"),
            "pipe": ""  # 延迟加载
        },
        {
            "name": "meinamix (Preload)",
            "local_path": os.path.join(SmartFreeEdit_path, "base_model/meinamix_meinaV11"),
            "pipe": ""  # 延迟加载
        },
        {
            "name": "realisticVision (Default)",
            "local_path": os.path.join(SmartFreeEdit_path, "base_model/realisticVisionV60B1_v51VAE"),
            "pipe": ""  # 延迟加载
        },
    ]
    return base_models_list

base_models_list = _get_base_models_list()

base_models_template = {k["name"]: (k["local_path"], k["pipe"]) for k in base_models_list}
