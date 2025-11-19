"""
本地模型路径配置文件
修改此文件以使用你的本地模型路径
"""
import os

# ============================================
# 模型路径配置
# ============================================
# 修改为你的模型目录路径
SMARTFREEEDIT_MODEL_PATH = "/home/liying/Documents/smart_free_edit_huggingface"

# 如果路径不存在，是否自动从HuggingFace下载
AUTO_DOWNLOAD_IF_NOT_EXISTS = False

# ============================================
# 子模型路径（通常不需要修改）
# ============================================
BASE_MODEL_PATH = os.path.join(SMARTFREEEDIT_MODEL_PATH, "base_model")
BRUSHNET_PATH = os.path.join(SMARTFREEEDIT_MODEL_PATH, "checkpoint-100000/brushnet")
LISA_PATH = os.path.join(SMARTFREEEDIT_MODEL_PATH, "LISA-7B-v1-explanatory")
GROUNDINGDINO_PATH = os.path.join(SMARTFREEEDIT_MODEL_PATH, "grounding_dino/groundingdino_swint_ogc.pth")

# Stable Diffusion v1.5 路径
# 如果 smart_free_edit_huggingface 目录下的模型未下载完整，可以使用其他路径
STABLE_DIFFUSION_V1_5_PATH = "/home/liying/Documents/stable-diffusion-v1-5"
# 如果上面的路径不存在，尝试使用 smart_free_edit_huggingface 目录下的
if not os.path.exists(STABLE_DIFFUSION_V1_5_PATH):
    STABLE_DIFFUSION_V1_5_PATH = os.path.join(SMARTFREEEDIT_MODEL_PATH, "stable-diffusion-v1-5")

# 默认使用的base model
DEFAULT_BASE_MODEL = "realisticVisionV60B1_v51VAE"
DEFAULT_BASE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, DEFAULT_BASE_MODEL)

# ============================================
# 验证路径是否存在
# ============================================
def check_paths():
    """检查关键路径是否存在"""
    paths_to_check = {
        "Base Model Path": BASE_MODEL_PATH,
        "BrushNet Path": BRUSHNET_PATH,
        "LISA Path": LISA_PATH,
        "GroundingDINO Path": GROUNDINGDINO_PATH,
        "Default Base Model": DEFAULT_BASE_MODEL_PATH,
        "Stable Diffusion v1.5": STABLE_DIFFUSION_V1_5_PATH,
    }

    missing_paths = []
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            missing_paths.append(f"{name}: {path}")
        else:
            print(f"✅ {name}: {path}")

    if missing_paths:
        print("\n⚠️  以下路径不存在：")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    return True

if __name__ == "__main__":
    print("检查模型路径配置...")
    print(f"模型根目录: {SMARTFREEEDIT_MODEL_PATH}\n")
    check_paths()
