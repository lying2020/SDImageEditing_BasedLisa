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
import matplotlib.colors as colors

current_dir = os.path.dirname(os.path.abspath(__file__))

# pretrained model path
INPAINTING_MODEL_PATH = "/home/liying/Documents/stable-diffusion-inpainting"
INPAINTING_2_MODEL_PATH = "/home/liying/Documents/stable-diffusion-2-inpainting"

STABLE_DIFFUSION_V1_5_MODEL_PATH = "/home/liying/Documents/stable-diffusion-v1-5"

# 模型路径
BASE_MODEL_PATH = "/home/liying/Documents/smart_free_edit_huggingface/base_model/realisticVisionV60B1_v51VAE"
BRUSHNET_PATH = "/home/liying/Documents/smart_free_edit_huggingface/checkpoint-100000/brushnet"
LISA_PATH = "/home/liying/Documents/smart_free_edit_huggingface/LISA-7B-v1-explanatory"


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
if not os.path.exists(IMAGES_JSON_FILE):
    raise FileNotFoundError(f"Images JSON file not found at {IMAGES_JSON_FILE}")


# result path
OUTPUT_DIR = os.path.join(current_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDITING_RESULTS_DIR = os.path.join(OUTPUT_DIR, tmp_name)
os.makedirs(EDITING_RESULTS_DIR, exist_ok=True)
