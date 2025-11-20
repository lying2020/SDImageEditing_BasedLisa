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

# dataset path
DATASETS_DIR = os.path.join(current_dir, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)


# result path
OUTPUT_DIR = os.path.join(current_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
