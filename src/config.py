# config.py
import torch
import os
from datetime import datetime

# === Model Parameters ===
N_FEATURES = 18    # 입력 변수 개수 (date, passorfail 제외)
N_HIDDENS = 200
N_LAYERS = 3
BATCH_SIZE = 512
EPOCHS = 100

# === Training Parameters ===
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
DEVICE = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# === Path Configuration ===
DATA_PATH = "/dshome/ddualab/gyu/kamp/data/2. 소성가공 품질보증 AI 데이터셋.csv"
OUTPUT_DIR = "./checkpoint"

# === Dynamic Subfolder for Checkpoints ===
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = os.path.join(OUTPUT_DIR, f"model_{TIMESTAMP}")
os.makedirs(SAVE_PATH, exist_ok=True)
