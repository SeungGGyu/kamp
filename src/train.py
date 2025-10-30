# ============================================
# train.py — Reconstruction 기반 이상탐지 학습 (tqdm + logging.info)
# ============================================
import os
import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
from torch.nn.utils import clip_grad_value_

# 내부 모듈
from config import *
from model import StackedLSTM
from utils import get_deformation_dataloaders


# =========================================================
# 전역 로깅 설정
# =========================================================
def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("📜 Logger initialized. Logs will be saved to %s", log_file)


# =========================================================
# 학습 메인 함수
# =========================================================
def main():
    # === 환경 설정 ===
    device = DEVICE
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED)

    setup_logging(SAVE_PATH)

    logging.info("🚀 Device: %s", device)
    logging.info("📂 Data: %s", DATA_PATH)
    logging.info("💾 Save Path: %s", SAVE_PATH)

    # =========================================================
    # 1️⃣ 데이터셋 준비
    # =========================================================
    logging.info("📊 Loading deformation dataloaders...")
    train_loader, valid_loader, test_loader, n_sensor = get_deformation_dataloaders(
        root=DATA_PATH,
        batch_size=BATCH_SIZE,
        label=False   # Reconstruction 학습용
    )

    logging.info("✅ Dataloaders ready (Train=%d | Val=%d | Test=%d)",
                 len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset))

    # =========================================================
    # 2️⃣ 모델 정의
    # =========================================================
    model = StackedLSTM(input_dim=N_FEATURES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()

    best_loss = np.inf
    history = []

    # =========================================================
    # 3️⃣ 학습 루프
    # =========================================================
    logging.info("🧠 Start training for %d epochs", EPOCHS)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()

            # === 배치 데이터 처리 ===
            if isinstance(batch, (list, tuple)):
                given = batch[0].to(device)
            else:
                given = batch.to(device)

            # === 입력 차원 정리 ===
            if given.ndim == 4:
                given = given.squeeze(-1)
            if given.size(1) < given.size(2):
                given = given.transpose(1, 2)

            # === Reconstruction ===
            guess = model(given)
            target = given[:, -1, :] if given.ndim == 3 else given
            loss = criterion(guess, target)

            loss.backward()
            clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"Train Loss": f"{loss.item():.6f}"})

        avg_train_loss = train_loss / len(train_loader)
        history.append(avg_train_loss)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                if isinstance(batch, (list, tuple)):
                    given = batch[0].to(device)
                else:
                    given = batch.to(device)

                if given.ndim == 4:
                    given = given.squeeze(-1)
                if given.size(1) < given.size(2):
                    given = given.transpose(1, 2)

                guess = model(given)
                target = given[:, -1, :] if given.ndim == 3 else given
                loss = criterion(guess, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        logging.info("[Epoch %03d] Train Loss: %.6f | Val Loss: %.6f",
                     epoch, avg_train_loss, avg_val_loss)

        # === Best Model 저장 ===
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model_best.pt"))
            logging.info("💾 Best model saved at epoch %d (Val Loss=%.6f)", epoch, best_loss)

    # =========================================================
    # 4️⃣ 학습 완료 로그
    # =========================================================
    logging.info("✅ Training complete!")
    logging.info("🏆 Best Validation Loss: %.6f", best_loss)

    torch.save({
        "state": model.state_dict(),
        "history": history,
        "best_loss": best_loss
    }, os.path.join(SAVE_PATH, "final_checkpoint.pt"))

    logging.info("🧾 Final checkpoint saved at: %s", SAVE_PATH)


if __name__ == "__main__":
    main()
