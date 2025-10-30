# ============================================
# inference.py — Reconstruction 기반 이상탐지 평가 + 시각화 + 라벨 매칭
# ============================================
import os
import glob
import torch
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)

from model import StackedLSTM
from utils import get_deformation_dataloaders
from config import *


# =========================================================
# Logger 설정
# =========================================================
def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "inference.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info("📜 Logger initialized. Logs saved to %s", log_file)


# =========================================================
# Reconstruction 기반 Anomaly Score 계산
# =========================================================
def evaluate_reconstruction(dataset, model, batch_size=512, device=DEVICE):
    """
    ✅ Reconstruction 기반 이상탐지 inference 함수
    - dataset 구조(dict, tuple, tensor 모두) 자동 인식
    - reconstruction error (MSE per sample) 계산
    """
    model.eval()
    all_errors, all_labels = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="🔍 Evaluating", ncols=90, leave=False):

            # ✅ batch 형태 자동 판별
            if isinstance(batch, dict):
                given = batch.get("given", None)
                labels = batch.get("label", None)
            elif isinstance(batch, (list, tuple)):
                given = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            elif torch.is_tensor(batch):  # ←🔥 지금 이 케이스!
                given = batch
                labels = None
            else:
                raise ValueError(f"❌ Unexpected batch type: {type(batch)}")

            given = given.to(device)
            labels = labels.cpu().numpy() if labels is not None else None

            # ✅ 입력 차원 정리
            # (B, 1, T, F) → (B, T, F)
            if given.ndim == 4 and given.size(1) == 1:
                given = given.squeeze(1)
            # (B, F, T, 1) → (B, T, F)
            elif given.ndim == 4 and given.size(-1) == 1:
                given = given.squeeze(-1)
            # (B, F, T) → (B, T, F)
            if given.ndim == 3 and given.size(1) < given.size(2):
                given = given.transpose(1, 2)

            # ✅ Reconstruction
            guess = model(given)
            target = given[:, -1, :] if given.ndim == 3 else given
            errors = torch.mean((guess - target) ** 2, dim=1).cpu().numpy()

            all_errors.append(errors)
            if labels is not None:
                all_labels.append(labels)

    errors = np.concatenate(all_errors)
    labels = np.concatenate(all_labels) if all_labels else None
    return errors, labels



# =========================================================
# 시각화 함수
# =========================================================
def plot_results(errors, labels, threshold, metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # --- 1️⃣ Reconstruction Error 분포 ---
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, color='skyblue', alpha=0.7, label='Reconstruction Error')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution with Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_distribution.png"))
    plt.close()

    # --- 2️⃣ ROC Curve ---
    fpr, tpr, _ = roc_curve(labels, errors)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()

    # --- 3️⃣ Confusion Matrix ---
    preds = (errors > threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # --- 4️⃣ Metric Bar Chart ---
    plt.figure(figsize=(6, 4))
    metric_names = ["AUC", "Precision", "Recall", "F1"]
    metric_values = [metrics["AUC"], metrics["Precision"], metrics["Recall"], metrics["F1"]]
    plt.bar(metric_names, metric_values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"], alpha=0.8)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_bar.png"))
    plt.close()


# =========================================================
# 메인 실행
# =========================================================
def main():
    # === 최신 학습 모델 자동 탐색 ===
    folders = sorted(glob.glob(os.path.join(OUTPUT_DIR, "model_*")), key=os.path.getmtime)
    valid_folders = [f for f in folders if os.path.exists(os.path.join(f, "model_best.pt"))]
    latest_model_dir = valid_folders[-1] if valid_folders else None
    if latest_model_dir is None:
        raise FileNotFoundError("❌ No trained model with model_best.pt found in ./checkpoint")

    setup_logging(latest_model_dir)
    logging.info("🚀 Starting inference...")
    logging.info("📂 Data: %s", DATA_PATH)
    logging.info("💾 Using model from: %s", latest_model_dir)

    # === 데이터 준비 ===
    _, _, test_loader, _ = get_deformation_dataloaders(
        root=DATA_PATH,
        batch_size=BATCH_SIZE,
        label=True
    )
    test_dataset = test_loader.dataset

    # === 모델 로드 ===
    model_path = os.path.join(latest_model_dir, "model_best.pt")
    model = StackedLSTM(input_dim=N_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    logging.info("✅ Loaded latest model: %s", model_path)

    # === 이상치 점수 계산 ===
    logging.info("🔍 Evaluating anomaly scores...")
    errors, labels = evaluate_reconstruction(test_dataset, model, batch_size=BATCH_SIZE, device=DEVICE)

    # === Threshold 설정 ===
    threshold = np.percentile(errors, 95)
    preds = (errors > threshold).astype(int)

    # === 결과 통계 ===
    mean_err = errors.mean()
    detected = np.sum(preds)
    total = len(preds)
    ratio = detected / total * 100
    logging.info("📊 Mean reconstruction error: %.6f", mean_err)
    logging.info("📈 Threshold (95th percentile): %.6f", threshold)
    logging.info("🚨 Detected anomalies: %d / %d (%.2f%%)", detected, total, ratio)

    # === 평가 지표 계산 ===
    true_anomaly = (labels == 0).astype(int)  # 0: fail → anomaly
    if len(np.unique(true_anomaly)) == 1:
        logging.warning("⚠️ ROC AUC cannot be calculated — only one class present in y_true.")
        auc_score = float('nan')
    else:
        auc_score = roc_auc_score(true_anomaly, errors)
    prec = precision_score(true_anomaly, preds)
    rec = recall_score(true_anomaly, preds)
    f1 = f1_score(true_anomaly, preds)
    metrics = {"AUC": auc_score, "Precision": prec, "Recall": rec, "F1": f1}

    total_anomalies = np.sum(true_anomaly)
    detected_anomalies = np.sum((preds == 1) & (true_anomaly == 1))
    detection_rate = detected_anomalies / total_anomalies * 100 if total_anomalies > 0 else 0

    logging.info("📈 Evaluation Metrics — AUC: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
                 auc_score, prec, rec, f1)
    logging.info("🎯 Detection Performance — Detected anomalies: %d / %d (%.2f%%)",
                 detected_anomalies, total_anomalies, detection_rate)

    # === 결과 저장 ===
    result_path = os.path.join(latest_model_dir, "anomaly_results.csv")
    result_df = pd.DataFrame({
        "Error": errors,
        "Threshold": threshold,
        "Pred_anomaly": preds,
        "Passorfail": labels,
        "Correct": (preds == true_anomaly).astype(int)
    })
    result_df.to_csv(result_path, index=False)
    logging.info("💾 Results saved to %s", result_path)

    # === 시각화 저장 ===
    plot_results(errors, true_anomaly, threshold, metrics, latest_model_dir)
    logging.info("📊 Plots saved (error distribution, ROC curve, confusion matrix, metrics bar).")

    logging.info("✅ Inference complete!")


if __name__ == "__main__":
    main()
