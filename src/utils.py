# ================================================
# utils_deformation.py — 소성가공 품질보증 데이터셋 전처리
# ================================================
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1️⃣ 데이터 로드 및 기본 전처리
# ============================================================
def load_and_preprocess_deformation(root):
    """
    Load and preprocess deformation dataset
    - CSV 로드
    - 'date' → datetime 변환
    - 'passorfail' → label로 변경 및 결측치 처리
    - Feature 정규화(mean/std)
    """
    data = pd.read_csv(root)
    data = data.rename(columns={"passorfail": "label"})
    data["label"] = data["label"].fillna(0)
    data["Timestamp"] = pd.to_datetime(data["date"])
    data = data.set_index("Timestamp").sort_index()

    # Feature 선택 및 정규화
    feature = data.iloc[:, 1:19]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)
    norm_feature = (feature - mean_df) / std_df
    norm_feature = norm_feature.dropna(axis=1)

    n_sensor = len(norm_feature.columns)
    return data, norm_feature, n_sensor


# ============================================================
# 2️⃣ Train / Validation / Test Split
# ============================================================
def split_deformation(data, features, ratios=(0.7, 0.15, 0.15)):
    """
    시간 순서를 기준으로 비율에 따라 데이터 분할
    """
    n = len(data)
    n_train = int(n * ratios[0])
    n_val = int(n * (ratios[0] + ratios[1]))

    train_df = features.iloc[:n_train]
    val_df = features.iloc[n_train:n_val]
    test_df = features.iloc[n_val:]

    train_label = data.label.iloc[:n_train]
    val_label = data.label.iloc[n_train:n_val]
    test_label = data.label.iloc[n_val:]

    print("✅ Split complete")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return (train_df, val_df, test_df), (train_label, val_label, test_label)


# ============================================================
# 3️⃣ Dataset 클래스 정의
# ============================================================
class DeformationDataset(Dataset):
    """
    기본 Deformation Dataset (label 없이 reconstruction 학습용)
    """
    def __init__(self, df, label, window_size=40, stride_size=10):
        super().__init__()
        self.df = df
        self.label = label.reset_index(drop=True)  # ✅ 인덱스 리셋 추가
        self.window_size = window_size
        self.stride_size = stride_size
        self.data, self.idx, self.label = self._preprocess(df, self.label)

    def _preprocess(self, df, label):
        if len(df) < self.window_size:
            return df.values, np.array([], dtype=int), np.array([], dtype=float)
        start_idx = np.arange(0, len(df) - self.window_size, self.stride_size)
        return df.values, start_idx, label.reset_index(drop=True)[start_idx]  # ✅ 다시 reset_index

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size, -1, 1])
        return torch.FloatTensor(data).transpose(0, 1), float(self.label.iloc[index])  # ✅ iloc로 안전 접근



class DeformationLabelDataset(DeformationDataset):
    """
    Reconstruction + 라벨 포함 학습용 Dataset
    (1 → -1, 0 → 1 변환)
    """
    def __init__(self, df, label, window_size=40, stride_size=10):
        super().__init__(df, label, window_size, stride_size)
        self.label = 1.0 - 2 * self.label  # 1→-1, 0→1 변환
        self.label = pd.Series(self.label).reset_index(drop=True)  # ✅ 인덱스 리셋 추가

    def __getitem__(self, index):
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size, -1, 1])
        return torch.FloatTensor(data).transpose(0, 1), float(self.label.iloc[index])  # ✅ iloc 사용



# ============================================================
# 4️⃣ 통합 데이터 로더 생성 함수
# ============================================================
def get_deformation_dataloaders(root, batch_size=512, label=False):
    """
    전체 파이프라인: Load → Split → Dataset → Dataloader
    """
    data, features, n_sensor = load_and_preprocess_deformation(root)
    (train_df, val_df, test_df), (train_label, val_label, test_label) = split_deformation(data, features)

    # ✅ 데이터 길이에 따라 window_size 자동 조정
    window_size = min(60, max(10, len(train_df) // 10))  # 데이터의 1/10 ~ 60 사이로 제한
    stride_size = 10

    # === Dataset 생성 ===
    # 🔹 train, val → Reconstruction 학습용 (라벨 X)
    train_ds = DeformationDataset(train_df, train_label, window_size=window_size, stride_size=stride_size)
    val_ds = DeformationDataset(val_df, val_label, window_size=window_size, stride_size=stride_size)

    # 🔹 test → Reconstruction + 라벨 포함 (평가용)
    test_ds = DeformationLabelDataset(test_df, test_label, window_size=window_size, stride_size=stride_size)

    # === 유효성 검사 ===
    for name, ds in zip(["train", "val", "test"], [train_ds, val_ds, test_ds]):
        if len(ds) == 0:
            print(f"⚠️ Warning: {name} dataset has 0 valid windows. "
                f"(len={len(eval(name+'_df'))}, window={window_size}, stride={stride_size})")

    # === DataLoader 생성 ===
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor
