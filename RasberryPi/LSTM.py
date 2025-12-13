import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split



# =====================================================================
# 1. IMU Encoder (3 channels: ax, ay, az)
# =====================================================================

class IMUEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, T, 3]
        x = x.permute(0, 2, 1)  # [B, 3, T]
        x = self.net(x)         # [B, hidden, T]
        x = x.permute(0, 2, 1)  # [B, T, hidden]
        return self.proj(x)     # [B, T, out_dim]



# =====================================================================
# 2. Thermal Encoder (AMG8833 8×8)
# =====================================================================

class ThermalEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8→4

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 4→2

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )

        self.temporal = nn.Conv1d(64, out_dim, 3, padding=1)

    def forward(self, x):
        # x: [B, T, 1, 8, 8]
        B, T, _, H, W = x.shape
        x = x.reshape(B*T, 1, H, W)
        x = self.cnn(x).mean(dim=[2,3])  # [B*T, 64]
        x = x.reshape(B, T, 64)          # [B, T, 64]

        x = x.permute(0,2,1)             # [B, 64, T]
        x = self.temporal(x)
        return x.permute(0,2,1)          # [B, T, out_dim]



# =====================================================================
# 3. Ultrasound Encoder (front + side)
# =====================================================================

class UltrasoundEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=32, out_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, T, 2]
        x = x.permute(0,2,1)  # [B, 2, T]
        x = self.net(x)       # [B, hidden, T]
        x = x.permute(0,2,1)  # [B, T, hidden]
        return self.proj(x)



# =====================================================================
# 4. Complete Multimodal Gesture Recognition Model
# =====================================================================

class GestureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.imu = IMUEncoder()
        self.thermal = ThermalEncoder()
        self.ultra = UltrasoundEncoder()

        fusion_dim = 64 + 64 + 32

        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, imu, th, ultra):
        f1 = self.imu(imu)
        f2 = self.thermal(th)
        f3 = self.ultra(ultra)

        fused = torch.cat([f1, f2, f3], dim=-1)
        out, _ = self.lstm(fused)
        last = out[:, -1, :]
        return self.fc(last)



# =====================================================================
# 5. GestureDataset (with DS_Store filtering + normalization)
# =====================================================================

class GestureDataset(Dataset):

    def __init__(self, root, window_sec=1.0, fuse_hz=50):
        self.root = root
        self.window_sec = window_sec
        self.fuse_hz = fuse_hz
        self.T = int(window_sec * fuse_hz)

        # Filter out .DS_Store or hidden files
        all_items = os.listdir(root)
        self.classes = sorted([c for c in all_items if not c.startswith(".")])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []

        # Scan for gesture recordings
        for gesture in self.classes:
            gdir = os.path.join(root, gesture)
            imu_files = sorted(glob.glob(os.path.join(gdir, "*_imu.csv")))

            for imu_file in imu_files:
                base = imu_file[:-8]  # strip "_imu.csv"
                therm = base + "_thermal.csv"
                uf    = base + "_ultra_front.csv"
                us    = base + "_ultra_side.csv"

                if os.path.exists(therm) and os.path.exists(uf) and os.path.exists(us):
                    self.samples.append((imu_file, therm, uf, us, self.class_to_idx[gesture]))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found. Check directory structure.")

    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------
    # Resample onto 1 second, 50-step normalized timeline
    # -----------------------------------------------------
    def _resample(self, ts, vals):
        t0 = ts[0]
        t_grid = np.linspace(t0, t0 + self.window_sec, self.T, endpoint=False)

        if vals.ndim == 2:  # IMU or ultrasound
            out = np.zeros((self.T, vals.shape[1]), dtype=np.float32)
            for c in range(vals.shape[1]):
                f = interp1d(ts, vals[:, c], fill_value="extrapolate")
                out[:, c] = f(t_grid)
            return out

        # Thermal image stack: [N,1,8,8]
        N, _, H, W = vals.shape
        out = np.zeros((self.T, 1, H, W), dtype=np.float32)
        for h in range(H):
            for w in range(W):
                pix = vals[:,0,h,w]
                f = interp1d(ts, pix, fill_value="extrapolate")
                out[:,0,h,w] = f(t_grid)
        return out

    def __getitem__(self, idx):
        imu_file, ther_file, uf_file, us_file, label = self.samples[idx]

        # IMU
        imu = pd.read_csv(imu_file)
        ts = imu["timestamp_ms"].to_numpy() / 1000.0
        vals = imu[["ax","ay","az"]].to_numpy().astype(np.float32)
        imu50 = self._resample(ts, vals)

        # Thermal
        th = pd.read_csv(ther_file)
        ts2 = th["timestamp_ms"].to_numpy() / 1000.0
        cols = [f"t{i}" for i in range(64)]
        th_vals = th[cols].to_numpy().astype(np.float32).reshape(-1,1,8,8)
        th50 = self._resample(ts2, th_vals)

        # Ultrasound
        uf = pd.read_csv(uf_file)
        ts3 = uf["timestamp_ms"].to_numpy() / 1000.0
        uf_vals = uf[["front_cm"]].to_numpy().astype(np.float32)

        us = pd.read_csv(us_file)
        ts4 = us["timestamp_ms"].to_numpy() / 1000.0
        us_vals = us[["side_cm"]].to_numpy().astype(np.float32)

        uf50 = self._resample(ts3, uf_vals)
        us50 = self._resample(ts4, us_vals)
        ultra50 = np.concatenate([uf50, us50], axis=1)

        return (
            torch.tensor(imu50, dtype=torch.float32),
            torch.tensor(th50, dtype=torch.float32),
            torch.tensor(ultra50, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )



# =====================================================================
# 6. Train / Val / Test Loader Helper
# =====================================================================

def make_loaders(root, batch_size=8, train_pct=0.7, val_pct=0.15):
    dataset = GestureDataset(root)

    N = len(dataset)
    n_train = int(N * train_pct)
    n_val   = int(N * val_pct)
    n_test  = N - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print("\nDataset split:")
    print(f"  Train: {n_train}")
    print(f"  Val:   {n_val}")
    print(f"  Test:  {n_test}")
    print("\nClasses:", dataset.classes)

    return train_loader, val_loader, test_loader, dataset



# =====================================================================
# 7. Training & Evaluation
# =====================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    for imu, th, ultra, y in loader:
        imu, th, ultra, y = imu.to(device), th.to(device), ultra.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(imu, th, ultra)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imu, th, ultra, y in loader:
            imu, th, ultra, y = imu.to(device), th.to(device), ultra.to(device), y.to(device)
            pred = model(imu, th, ultra).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total



# =====================================================================
# 8. Main Training Script
# =====================================================================

def main():
    data_root = "data_root"
    train_loader, val_loader, test_loader, dataset = make_loaders(data_root)

    num_classes = len(dataset.classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GestureModel(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nStarting training...\n")

    best_val = 0
    best_epoch = 0

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}")

        # Track and save best model
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "models/LSTM.pt")

    print(f"\nBest validation accuracy: {best_val:.3f} (Epoch {best_epoch})")

    # Load best model before testing
    model.load_state_dict(torch.load("models/LSTM.pt"))

    # Final test accuracy
    test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

    torch.save(model.state_dict(), "gesture_model_final.pt")
    print("Saved final model as gesture_model_final.pt")



if __name__ == "__main__":
    main()
