import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return self.proj(x)



# =====================================================================
# 2. Thermal Encoder (AMG8833 8×8)
# =====================================================================

class ThermalEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )

        self.temporal = nn.Conv1d(64, out_dim, 3, padding=1)

    def forward(self, x):
        B, T, _, H, W = x.shape
        x = x.reshape(B * T, 1, H, W)
        x = self.cnn(x).mean(dim=[2, 3])
        x = x.reshape(B, T, 64)
        x = x.permute(0, 2, 1)
        x = self.temporal(x)
        return x.permute(0, 2, 1)



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
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return self.proj(x)



# =====================================================================
# 4. Multimodal Gesture Recognition Model
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

        self.fc = nn.Linear(256, num_classes)

    def forward(self, imu, th, ultra):
        f1 = self.imu(imu)
        f2 = self.thermal(th)
        f3 = self.ultra(ultra)

        fused = torch.cat([f1, f2, f3], dim=-1)
        out, _ = self.lstm(fused)
        last = out[:, -1, :]
        return self.fc(last)



# =====================================================================
# 5. Gesture Dataset
# =====================================================================

class GestureDataset(Dataset):
    def __init__(self, root, window_sec=1.0, fuse_hz=50):
        self.root = root
        self.window_sec = window_sec
        self.fuse_hz = fuse_hz
        self.T = int(window_sec * fuse_hz)

        self.classes = sorted([c for c in os.listdir(root) if not c.startswith(".")])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []

        for gesture in self.classes:
            gdir = os.path.join(root, gesture)
            imu_files = sorted(glob.glob(os.path.join(gdir, "*_imu.csv")))

            for imu_file in imu_files:
                base = imu_file[:-8]
                therm = base + "_thermal.csv"
                uf = base + "_ultra_front.csv"
                us = base + "_ultra_side.csv"

                if os.path.exists(therm) and os.path.exists(uf) and os.path.exists(us):
                    self.samples.append((imu_file, therm, uf, us, self.class_to_idx[gesture]))

        if not self.samples:
            raise RuntimeError("No valid samples found")

    def __len__(self):
        return len(self.samples)

    def _resample(self, ts, vals):
        t0 = ts[0]
        t_grid = np.linspace(t0, t0 + self.window_sec, self.T, endpoint=False)

        if vals.ndim == 2:
            out = np.zeros((self.T, vals.shape[1]), dtype=np.float32)
            for c in range(vals.shape[1]):
                f = interp1d(ts, vals[:, c], fill_value="extrapolate")
                out[:, c] = f(t_grid)
            return out

        N, _, H, W = vals.shape
        out = np.zeros((self.T, 1, H, W), dtype=np.float32)
        for h in range(H):
            for w in range(W):
                f = interp1d(ts, vals[:, 0, h, w], fill_value="extrapolate")
                out[:, 0, h, w] = f(t_grid)
        return out

    def __getitem__(self, idx):
        imu_file, th_file, uf_file, us_file, label = self.samples[idx]

        imu = pd.read_csv(imu_file)
        imu50 = self._resample(
            imu["timestamp_ms"].to_numpy() / 1000.0,
            imu[["ax", "ay", "az"]].to_numpy().astype(np.float32)
        )

        th = pd.read_csv(th_file)
        th50 = self._resample(
            th["timestamp_ms"].to_numpy() / 1000.0,
            th[[f"t{i}" for i in range(64)]].to_numpy().astype(np.float32).reshape(-1, 1, 8, 8)
        )

        uf = pd.read_csv(uf_file)
        us = pd.read_csv(us_file)

        uf50 = self._resample(uf["timestamp_ms"].to_numpy() / 1000.0, uf[["front_cm"]].to_numpy())
        us50 = self._resample(us["timestamp_ms"].to_numpy() / 1000.0, us[["side_cm"]].to_numpy())

        ultra50 = np.concatenate([uf50, us50], axis=1)

        return (
            torch.tensor(imu50),
            torch.tensor(th50),
            torch.tensor(ultra50),
            torch.tensor(label)
        )



# =====================================================================
# 6. Data Loaders
# =====================================================================

def make_loaders(root, batch_size=8):
    dataset = GestureDataset(root)

    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        dataset
    )



# =====================================================================
# 7. Training / Evaluation
# =====================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0
    for imu, th, ultra, y in loader:
        imu, th, ultra, y = imu.to(device), th.to(device), ultra.to(device), y.to(device)

        optimizer.zero_grad()
        loss = F.cross_entropy(model(imu, th, ultra), y)
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
# 8. Main
# =====================================================================

def main():
    train_loader, val_loader, test_loader, dataset = make_loaders("data_root")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GestureModel(len(dataset.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_accs, val_accs = [], []
    best_val = 0

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "models/LSTM.pt")

    # Plot accuracy
    epochs = range(1, len(train_accs) + 1)
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve_lstm.png")
    plt.show()

    model.load_state_dict(torch.load("models/LSTM.pt"))
    test_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
