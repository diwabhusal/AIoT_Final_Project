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
# 1. IMU Encoder (3-axis accelerometer)
# =====================================================================

class IMUEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden=32, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, out_dim, 3, padding=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)



# =====================================================================
# 2. Thermal Encoder (8×8 frames)
# =====================================================================

class ThermalEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )

        self.temporal = nn.Conv1d(128, out_dim, 1)

    def forward(self, x):
        B, T, _, H, W = x.shape
        x = x.reshape(B * T, 1, H, W)
        x = self.cnn(x)
        x = x.mean(dim=[2, 3])
        x = x.reshape(B, T, 128)

        x = x.permute(0, 2, 1)
        x = self.temporal(x)
        return x.permute(0, 2, 1)



# =====================================================================
# 3. Ultrasound Encoder (front + side)
# =====================================================================

class UltrasoundEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden=32, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, out_dim, 3, padding=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)



# =====================================================================
# 4. Multimodal Transformer Gesture Model
# =====================================================================

class GestureTransformer(nn.Module):
    def __init__(self, num_classes, embed_dim=160, num_heads=4, num_layers=4):
        super().__init__()

        self.imu = IMUEncoder(out_dim=64)
        self.thermal = ThermalEncoder(out_dim=64)
        self.ultra = UltrasoundEncoder(out_dim=32)

        self.embed_dim = embed_dim
        self.proj = nn.Linear(160, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, imu, th, ultra):
        f1 = self.imu(imu)
        f2 = self.thermal(th)
        f3 = self.ultra(ultra)

        fused = torch.cat([f1, f2, f3], dim=-1)
        x = self.proj(fused)

        B = x.size(0)
        cls = self.cls_token.expand(B, 1, self.embed_dim)
        x = torch.cat([cls, x], dim=1)

        encoded = self.encoder(x)
        return self.fc(encoded[:, 0, :])



# =====================================================================
# 5. Dataset
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
                t = base + "_thermal.csv"
                uf = base + "_ultra_front.csv"
                us = base + "_ultra_side.csv"

                if os.path.exists(t) and os.path.exists(uf) and os.path.exists(us):
                    self.samples.append((imu_file, t, uf, us, self.class_to_idx[gesture]))

    def __len__(self):
        return len(self.samples)

    def _resample(self, ts, vals):
        grid = np.linspace(ts[0], ts[0] + self.window_sec, self.T, endpoint=False)

        if vals.ndim == 2:
            out = np.zeros((self.T, vals.shape[1]), np.float32)
            for i in range(vals.shape[1]):
                f = interp1d(ts, vals[:, i], fill_value="extrapolate")
                out[:, i] = f(grid)
            return out

        _, _, H, W = vals.shape
        out = np.zeros((self.T, 1, H, W), np.float32)
        for h in range(H):
            for w in range(W):
                f = interp1d(ts, vals[:, 0, h, w], fill_value="extrapolate")
                out[:, 0, h, w] = f(grid)
        return out

    def __getitem__(self, idx):
        imu_file, t_file, uf_file, us_file, label = self.samples[idx]

        imu = pd.read_csv(imu_file)
        imu50 = self._resample(
            imu["timestamp_ms"].to_numpy() / 1000,
            imu[["ax", "ay", "az"]].to_numpy().astype(np.float32)
        )

        t = pd.read_csv(t_file)
        th50 = self._resample(
            t["timestamp_ms"].to_numpy() / 1000,
            t[[f"t{i}" for i in range(64)]].to_numpy().astype(np.float32).reshape(-1, 1, 8, 8)
        )

        uf = pd.read_csv(uf_file)
        us = pd.read_csv(us_file)

        uf50 = self._resample(uf["timestamp_ms"].to_numpy() / 1000, uf[["front_cm"]].to_numpy())
        us50 = self._resample(us["timestamp_ms"].to_numpy() / 1000, us[["side_cm"]].to_numpy())

        ultra = np.concatenate([uf50, us50], axis=1)

        return (
            torch.tensor(imu50),
            torch.tensor(th50),
            torch.tensor(ultra),
            torch.tensor(label)
        )



# =====================================================================
# 6. Loaders / Training / Eval
# =====================================================================

def make_loaders(root, batch=8):
    dataset = GestureDataset(root)

    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val

    train, val, test = random_split(dataset, [n_train, n_val, n_test])

    return (
        DataLoader(train, batch_size=batch, shuffle=True),
        DataLoader(val, batch_size=batch),
        DataLoader(test, batch_size=batch),
        dataset
    )


def train_epoch(model, loader, opt, device):
    model.train()
    total = 0
    for imu, th, ultra, y in loader:
        imu, th, ultra, y = imu.to(device), th.to(device), ultra.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(imu, th, ultra), y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imu, th, ultra, y in loader:
            imu, th, ultra, y = imu.to(device), th.to(device), ultra.to(device), y.to(device)
            pred = model(imu, th, ultra).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total



# =====================================================================
# 7. Main
# =====================================================================

def main():
    train_loader, val_loader, test_loader, dataset = make_loaders("data_root", batch=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GestureTransformer(len(dataset.classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_accs, val_accs = [], []
    best_val = 0

    print("\nTraining Transformer model...\n")

    for ep in range(25):
        loss = train_epoch(model, train_loader, opt, device)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {ep+1:02d} | "
            f"Loss {loss:.4f} | "
            f"Train Acc {train_acc:.3f} | "
            f"Val Acc {val_acc:.3f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "models/transformer.pt")

    print("\nBest Val Acc:", best_val)

    # Plot accuracy curves
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy (Transformer)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("transformer_training_curve.png")
    plt.show()

    model.load_state_dict(torch.load("models/transformer.pt"))
    test_acc = evaluate(model, test_loader, device)
    print("\nTest Accuracy:", test_acc)



if __name__ == "__main__":
    main()
