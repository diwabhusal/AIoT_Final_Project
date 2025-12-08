#!/usr/bin/env python3
"""
Live gesture inference pipeline for Raspberry Pi.

- Reads sensors via sensors_live.read_frame()
- Builds 1-second sequences at ~50 Hz (T=50)
- Feeds into a trained model (LSTM or Transformer)
- Maps predicted class → gesture string (same order as training)
- Optionally sends that gesture to the robot via robot_control.execute_gesture()

No RGB camera is used.
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
from scipy.interpolate import interp1d

import models                      # LSTM-based GestureModel
import transformer as tf_models    # Transformer-based GestureTransformer

import board
import busio
from adafruit_amg88xx import AMG88XX
import adafruit_adxl34x
from gpiozero import DistanceSensor
import RPi.GPIO as GPIO

# free any RPi.GPIO reservations in this process
try:
    GPIO.cleanup()
except Exception:
    pass

# Try to import robot control; allow running without robot (for testing).
try:
    from robot_control import execute_gesture
    ROBOT_AVAILABLE = True
except Exception as e:
    print(f"[live_inference] ⚠️ Robot control not available: {e}")
    print("                   You can still test inference with --no-robot.")
    ROBOT_AVAILABLE = False

# Sequence length / loop frequency must match training (1.0 sec @ 50 Hz).
SEQ_LEN = 50
FUSE_HZ = 50.0
DT = 1.0 / FUSE_HZ

def resample(ts, vals, window_sec=1.0, T=SEQ_LEN):
    """
    Resample (interpolate) onto a fixed time grid of length T starting at ts[0]
    and spanning `window_sec` secondimu_bufs.

    Implements the same behaviour as GestureDataset._resample used during
    training: per-channel interp1d with fill_value='extrapolate'.
    """
    ts = np.asarray(ts, dtype=float)
    t0 = ts[0]
    t_grid = np.linspace(t0, t0 + window_sec, T, endpoint=False)

    # Vector data (IMU or ultrasound)
    if vals.ndim == 2:
        out = np.zeros((T, vals.shape[1]), dtype=np.float32)
        for c in range(vals.shape[1]):
            f = interp1d(ts, vals[:, c], fill_value="extrapolate")
            out[:, c] = f(t_grid)
        return out

    # Thermal image data: [N, 1, H, W]
    N, _, H, W = vals.shape
    out = np.zeros((T, 1, H, W), dtype=np.float32)
    for h in range(H):
        for w in range(W):
            pix = vals[:, 0, h, w]
            f = interp1d(ts, pix, fill_value="extrapolate")
            out[:, 0, h, w] = f(t_grid)
    return out



# -------------------------------------------------------------------
# 1. Label order helper (must match training)
# -------------------------------------------------------------------

def get_gesture_labels(root: str = "data_root_5_classes"):
    """
    Recreate the same label order used during training:
        classes = sorted([c for c in os.listdir(root) if not c.startswith(".")])
    """
    all_items = os.listdir(root)
    labels = sorted([c for c in all_items if not c.startswith(".")])
    if not labels:
        raise RuntimeError(f"No gesture folders found in {root}")
    print(f"[live_inference] Gesture classes: {labels}")
    return labels


# -------------------------------------------------------------------
# 2. Model loader for LSTM vs Transformer
# -------------------------------------------------------------------

def load_model(model_type: str, device: torch.device, num_classes: int) -> torch.nn.Module:
    """
    Load a trained model state dict into the appropriate architecture.

    model_type: "lstm" or "transformer"
    """
    if model_type == "lstm":
        print("[live_inference] Using LSTM-based GestureModel (models.GestureModel)")
        ModelClass = models.GestureModel
        state_path = "gesture_model_5_classes.pt"

    elif model_type == "transformer":
        print("[live_inference] Using Transformer-based model (transformer.GestureTransformer)")
        ModelClass = tf_models.GestureTransformer
        state_path = "best_transformer_5_classes.pt"

    else:
        raise ValueError(f"Unknown model_type '{model_type}' (expected 'lstm' or 'transformer').")

    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Model weights file '{state_path}' not found in {os.getcwd()}")

    model = ModelClass(num_classes=num_classes).to(device)
    print(f"[live_inference] Loading weights from {state_path}...")
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[live_inference] Model loaded and in eval mode.")
    return model


# (preprocessing function removed) we use resampled sequences directly as model inputs


# -------------------------------------------------------------------
# 4. Main live loop
# -------------------------------------------------------------------

def run_live_inference(model_type: str, use_robot: bool):

    # -------------------------
    # Sensor initialization
    # -------------------------

    _i2c = busio.I2C(board.SCL, board.SDA)

    # ADXL345 accelerometer
    _accel = adafruit_adxl34x.ADXL345(_i2c)

    # AMG8833 thermal camera
    _thermal = AMG88XX(_i2c)

    # GPIO setup for ultrasonic sensors
    front_ultra = DistanceSensor(trigger=17, echo=27, max_distance=4)
    side_ultra  = DistanceSensor(trigger=23, echo=24, max_distance=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[live_inference] Using device: {device}")

    # Get labels in the *same order* as training
    labels = get_gesture_labels("data_root_5_classes")
    num_classes = len(labels)

    model = load_model(model_type, device, num_classes)

    imu_buf = deque(maxlen=SEQ_LEN)      # each: [3]
    th_buf  = deque(maxlen=SEQ_LEN)      # each: [1,8,8]
    ultra_buf = deque(maxlen=SEQ_LEN)    # each: [2]

    last_sent_time = 0.0

    print("[live_inference] Starting live inference loop...")
    print("  Press Ctrl+C to stop.")
    print("  Running at ~50 Hz with a 1-second (50 frame) window.\n")

    ACCEL_HZ = 200.0
    ACCEL_DT = 1.0 / ACCEL_HZ          # ~0.005 s

    ULTRA_DECIM = 5    # 200 / 5 = 40 Hz
    THERM_DECIM = 20   # 200 / 20 = 10 Hz

    accel_count = 0

    accel_buff = []
    thermal_buff = []
    ultra_front_buff = []
    ultra_side_buff = []


    try:
        while True:
            
            # Make the sample code similar to the sample collection code in collect_samples_2.py
            t0 = time.time()
            timestamp_ms = int(t0 * 1000)

            # ---- Accelerometer @ ~200 Hz ----
            ax, ay, az = _accel.acceleration
            accel_buff.append((timestamp_ms, ax, ay, az))

            # ---- Ultrasonic @ ~40 Hz ----
            if accel_count % ULTRA_DECIM == 0:
                front_cm = front_ultra.distance * 100.0
                side_cm = side_ultra.distance * 100.0
                ultra_front_buff.append((timestamp_ms, front_cm))
                ultra_side_buff.append((timestamp_ms, side_cm))

            # ---- Thermal @ ~10 Hz ----
            if accel_count % THERM_DECIM == 0:
                grid = _thermal.pixels  # 8x8
                flat = [temp for row in grid for temp in row]
                thermal_buff.append((timestamp_ms, flat))

            accel_count += 1

            # Every 200 accel samples (~1 second), build a frame
            if accel_count >= ACCEL_HZ:
                print("Building 1-second sample for inference...")
                accel_count = 0

                # Resample the buffers to 50 Hz in line with the model
                imu_ts = np.array([t for t,_,_,_ in accel_buff], dtype=np.float32) / 1000.0
                imu_vals = np.array([[ax, ay, az] for _,ax,ay,az in accel_buff], dtype=np.float32)

                imu50 = resample(imu_ts, imu_vals)
                print("imu shape", imu50.shape)

                th_times = np.array([t for t,_ in thermal_buff], dtype=np.float32) / 1000.0
                th_vals = np.array([[flat] for _,flat in thermal_buff], dtype=np.float32).reshape(-1,1,8,8)
                th50 = resample(th_times, th_vals)
                print("thermal 50 shape", th50.shape)
                # Convert to fp32
                th50 = th50.astype(np.float32)
                # reshape to [SEQ_LEN, 1, 8, 8]
                th50 = th50.reshape(SEQ_LEN, 1, 8, 8)

                uf_times = np.array([t for t,_ in ultra_front_buff], dtype=np.float32) / 1000.0
                uf_vals = np.array([[dist] for _,dist in ultra_front_buff], dtype=np.float32)
                uf50 = resample(uf_times, uf_vals)
                # Convert to fp32
                uf50 = uf50.astype(np.float32)
                print("uf50.shape", uf50.shape)

                us_times = np.array([t for t,_ in ultra_side_buff], dtype=np.float32) / 1000.0
                us_vals = np.array([[dist] for _,dist in ultra_side_buff], dtype=np.float32)
                print("resampling us")
                us50 = resample(us_times, us_vals)
                print("us50", us50.shape)
                # Convert to fp32
                us50 = us50.astype(np.float32)

                # clear raw buffers so the next second starts fresh
                accel_buff.clear()
                thermal_buff.clear()
                ultra_front_buff.clear()
                ultra_side_buff.clear()

                imu_buf = imu50
                th_buf = th50
                # Concat uf and us and keep that as ultra_buf
                ultra_buf = np.concatenate([uf50, us50], axis=-1)   # [SEQ_LEN, 2]
                print("ultra_buf", ultra_buf.shape)

            else:
                # Sleep to hit ~200 Hz base rate
                time.sleep(ACCEL_DT)
                t0 = time.time()
                continue

            imu_t = torch.from_numpy(imu_buf).unsqueeze(0).to(device)      # [1,T,3]
            th_t  = torch.from_numpy(th_buf).unsqueeze(0).to(device)       # [1,T,1,8,8]
            ultra_t = torch.from_numpy(ultra_buf).unsqueeze(0).to(device)  # [1,T,2]

            # Print all shapes
            print("imu_t", imu_t.shape)
            print("th_t", th_t.shape)
            print("ultra_t", ultra_t.shape)

            with torch.no_grad():
                logits = model(imu_t, th_t, ultra_t)   # [1, num_classes]
                probs = torch.softmax(logits, dim=-1)  # [1, num_classes]
                pred_idx = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[0, pred_idx].item())

            gesture = labels[pred_idx]

            # Print prediction for debugging
            ts_str = time.strftime("%H:%M:%S")
            print(f"[{ts_str}] Predicted: {gesture:12s}  (p={confidence:.3f})")

            # Debounce: only send to robot if gesture changes, is not "nothing",
            # and we have enough confidence.
            now = time.time()
            if (
                use_robot
                and ROBOT_AVAILABLE
                and gesture != "nothing"
            ):
                print(f"[live_inference] → Executing gesture '{gesture}' on robot.")
                execute_gesture(gesture)
            print("[live_inference] Resuming inference on ENTER key...")
            input()
                

    except KeyboardInterrupt:
        print("\n[live_inference] Stopped by user (Ctrl+C).")


# -------------------------------------------------------------------
# 5. CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live gesture inference pipeline (LSTM / Transformer)."
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer"],
        default="lstm",
        help="Which trained model to use for inference."
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Run inference but do NOT send commands to the robot (debug mode)."
    )

    args = parser.parse_args()
    run_live_inference(model_type=args.model, use_robot=not args.no_robot)
