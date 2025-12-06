#!/usr/bin/env python3
import time
import csv
import os
import sys
import gc
import argparse

import board
import busio
from adafruit_amg88xx import AMG88XX
import adafruit_adxl34x
from gpiozero import DistanceSensor
import RPi.GPIO as GPIO

"""
Multi-rate gesture recording script with fixed gesture set.

Sampling rates:
- Accelerometer: ~200 Hz
- Ultrasonic (front, side): ~40 Hz
- Thermal (AMG8833): ~10 Hz

Gestures (classes):
 1) swipe_right   (rotate right)
 2) swipe_left    (rotate left)
 3) move_up
 4) move_down
 5) elbow_front
 6) elbow_back
 7) claw_open
 8) claw_close
 9) nothing       (no gesture / idle)

Usage:

    # Interactive gesture menu, variable length (Ctrl+C to stop)
    python3 record_gesture.py

    # Interactive gesture menu, fixed 1.0 s duration
    python3 record_gesture.py --duration 1.0

    # Direct gesture, no menu (useful for scripts)
    python3 record_gesture.py --gesture swipe_right --duration 1.5
"""

# Fixed gesture definitions
GESTURES = [
    ("swipe_right", "Swipe right (rotate right)"),
    ("swipe_left",  "Swipe left (rotate left)"),
    ("move_up",     "Move up"),
    ("move_down",   "Move down"),
    ("elbow_front", "Elbow front"),
    ("elbow_back",  "Elbow back"),
    ("claw_open",   "Claw open"),
    ("claw_close",  "Claw close"),
    ("nothing",     "Nothing / no gesture"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record multi-rate sensor data for a single labeled gesture."
    )
    parser.add_argument(
        "--gesture",
        choices=[g[0] for g in GESTURES],
        help="Optional gesture label to record (skips menu)."
    )
    parser.add_argument(
        "--num_cycles",
        type=float,
        default=None,
        help="Optional number of accelerometer cycles to collect. If omitted, press Ctrl+C to stop."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of gesture samples to record (default: 1)."
    )
    return parser.parse_args()


def choose_gesture_from_menu():
    print("📚 Gesture classes:")
    for idx, (key, desc) in enumerate(GESTURES, start=1):
        print(f"  {idx}) {key:<12} - {desc}")
    print()

    while True:
        choice = input("👉 Select gesture number [1-9]: ").strip()
        if not choice.isdigit():
            print("  Invalid input, please enter a number.")
            continue
        n = int(choice)
        if 1 <= n <= len(GESTURES):
            key, desc = GESTURES[n - 1]
            print(f"\n✅ Selected gesture: {key}  ({desc})\n")
            return key, desc
        else:
            print("  Number out of range, try again.")


def determine_next_rec_prefix(root, gesture_name):
    gesture_dir = os.path.join(root, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    existing = [f for f in os.listdir(gesture_dir) if f.endswith("_imu.csv")]
    if existing:
        nums = [int(f.split("_")[0][3:]) for f in existing]  # recX_imu.csv → X
        next_rec = max(nums) + 1
    else:
        next_rec = 1

    rec_prefix = f"rec{next_rec}"
    return gesture_dir, rec_prefix



def ensure_saved(path):
    """Block until file exists and is non-empty."""
    while not os.path.exists(path):
        time.sleep(0.01)
    while os.path.getsize(path) == 0:
        time.sleep(0.01)


def main():
    args = parse_args()

    num_samples = args.num_samples

    n = 0

    while n < num_samples:
        n += 1
        print(f"\n=== Recording sample {n} of {num_samples} ===\n")

        # Decide gesture label
        if args.gesture is not None:
            # Map key back to human description for logging
            desc = next((d for k, d in GESTURES if k == args.gesture), args.gesture)
            gesture_key = args.gesture
            gesture_desc = desc
        else:
            gesture_key, gesture_desc = choose_gesture_from_menu()

        accelerometer_cycles = args.num_cycles

        root = "data_root"
        gesture_dir, rec_prefix = determine_next_rec_prefix(root, gesture_key)

        print(f"🎬 Gesture key : {gesture_key}")
        print(f"📝 Description : {gesture_desc}")
        print(f"📝 Session     : {rec_prefix}")
        print(f"📁 Folder      : {gesture_dir}")

        # -------------------------
        # Cleanup before starting
        # -------------------------
        GPIO.cleanup()
        gc.collect()
        time.sleep(0.2)

        # -------------------------
        # Init sensors
        # -------------------------
        print("\n🔧 Initializing sensors...")
        i2c = busio.I2C(board.SCL, board.SDA)
        thermal = AMG88XX(i2c)                      # AMG8833 thermal camera
        accel = adafruit_adxl34x.ADXL345(i2c)       # ADXL345 accelerometer

        # Ultrasonic sensors (pins consistent with your other scripts)
        front_ultra = DistanceSensor(trigger=17, echo=27, max_distance=4)
        side_ultra  = DistanceSensor(trigger=23, echo=24, max_distance=4)

        print("✅ Sensors ready.")

        # -------------------------
        # Open CSV files
        # -------------------------
        imu_path     = os.path.join(gesture_dir, f"{rec_prefix}_imu.csv")
        thermal_path = os.path.join(gesture_dir, f"{rec_prefix}_thermal.csv")
        front_path   = os.path.join(gesture_dir, f"{rec_prefix}_ultra_front.csv")
        side_path    = os.path.join(gesture_dir, f"{rec_prefix}_ultra_side.csv")

        imu_file     = open(imu_path, "w", newline="")
        thermal_file = open(thermal_path, "w", newline="")
        front_file   = open(front_path, "w", newline="")
        side_file    = open(side_path, "w", newline="")

        imu_writer     = csv.writer(imu_file)
        thermal_writer = csv.writer(thermal_file)
        front_writer   = csv.writer(front_file)
        side_writer    = csv.writer(side_file)

        # Headers
        imu_writer.writerow(["timestamp_ms", "ax", "ay", "az"])
        thermal_writer.writerow(["timestamp_ms"] + [f"t{i}" for i in range(64)])
        front_writer.writerow(["timestamp_ms", "front_cm"])
        side_writer.writerow(["timestamp_ms", "side_cm"])

        # -------------------------
        # Sampling rates
        # -------------------------
        ACCEL_HZ = 200.0
        ACCEL_DT = 1.0 / ACCEL_HZ          # ~0.005 s

        ULTRA_DECIM = 5    # 200 / 5 = 40 Hz
        THERM_DECIM = 20   # 200 / 20 = 10 Hz

        print()
        print("📡 Sampling rates:")
        print(f"  Accelerometer : ~{ACCEL_HZ:.0f} Hz")
        print(f"  Ultrasonic    : ~{ACCEL_HZ/ULTRA_DECIM:.0f} Hz")
        print(f"  Thermal       : ~{ACCEL_HZ/THERM_DECIM:.0f} Hz")
        print()

        print("ℹ️  Instructions:")
        print(f"  Gesture: {gesture_key}  ({gesture_desc})")
        print("  1) Get into starting position.")
        if accelerometer_cycles is None:
            print("  2) When ready, press ENTER in this terminal.")
            print("  3) Perform the gesture (any length).")
            print("  4) Press Ctrl+C to stop recording right after the gesture ends.")
        else:
            print(f"  2) When ready, press ENTER. Recording will run for {accelerometer_cycles * ACCEL_DT:.2f} seconds.")
            print("  3) Perform the gesture during that window.")

        input("\n👉 Press ENTER when ready to start recording...")

        print("✅ Recording... Perform the gesture NOW!")

        start_time = time.time()
        accel_count = 0

        acc_readings = []
        thermal_readings = []
        front_readings = []
        side_readings = []

        try:
            while True:
                now = time.time()
                if accelerometer_cycles is not None and accel_count >= accelerometer_cycles:
                    print("\n⏱️  Num samples reached, stopping recording.")
                    break

                timestamp_ms = int(now * 1000)

                # ---- Accelerometer @ ~200 Hz ----
                ax, ay, az = accel.acceleration
                acc_readings.append([timestamp_ms, ax, ay, az])

                # ---- Ultrasonic @ ~40 Hz ----
                if accel_count % ULTRA_DECIM == 0:
                    front_cm = front_ultra.distance * 100.0
                    side_cm = side_ultra.distance * 100.0

                    front_readings.append([timestamp_ms, front_cm])
                    side_readings.append([timestamp_ms, side_cm])

                # ---- Thermal @ ~10 Hz ----
                if accel_count % THERM_DECIM == 0:
                    grid = thermal.pixels  # 8x8
                    flat = [temp for row in grid for temp in row]
                    thermal_readings.append([timestamp_ms] + flat)

                accel_count += 1

                # Sleep to hit ~200 Hz base rate
                time.sleep(ACCEL_DT)

        except KeyboardInterrupt:
            print("\n🛑 Recording stopped by user (Ctrl+C).")

        finally:
            # ---------------------
            # Flush & close files
            # ----------------------
            # Summary
            elapsed = time.time() - start_time
            if elapsed > 0:
                approx_accel_hz = accel_count / elapsed
                print(f"\n📊 Stats:")
                print(f"  Duration         : {elapsed:.3f} s")
                print(f"  Accel samples    : {accel_count} (~{approx_accel_hz:.1f} Hz)")
                print(f"  Ultra intervals  : every {ULTRA_DECIM} accel samples (~{approx_accel_hz/ULTRA_DECIM:.1f} Hz)")
                print(f"  Thermal intervals: every {THERM_DECIM} accel samples (~{approx_accel_hz/THERM_DECIM:.1f} Hz)")

            imu_writer.writerows(acc_readings)
            thermal_writer.writerows(thermal_readings)
            front_writer.writerows(front_readings)
            side_writer.writerows(side_readings)

            print("💾 Flushing and closing files...")

            # Press enter to save data
            key = input("👉 Press ENTER to save data or any other key to discard...")
            if key.strip() != "":
                # Discard this sample and continue to next
                print("❌ Discarding recorded data for this sample.")
                for f in [imu_file, thermal_file, front_file, side_file]:
                    f.close()
                for p in [imu_path, thermal_path, front_path, side_path]:
                    os.remove(p)
                # cleanup ultrasonic sensors
                try:
                    front_ultra.close()
                    side_ultra.close()
                except Exception:
                    pass
                # continue outer loop
                continue

            for f in [imu_file, thermal_file, front_file, side_file]:
                try:
                    f.flush()
                    os.fsync(f.fileno())
                    f.close()
                except Exception:
                    pass

            # Confirm each file is saved
            for p in [imu_path, thermal_path, front_path, side_path]:
                ensure_saved(p)

            # Cleanup ultrasonic sensors
            try:
                front_ultra.close()
                side_ultra.close()
            except Exception:
                pass

            print("✅ All files saved:")
            print(" ", imu_path)
            print(" ", thermal_path)
            print(" ", front_path)
            print(" ", side_path)

if __name__ == "__main__":
    main()