import time 
import csv
import os
import sys
import board
import busio
from adafruit_amg88xx import AMG88XX
import adafruit_adxl34x
from gpiozero import DistanceSensor
import RPi.GPIO as GPIO
import gc

# -------------------------
# Check args
# -------------------------
if len(sys.argv) != 2:
    print("Usage: python3 record_gesture.py <gesture_name>")
    sys.exit(1)

gesture_name = sys.argv[1]
root = "data_root"
gesture_dir = os.path.join(root, gesture_name)

os.makedirs(gesture_dir, exist_ok=True)

# -------------------------
# Determine next recording index
# -------------------------
existing = [f for f in os.listdir(gesture_dir) if f.endswith("_imu.csv")]

if existing:
    nums = [int(f.split("_")[0][3:]) for f in existing]  # recX_imu.csv → X
    next_rec = max(nums) + 1
else:
    next_rec = 1

rec_prefix = f"rec{next_rec}"
print(f"🎬 Recording {gesture_name}, session {rec_prefix}")

# -------------------------
# Cleanup before starting a new recording
# -------------------------
GPIO.cleanup()
gc.collect()
time.sleep(0.2)

# -------------------------
# Init Sensors
# -------------------------
i2c = busio.I2C(board.SCL, board.SDA)
thermal = AMG88XX(i2c)
accel = adafruit_adxl34x.ADXL345(i2c)
front_ultra = DistanceSensor(trigger=17, echo=27, max_distance=4)
side_ultra  = DistanceSensor(trigger=23, echo=24, max_distance=4)

# -------------------------
# Open CSV Files
# -------------------------
imu_file     = open(os.path.join(gesture_dir, f"{rec_prefix}_imu.csv"), "w", newline="")
thermal_file = open(os.path.join(gesture_dir, f"{rec_prefix}_thermal.csv"), "w", newline="")
front_file   = open(os.path.join(gesture_dir, f"{rec_prefix}_ultra_front.csv"), "w", newline="")
side_file    = open(os.path.join(gesture_dir, f"{rec_prefix}_ultra_side.csv"), "w", newline="")

imu_writer = csv.writer(imu_file)
thermal_writer = csv.writer(thermal_file)
front_writer = csv.writer(front_file)
side_writer = csv.writer(side_file)

# -------------------------
# Write Headers (NO GYRO)
# -------------------------
imu_writer.writerow(["timestamp_ms","ax","ay","az"])
thermal_writer.writerow(["timestamp_ms"] + [f"t{i}" for i in range(64)])
front_writer.writerow(["timestamp_ms","front_cm"])
side_writer.writerow(["timestamp_ms","side_cm"])

print("⏳ Preparing… Starting in 2 seconds. Get ready!")
time.sleep(2)

print("📡 Recording 1 second at 10 Hz…")
start = int(time.time() * 1000)

# -------------------------
# Record exactly 1 second
# -------------------------
for i in range(10):  # 10 samples → 1 second
    timestamp = int(time.time() * 1000)

    # ----- Accelerometer Only -----
    ax, ay, az = accel.acceleration
    imu_writer.writerow([timestamp, ax, ay, az])

    # ----- Thermal -----
    grid = thermal.pixels
    flat = [temp for row in grid for temp in row]
    thermal_writer.writerow([timestamp] + flat)

    # ----- Ultrasound -----
    try:
        front = front_ultra.distance * 100
    except:
        front = None

    try:
        side = side_ultra.distance * 100
    except:
        side = None

    front_writer.writerow([timestamp, front])
    side_writer.writerow([timestamp, side])

    time.sleep(0.1)

print("✅ DONE! Saved:")
print(" ", os.path.join(gesture_dir, f"{rec_prefix}_imu.csv"))
print(" ", os.path.join(gesture_dir, f"{rec_prefix}_thermal.csv"))
print(" ", os.path.join(gesture_dir, f"{rec_prefix}_ultra_front.csv"))
print(" ", os.path.join(gesture_dir, f"{rec_prefix}_ultra_side.csv"))

def ensure_saved(path):
    """Block until file exists, is non-empty, and fully flushed."""
    # 1. Ensure file exists
    while not os.path.exists(path):
        time.sleep(0.01)

    # 2. Ensure file has some content
    while os.path.getsize(path) == 0:
        time.sleep(0.01)


# ----------------------
# CLOSE FILES + FLUSH
# ----------------------
for f in [imu_file, thermal_file, front_file, side_file]:
    f.flush()
    os.fsync(f.fileno())
    f.close()

# Confirm each file is really saved
ensure_saved(os.path.join(gesture_dir, f"{rec_prefix}_imu.csv"))
ensure_saved(os.path.join(gesture_dir, f"{rec_prefix}_thermal.csv"))
ensure_saved(os.path.join(gesture_dir, f"{rec_prefix}_ultra_front.csv"))
ensure_saved(os.path.join(gesture_dir, f"{rec_prefix}_ultra_side.csv"))

# Cleanup ultrasonic sensors *after* save confirmation
front_ultra.close()
side_ultra.close()

print("✅ all files saved!")
