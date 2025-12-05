#!/usr/bin/env python3
import time
import json
from flask import Flask, request, jsonify

# -----------------------------
# Sensor Libraries
# -----------------------------
import board
import busio
from adafruit_amg88xx import AMG88XX
import adafruit_adxl34x
from gpiozero import DistanceSensor

# -----------------------------
# Initialize Flask
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Initialize I2C & Sensors
# -----------------------------
print("🔧 Initializing sensors...")

i2c = busio.I2C(board.SCL, board.SDA)

thermal = AMG88XX(i2c)                    # Thermal camera
accel = adafruit_adxl34x.ADXL345(i2c)     # Accelerometer

# Ultrasonic sensors using gpiozero (working configuration)
ultra_front = DistanceSensor(trigger=17, echo=27, max_distance=4)
ultra_side  = DistanceSensor(trigger=23, echo=24, max_distance=4)

print("✅ Sensors initialized successfully!")


# ------------------------------------------------
# Helper: Read all sensors and package into JSON
# ------------------------------------------------
def read_all_sensors():
    # --- Thermal camera ---
    thermal_grid = thermal.pixels  # 8x8 temperature grid

    # --- Accelerometer ---
    ax, ay, az = accel.acceleration

    # --- Ultrasonic ---
    try:
        front_cm = ultra_front.distance * 100
    except:
        front_cm = None

    try:
        side_cm = ultra_side.distance * 100
    except:
        side_cm = None

    packet = {
        "thermal": thermal_grid,
        "accelerometer": {
            "x": round(ax, 3),
            "y": round(ay, 3),
            "z": round(az, 3)
        },
        "ultrasonic": {
            "front": None if front_cm is None else round(front_cm, 2),
            "side":  None if side_cm is None else round(side_cm, 2)
        },
        "timestamp": time.time()
    }

    return packet


# ------------------------------------------------
# ROUTES
# ------------------------------------------------

@app.route("/")
def home():
    """Serve the web UI."""
    return open("index.html").read()

@app.route("/sensor_data", methods=["GET"])
def sensor_data():
    """Return JSON sensor readings."""
    data = read_all_sensors()
    return jsonify(data)

@app.route("/command", methods=["POST"])
def command():
    """Receive JSON command from UI."""
    incoming = request.get_json()

    print("📥 Received JSON command:", incoming)

    # For now, we simply acknowledge the command.
    # Later you can connect robot-arm motion functions here.
    response = {
        "received": True,
        "command": incoming,
        "message": "Raspberry Pi processed the command."
    }

    return jsonify(response)


# ------------------------------------------------
# RUN SERVER
# ------------------------------------------------
if __name__ == "__main__":
    print("🌐 Starting Flask server at http://0.0.0.0:5000 ...")
    app.run(host="0.0.0.0", port=5000)
