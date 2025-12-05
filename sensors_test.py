#!/usr/bin/env python3

import time
import json

# for the i2c sensors
import board
import busio
from adafruit_amg88xx import AMG88XX
import adafruit_adxl34x

from gpiozero import DistanceSensor #for ultrasonic sensor

# 1. Initialize I2C Bus
i2c = busio.I2C(board.SCL, board.SDA)
thermal = AMG88XX(i2c)                  # Thermal Camera (AMG8833)
accel = adafruit_adxl34x.ADXL345(i2c)   # Accelerometer (ADXL345)

# 2. Initialize Ultrasonic Sensors
ultra1 = DistanceSensor(trigger=17, echo=27, max_distance=4)    # Sensor #1 → GPIO17 (TRIG), GPIO27 (ECHO)
ultra2 = DistanceSensor(trigger=23, echo=24, max_distance=4)    # Sensor #2 → GPIO23 (TRIG), GPIO24 (ECHO)

print("📡 Starting sensor fusion loop...")

try:
    while True:

        # ---- Thermal Camera (8x8 Grid) ----
        thermal_grid = thermal.pixels    # 8x8 array of temperatures

        # ---- Accelerometer ----
        accel_x, accel_y, accel_z = accel.acceleration

        # ---- Ultrasonic Distances ----
        dist1 = ultra1.distance * 100    # to cm
        dist2 = ultra2.distance * 100

        # ---- Package everything into one JSON object ----
        sensor_packet = {
            "thermal": thermal_grid,
            "accelerometer": {
                "x": round(accel_x, 3),
                "y": round(accel_y, 3),
                "z": round(accel_z, 3)
            },
            "ultrasonic": {
                "front": round(dist1, 2),
                "side": round(dist2, 2)
            },
            "timestamp": time.time()
        }

        # Print clean JSON
        print(json.dumps(sensor_packet))
        time.sleep(0.2)   # 5 Hz loop

except KeyboardInterrupt:
    print("Stopping sensors...")
