import time
import busio
import board
import adafruit_amg88xx

print("Initializing I2C...")
i2c = busio.I2C(board.SCL, board.SDA)

print("Initializing AMG8833 sensor...")
sensor = adafruit_amg88xx.AMG88XX(i2c)

print("Reading thermal data...\n")

while True:
    pixels = sensor.pixels  # 8x8 temperature array

    # Print the center temperature just for testing
    center_temp = pixels[3][3]
    print("Center Pixel Temperature:", center_temp, "°C")

    time.sleep(0.5)
