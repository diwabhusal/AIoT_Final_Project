import time
import board
import busio
import adafruit_adxl34x

i2c = busio.I2C(board.SCL, board.SDA)
accel = adafruit_adxl34x.ADXL345(i2c)

print("ADXL345 Accelerometer Test")

while True:
    x, y, z = accel.acceleration
    print("X: {:.2f}  Y: {:.2f}  Z: {:.2f}".format(x, y, z))
    time.sleep(0.2)
