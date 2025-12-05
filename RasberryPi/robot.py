import serial
import time

# OPEN CONNECTION TO ARDUINO ON /dev/ttyUSB0
arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # allow Arduino to reset after opening port

def send(cmd):
    """Send a command to the Arduino robot arm."""
    arduino.write((cmd + "\n").encode())
    print("SENT:", cmd)
    time.sleep(0.1)

# ----------- TEST COMMANDS -----------

send("SAFE")
time.sleep(2)

send("S1:45")
time.sleep(1)

send("S1:135")
time.sleep(1)

send("S2:120")
time.sleep(1)

send("S3:60")
time.sleep(1)

send("OPEN")
time.sleep(1)

send("CLOSE")
