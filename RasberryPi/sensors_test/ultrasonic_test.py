from gpiozero import DistanceSensor
from time import sleep

sensor1 = DistanceSensor(echo=27, trigger=17, max_distance=4)
sensor2 = DistanceSensor(echo=24, trigger=23, max_distance=4)

while True:
    d1 = sensor1.distance * 100
    d2 = sensor2.distance * 100

    print(f"Sensor 1: {d1:.1f} cm   |   Sensor 2: {d2:.1f} cm")
    sleep(0.2)
