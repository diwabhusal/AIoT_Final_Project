#!/usr/bin/env python3
"""
Robot control hook for gesture-based control.

This wraps the Arduino serial interface and defines execute_gesture(gesture),
which you can customize to send the right commands for your robot.

It currently just prints the gesture and shows example commands.
"""

import time
import serial

# Adjust if your Arduino is on a different port:
ARDUINO_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

print(f"[robot_control] Connecting to Arduino on {ARDUINO_PORT} @ {BAUD_RATE}...")
arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # allow Arduino to reset
print("[robot_control] Connected.")


def send(cmd: str, delay: float = 0.05):
    """Send a command string to the Arduino (with newline)."""
    line = (cmd + "\n").encode()
    arduino.write(line)
    print("[ROBOT] SENT:", cmd)
    time.sleep(delay)


# Gesture → robot command mapping.
# TODO: Edit these to match the actual commands your Arduino code understands.
GESTURE_TO_COMMANDS = {
    "swipe_right": ["S1:60"],   # <- replace with your actual command(s)
    "swipe_left":  ["S1:120"],
    "move_up":     ["S2:120"],
    "move_down":   ["S2:60"],
    "elbow_front": ["S3:120"],
    "elbow_back":  ["S3:60"],
    "claw_open":   ["CLOSE"],           # you already use "OPEN"/"CLOSE" in robot.py
    "claw_close":  ["OPEN"],
    "nothing":     [],                 # no-op
}


def execute_gesture(gesture: str):
    """
    High-level hook: called by inference loop when a new gesture is predicted.

    You should adjust GESTURE_TO_COMMANDS above instead of changing this function.
    """
    cmds = GESTURE_TO_COMMANDS.get(gesture, [])
    if not cmds:
        print(f"[ROBOT] Gesture '{gesture}' → no action (nothing / unmapped).")
        return

    print(f"[ROBOT] Executing gesture '{gesture}' with {len(cmds)} command(s).")
    for c in cmds:
        send(c)
