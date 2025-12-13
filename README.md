# GhostArm
**Multimodal Gesture Recognition for Real-Time Robotic Arm Control**

GhostArm is an end-to-end gesture recognition and robotic control system built entirely on low-power, edge-compute hardware. The system uses a multimodal sensor array, including an IMU, thermal camera, and ultrasonic distance sensors, to interpret human arm gestures in real time and translate them into robotic arm movements. All inference is performed locally on a Raspberry Pi.

---

## Project Overview

The goal of GhostArm is to demonstrate that expressive and reliable human–robot interaction can be achieved using inexpensive sensors, efficient deep learning models, and edge-native computation. The system captures complementary motion cues from multiple sensing modalities, fuses them temporally, and classifies gestures with low latency before executing corresponding actions on a robotic arm.

Key features:
- Multimodal sensor fusion (IMU, thermal, ultrasonic)
- Real-time machine learning inference
- Transformer-based temporal modeling
- Closed-loop robotic control

---

## Hardware Components

- **Raspberry Pi 5**  
  Central processing unit for sensor fusion, model inference, and system control.

- **Robot Arm (KEYESTUDIO 4DOF Arduino Robot Arm)**  
  Controlled via an Arduino microcontroller using servo motors. Commands are sent from the Raspberry Pi over a UART serial connection.

- **Accelerometer (ADXL345)**  
  Captures 3-axis acceleration to detect dynamic arm motion.

- **Thermal Camera (AMG8833 8×8 IR Imager)**  
  Captures heat-based spatial information to help distinguish gestures.

- **Ultrasonic Distance Sensors (HC-SR04, two units)**  
  Measures distance changes to detect spatial motion along multiple axes.

---


## Software Requirements

- Python 3.9+
- PyTorch
- NumPy, SciPy, Pandas
- Matplotlib (for training curves)
- Arduino IDE

Install Python dependencies:

```bash
pip install torch numpy scipy pandas matplotlib
```

---

## How to Reproduce the Project

### 1. Hardware Setup

1. Assemble the robot arm and connect all servo motors to the Arduino.
2. Flash the Arduino using:
   ```
   RobotMC/Robot_MC_Codes/MC_ESP.ino
   ```
3. Connect the Arduino to the Raspberry Pi via UART (USB).
4. Connect sensors to the Raspberry Pi:
   - IMU + Thermal Camera → I2C
   - Ultrasonic Sensors → GPIO (with voltage dividers)
5. Verify sensor connections using the test scripts in:
   ```
   RaspberryPi/Sensors_Test/
   ```
6. Train LSTM and Transformer Based Models:
   ```
   cd RaspberryPi/
   python3 LSTM.py
   python3 transformer.py
   ```
5. Run live inference:
   ```
   cd RaspberryPi/
   python3 live_inference.py
   ```

---

### 2. Data Collection

1. Run the data collection script:
   ```bash
   python collect_samples_2.py
   ```
2. Perform gestures while recording sensor data.
3. Data will be saved under:
   ```
   RaspberryPi/data_root/<gesture_name>/
   ```

---

### 3. Model Training

1. Train the LSTM or Transformer model:
   ```bash
   python train_lstm.py
   # or
   python train_transformer.py
   ```
2. Training and validation accuracy plots will be saved as PNG files.
3. Best model weights are saved under:
   ```
   RaspberryPi/models/
   ```

---

### 4. Real-Time Inference

1. Start the server and inference pipeline:
   ```bash
   python RaspberryPi/server/server.py
   ```
2. Perform gestures in front of the sensors.
3. The predicted gesture is sent to the Arduino, controlling the robot arm in real time.

---

## Results

The system successfully classifies gestures in real time using only on-device computation. Transformer-based models achieved higher peak accuracy, while LSTM-based models demonstrated more stable training behavior on limited datasets. Both approaches were suitable for low-latency embedded deployment.

---

## Limitations and Future Work

- Dataset size limits generalization to complex gestures
- Transformer models may overfit without additional regularization
- Future work includes online adaptation, additional gestures, and improved sensor calibration

---

## Authors

Sri Iyengar

Diwa Bushal

Aymen Norain

Saivignesh Venkatraman
