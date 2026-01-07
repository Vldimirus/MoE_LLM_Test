# HARDWARE_GUIDE.md
# Руководство по подключению Hardware

---

## Содержание

1. [Обзор](#обзор)
2. [Список компонентов](#список-компонентов)
3. [Схема подключения](#схема-подключения)
4. [Сервоприводы](#сервоприводы)
5. [Камеры](#камеры)
6. [Микрофоны и динамики](#микрофоны-и-динамики)
7. [Сенсоры](#сенсоры)
8. [Контроллеры](#контроллеры)
9. [Питание](#питание)
10. [Калибровка](#калибровка)
11. [Тестирование](#тестирование)

---

## Обзор

Это руководство описывает подключение hardware компонентов к системе Domain-Specific MoE для создания embodied AI робота.

### Уровни сложности

1. **Desktop Arm** (Beginner) - Простая 6-DOF рука
2. **Humanoid Torso** (Intermediate) - Верхняя часть тела
3. **Full Humanoid** (Advanced) - Полный гуманоид

---

## Список компонентов

### Desktop Arm (6-DOF) - Начальная конфигурация

#### Сервоприводы

| Qty | Компонент | Спецификация | Цена (USD) |
|-----|-----------|--------------|------------|
| 6 | Dynamixel MX-28 | 6-18V, 2.5Nm, 360° | $240 (6x$40) |
| 1 | Gripper Servo | MG996R, 11kg-cm | $10 |

**Альтернативы (бюджет):**
- 6x MG996R servo + PCA9685 PWM controller ($60)

#### Камеры

| Qty | Компонент | Спецификация | Цена |
|-----|-----------|--------------|------|
| 1 | USB Camera | Logitech C920, 1080p, 30fps | $70 |
| 1 | Wrist Camera | 640x480, wide angle | $20 |

#### Микрофон и динамик

| Qty | Компонент | Спецификация | Цена |
|-----|-----------|--------------|------|
| 1 | USB Microphone | Blue Yeti Nano | $100 |
| 1 | Speaker | 5W, USB powered | $20 |

#### Сенсоры

| Qty | Компонент | Назначение | Цена |
|-----|-----------|------------|------|
| 1 | Force Sensor FSR402 | Grip force | $15 |
| 6 | Hall Effect Encoders | Joint position | $30 |
| 1 | IMU MPU6050 | Orientation | $5 |

#### Контроллер

| Компонент | Спецификация | Цена |
|-----------|--------------|------|
| NVIDIA Jetson Nano 4GB | ARM CPU, 128-core GPU | $150 |
| USB Hub 3.0 | 7-port powered | $30 |
| MicroSD 128GB | Class 10, A1 | $20 |

#### Питание

| Компонент | Спецификация | Цена |
|-----------|--------------|------|
| Power Supply 12V 10A | 120W | $30 |
| Voltage Regulator | 12V → 5V, 5A | $15 |
| Power Distribution Board | - | $10 |

**Total Desktop Arm Cost: ~$800**

---

### Full Humanoid - Продвинутая конфигурация

#### Дополнительные компоненты

| Qty | Компонент | Назначение | Цена |
|-----|-----------|------------|------|
| 24 | Dynamixel XM430-W350 | Body servos | $2400 |
| 4 | USB Cameras | Stereo vision | $280 |
| 2 | USB Microphone Array | Spatial audio | $300 |
| 8 | Force Sensors | Feet, hands | $120 |
| 20 | Touch Sensors | Skin | $100 |
| 1 | LIDAR | Navigation | $300 |
| 1 | Jetson Xavier NX | Main computer | $400 |
| 1 | Battery 24V 20Ah | Power | $300 |

**Total Humanoid Cost: ~$5000**

---

## Схема подключения

### Desktop Arm Wiring Diagram

```
┌─────────────────────────────────────────────────────┐
│              NVIDIA Jetson Nano                     │
│                                                     │
│  GPIO  │  USB 3.0  │  I2C  │  Serial  │  Power    │
└───┬──────────┬────────┬───────┬─────────┬──────────┘
    │          │        │       │         │
    │          │        │       │         │
    │     ┌────┴──┐  ┌──┴─┐  ┌──┴──┐   ┌─┴─────┐
    │     │ USB   │  │IMU │  │Dyna │   │ 12V   │
    │     │ Hub   │  │    │  │mixel│   │ PSU   │
    │     └───┬───┘  └────┘  │USB  │   └───────┘
    │         │              │Adapt│
    │    ┌────┴────┐         └──┬──┘
    │    │         │            │
    ├────┤ Camera  │       ┌────┴────┐
    │    │ Wrist   │       │ Servo   │
    │    │         │       │ Daisy   │
    │    └─────────┘       │ Chain   │
    │                      │         │
    ├──────────────────────┤ MX-28 #1│
    │ Force Sensor         │ MX-28 #2│
    │ (Analog)             │ MX-28 #3│
    │                      │ ...     │
    └──────────────────────└─────────┘
```

### Питание

```
[12V PSU] ──┬─→ [Jetson Nano] (5V/4A via barrel jack)
            │
            ├─→ [USB Hub] (5V/2A via regulator)
            │
            ├─→ [Dynamixel Bus] (12V direct)
            │
            └─→ [Sensors] (3.3V/5V via regulators)
```

---

## Сервоприводы

### Dynamixel Setup

#### 1. Подключение

**Daisy Chain topology:**

```
[U2D2] ──┬─→ [Servo #1] ──→ [Servo #2] ──→ ... ──→ [Servo #N]
         │
         └─→ [12V Power]
```

**Компоненты:**
- U2D2 USB adapter ($40)
- 3-pin cables (included)
- Power injection every 5-6 servos

#### 2. Настройка ID

```python
import dynamixel_sdk as dxl

# Connect
port_handler = dxl.PortHandler('/dev/ttyUSB0')
packet_handler = dxl.PacketHandler(2.0)
port_handler.openPort()
port_handler.setBaudRate(1000000)

# Scan for servos
for servo_id in range(1, 253):
    model_number, result, error = packet_handler.ping(port_handler, servo_id)
    if result == dxl.COMM_SUCCESS:
        print(f"Servo found at ID {servo_id}, Model: {model_number}")

# Change ID
old_id = 1
new_id = 5

packet_handler.write1ByteTxRx(port_handler, old_id, 7, new_id)  # Address 7 = ID
```

#### 3. Конфигурация параметров

```python
# Operating Mode (Address 11)
# 0 = Current Control
# 1 = Velocity Control
# 3 = Position Control (default)
# 4 = Extended Position Control

packet_handler.write1ByteTxRx(port_handler, servo_id, 11, 3)

# Velocity Limit (Address 44)
max_velocity = 200  # 0-1023 (0.229 rpm per unit)
packet_handler.write4ByteTxRx(port_handler, servo_id, 44, max_velocity)

# Torque Enable (Address 64)
packet_handler.write1ByteTxRx(port_handler, servo_id, 64, 1)
```

### PWM Servos (Budget Option)

#### Подключение PCA9685

```python
from adafruit_servokit import ServoKit
import board
import busio

# I2C setup
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize PCA9685
kit = ServoKit(channels=16, i2c=i2c)

# Configure servo
kit.servo[0].set_pulse_width_range(500, 2500)  # Calibrate range

# Move servo
kit.servo[0].angle = 90  # 0-180 degrees
```

**Wiring:**

```
PCA9685 Board:
  VCC → 5V
  GND → GND
  SDA → GPIO 2 (SDA)
  SCL → GPIO 3 (SCL)
  V+  → 6V (servo power)

Servos:
  Channel 0-15 → Servo signal
  All GND → Common ground
  All VCC → 6V rail
```

---

## Камеры

### USB Camera Setup

#### 1. Проверка устройств

```bash
# List video devices
ls /dev/video*

# Check camera info
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --all
```

#### 2. OpenCV capture

```python
import cv2

# Open camera
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Capture frame
ret, frame = cap.read()

if ret:
    cv2.imshow('Camera', frame)
    cv2.waitKey(0)

cap.release()
```

#### 3. Калибровка камеры

```python
import numpy as np
import cv2

# Checkerboard pattern (9x6)
pattern_size = (9, 6)
square_size = 25  # mm

# Capture images of checkerboard
images = []  # List of checkerboard images

# Find corners
objpoints = []  # 3D points
imgpoints = []  # 2D points

objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Save calibration
np.savez('camera_calibration.npz', 
         camera_matrix=mtx, 
         distortion=dist)
```

### CSI Camera (Raspberry Pi)

```python
from picamera2 import Picamera2

# Initialize
picam = Picamera2()
config = picam.create_still_configuration(main={"size": (1920, 1080)})
picam.configure(config)

# Start
picam.start()

# Capture
image = picam.capture_array()

# Stop
picam.stop()
```

---

## Микрофоны и динамики

### USB Microphone

#### 1. Настройка ALSA

```bash
# List audio devices
arecord -l

# Test recording
arecord -D hw:1,0 -f cd test.wav -d 5

# Play back
aplay test.wav
```

#### 2. PyAudio capture

```python
import pyaudio
import wave

# Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Initialize
p = pyaudio.PyAudio()

# Open stream
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# Record
frames = []
for i in range(0, int(RATE / CHUNK * 5)):  # 5 seconds
    data = stream.read(CHUNK)
    frames.append(data)

# Save
wf = wave.open('recording.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
```

### Text-to-Speech Output

```python
from gtts import gTTS
import pygame

# Generate speech
tts = gTTS("Hello world", lang='en')
tts.save('output.mp3')

# Play
pygame.mixer.init()
pygame.mixer.music.load('output.mp3')
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
```

---

## Сенсоры

### Force Sensor (FSR)

#### Подключение

```
FSR → 10kΩ resistor → GND
FSR middle → ADC input (A0)
```

#### Чтение (MCP3008 ADC)

```python
import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

def read_adc(channel):
    if channel < 0 or channel > 7:
        return -1
    
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Read force sensor on channel 0
force_raw = read_adc(0)
force_voltage = (force_raw / 1023.0) * 3.3

# Calibrate to Newtons
force_newtons = calibrate_force(force_voltage)
```

### IMU (MPU6050)

#### I2C Setup

```python
from mpu6050 import mpu6050
import time

# Initialize
imu = mpu6050(0x68)  # I2C address

# Read accelerometer
accel = imu.get_accel_data()
print(f"Accel X: {accel['x']:.2f} m/s²")

# Read gyroscope
gyro = imu.get_gyro_data()
print(f"Gyro X: {gyro['x']:.2f} °/s")

# Read temperature
temp = imu.get_temp()
print(f"Temp: {temp:.1f} °C")
```

#### Калибровка IMU

```python
def calibrate_imu(imu, samples=1000):
    """Calibrate IMU by averaging static readings"""
    
    accel_offsets = {'x': 0, 'y': 0, 'z': 0}
    gyro_offsets = {'x': 0, 'y': 0, 'z': 0}
    
    for _ in range(samples):
        accel = imu.get_accel_data()
        gyro = imu.get_gyro_data()
        
        for axis in ['x', 'y', 'z']:
            accel_offsets[axis] += accel[axis]
            gyro_offsets[axis] += gyro[axis]
        
        time.sleep(0.001)
    
    # Average
    for axis in ['x', 'y', 'z']:
        accel_offsets[axis] /= samples
        gyro_offsets[axis] /= samples
    
    # Z-axis should be -9.8 m/s² (gravity)
    accel_offsets['z'] += 9.8
    
    return accel_offsets, gyro_offsets
```

### Touch Sensors

```python
import RPi.GPIO as GPIO

# Setup
GPIO.setmode(GPIO.BCM)
touch_pin = 17

GPIO.setup(touch_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Callback
def touch_detected(channel):
    print("Touch detected!")

GPIO.add_event_detect(touch_pin, GPIO.RISING, callback=touch_detected)

# Keep running
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    GPIO.cleanup()
```

---

## Контроллеры

### Jetson Nano Setup

#### 1. Flash JetPack

```bash
# Download JetPack image
# Flash to SD card using Etcher

# First boot
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
  python3-pip \
  python3-dev \
  build-essential
```

#### 2. Настройка GPU

```bash
# Check CUDA
nvcc --version

# Test PyTorch with GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Increase swap

```bash
# Create 8GB swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Raspberry Pi 4 Setup

#### Performance mode

```bash
# Add to /boot/config.txt
arm_freq=2000
over_voltage=6
gpu_mem=256

# Reboot
sudo reboot
```

---

## Питание

### Расчёт мощности

```
Desktop Arm (6-DOF):
- Jetson Nano: 10W
- 6x Servos: 6 × 10W = 60W (peak)
- Cameras: 2 × 2.5W = 5W
- Sensors: 2W
- Total: ~80W
- Recommended PSU: 12V 10A (120W)

Full Humanoid:
- Jetson Xavier: 20W
- 30x Servos: 30 × 15W = 450W (peak)
- Cameras: 4 × 2.5W = 10W
- Sensors: 5W
- Total: ~500W
- Battery: 24V 20Ah LiPo
```

### Power Distribution

```
[Main Battery 12V] ──┬─→ [Buck Converter 5V 5A] → Jetson
                     │
                     ├─→ [Servo Bus 12V] → All servos
                     │
                     └─→ [Buck Converter 3.3V 1A] → Sensors
```

### Battery Management

```python
import smbus
import time

# INA219 current sensor
bus = smbus.SMBus(1)
address = 0x40

def read_voltage():
    # Read bus voltage register
    data = bus.read_i2c_block_data(address, 0x02, 2)
    voltage = ((data[0] << 8) | data[1]) >> 3
    return voltage * 0.004  # Convert to volts

def read_current():
    # Read current register
    data = bus.read_i2c_block_data(address, 0x04, 2)
    current = (data[0] << 8) | data[1]
    if current > 32767:
        current -= 65536
    return current * 0.001  # Convert to amps

# Monitor power
while True:
    voltage = read_voltage()
    current = read_current()
    power = voltage * current
    
    print(f"Voltage: {voltage:.2f}V, Current: {current:.3f}A, Power: {power:.2f}W")
    
    if voltage < 10.5:  # Low battery warning
        print("WARNING: Low battery!")
    
    time.sleep(1)
```

---

## Калибровка

### Калибровка сервоприводов

```python
def calibrate_servo(servo_id):
    """
    Калибровка home position и limits
    """
    
    print(f"Calibrating servo {servo_id}")
    
    # Move to extremes
    print("Moving to minimum...")
    move_servo(servo_id, 0)
    input("Press Enter when at minimum position")
    min_position = read_servo_position(servo_id)
    
    print("Moving to maximum...")
    move_servo(servo_id, 4095)
    input("Press Enter when at maximum position")
    max_position = read_servo_position(servo_id)
    
    # Find center
    print("Moving to center...")
    center = (min_position + max_position) // 2
    move_servo(servo_id, center)
    input("Press Enter when at home position")
    home_position = read_servo_position(servo_id)
    
    return {
        'min': min_position,
        'max': max_position,
        'home': home_position
    }
```

### Калибровка end-effector

```python
def calibrate_end_effector():
    """
    Калибровка позиции end-effector
    """
    
    calibration_points = []
    
    for i in range(5):
        print(f"Move end-effector to calibration point {i+1}")
        input("Press Enter when ready")
        
        # Read joint angles
        joint_angles = read_all_joint_angles()
        
        # Measure actual position (e.g., with motion capture)
        print("Measure actual position and enter:")
        x = float(input("X (mm): "))
        y = float(input("Y (mm): "))
        z = float(input("Z (mm): "))
        
        calibration_points.append({
            'joint_angles': joint_angles,
            'position': [x, y, z]
        })
    
    # Compute forward kinematics correction
    correction = compute_fk_correction(calibration_points)
    
    return correction
```

---

## Тестирование

### Функциональные тесты

```bash
# Тест всех компонентов
python tests/hardware/test_all.py

# Outputs:
# ✓ Servos: OK (6/6)
# ✓ Cameras: OK (2/2)
# ✓ Microphone: OK
# ✓ Sensors: OK (3/3)
# ✓ IMU: OK
```

### Отдельные тесты

```python
# test_servos.py
def test_servo_movement():
    """Test each servo can move"""
    
    for servo_id in range(1, 7):
        print(f"Testing servo {servo_id}...")
        
        # Move to 0°
        move_servo(servo_id, 0)
        time.sleep(1)
        pos = read_servo_position(servo_id)
        assert abs(pos - 0) < 10, f"Servo {servo_id} failed"
        
        # Move to 90°
        move_servo(servo_id, 90)
        time.sleep(1)
        pos = read_servo_position(servo_id)
        assert abs(pos - 90) < 10, f"Servo {servo_id} failed"
        
        print(f"✓ Servo {servo_id} OK")

# test_cameras.py
def test_camera_capture():
    """Test camera can capture"""
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    assert ret, "Camera capture failed"
    assert frame.shape[0] > 0, "Empty frame"
    
    print("✓ Camera OK")
    cap.release()
```

### Safety Test

```python
def test_safety_limits():
    """Test safety system"""
    
    # Test velocity limit
    print("Testing velocity limit...")
    move_servo_fast(1, 180)  # Should trigger safety
    time.sleep(0.1)
    assert safety_system.emergency_stop == True
    
    # Test force limit
    print("Testing force limit...")
    apply_excessive_force()  # Should trigger safety
    time.sleep(0.1)
    assert safety_system.emergency_stop == True
    
    print("✓ Safety system OK")
```

---

## Troubleshooting

### Servo не двигается

**Проблема:** Servo не реагирует на команды

**Решения:**
1. Проверь питание (12V присутствует?)
2. Проверь ID servo (правильный?)
3. Проверь baudrate (1000000?)
4. Проверь torque enable (включен?)

```bash
# Scan for servos
python scripts/scan_servos.py

# Check power
multimeter → 12V between V+ and GND
```

### Камера не определяется

**Проблема:** `/dev/video0` не существует

**Решения:**
```bash
# Check USB
lsusb

# Check permissions
sudo usermod -a -G video $USER
sudo chmod 666 /dev/video0

# Reload udev
sudo udevadm control --reload-rules
```

### IMU дрейфует

**Проблема:** IMU показывает неточные значения

**Решение:** Повторная калибровка
```python
# Calibrate on flat surface
calibration = calibrate_imu(imu, samples=2000)
save_calibration(calibration, 'imu_cal.json')
```

---

## Дополнительные ресурсы

- [Dynamixel SDK](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [Jetson Nano Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
- [ROS2 Integration](https://docs.ros.org/en/humble/index.html)

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-06  
**Hardware Revision**: A
