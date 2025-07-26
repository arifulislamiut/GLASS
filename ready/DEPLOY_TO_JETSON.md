# 🚀 Deploy GLASS to Jetson Orin Nano - Quick Guide

## ✅ Package Ready!
Your Jetson deployment package is created: **`glass_jetson_deploy.tar.gz` (211MB)**

## 📱 Step-by-Step Deployment

### 1️⃣ **Transfer Package to Jetson**

**Option A: SCP (Network Transfer)**
```bash
# Replace <jetson-ip> with your Jetson's IP address
scp glass_jetson_deploy.tar.gz jetson@<jetson-ip>:~/
```

**Option B: USB/SD Card**
```bash
# Copy file to USB drive or SD card
# Then transfer to Jetson manually
```

**Option C: Direct Download**
```bash
# On Jetson, if you have internet access
# Upload to cloud storage and download on Jetson
```

### 2️⃣ **Install on Jetson** (Run on Jetson Orin Nano)

```bash
# Extract package
tar -xzf glass_jetson_deploy.tar.gz
cd glass_jetson_deploy

# Run installation (requires sudo)
sudo ./install_jetson.sh
```

### 3️⃣ **Verify Installation**

```bash
# Quick system check
python3 verify_jetson.py

# Performance benchmark
python3 benchmark_jetson.py --tests 10
```

### 4️⃣ **Run Fabric Detection**

```bash
# Optimized for Jetson
python3 run_fabric_detection_jetson.py

# Or use the global command (after installation)
glass-detect
```

## ⚡ Expected Performance on Jetson Orin Nano 8GB

| Mode | Resolution | Inference Time | FPS | Power | Memory |
|------|------------|----------------|-----|-------|--------|
| **Balanced** | 256x256 | ~35-50ms | 20-28 FPS | 15W | 3GB |
| **Max Performance** | 288x288 | ~50-70ms | 15-20 FPS | 20W | 4GB |
| **Power Save** | 224x224 | ~25-35ms | 28-40 FPS | 12W | 2.5GB |

## 🔧 Configuration Options

### Performance Modes
```bash
# Max performance (highest accuracy, more power)
python3 run_fabric_detection_jetson.py --mode max_performance

# Balanced (recommended)
python3 run_fabric_detection_jetson.py --mode balanced

# Power save (longest battery life)
python3 run_fabric_detection_jetson.py --mode power_save
```

### Custom Settings
```bash
# Custom resolution
python3 run_fabric_detection_jetson.py --resolution 224

# Custom power mode
sudo nvpmodel -m 1  # 15W mode
python3 run_fabric_detection_jetson.py
```

## 📊 System Commands

### Monitoring
```bash
# Real-time system stats
sudo jtop

# GPU/Memory monitoring
nvidia-smi

# Temperature monitoring
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'
```

### Performance Tuning
```bash
# Max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Balanced (15W)
sudo nvpmodel -m 1

# Power save (10W)
sudo nvpmodel -m 2
```

### Service Management (Optional)
```bash
# If installed as service
sudo systemctl start glass-detection
sudo systemctl status glass-detection
sudo systemctl stop glass-detection
```

## 🎯 Industrial Integration

### GPIO Control Example
```python
# Connect detection results to industrial equipment
import RPi.GPIO as GPIO

def setup_industrial_io():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)  # Reject mechanism
    GPIO.setup(19, GPIO.OUT)  # Ready indicator
    GPIO.setup(20, GPIO.OUT)  # Alarm

def handle_detection(defect_detected):
    if defect_detected:
        GPIO.output(18, GPIO.HIGH)  # Trigger reject
        GPIO.output(20, GPIO.HIGH)  # Sound alarm
    else:
        GPIO.output(19, GPIO.HIGH)  # Ready for next
```

### Network API (Future)
```bash
# Enable remote monitoring
python3 run_fabric_detection_jetson.py --api-enabled

# Access web interface
http://<jetson-ip>:8080
```

## 🔍 Troubleshooting

### Common Issues

**❌ "CUDA not available"**
```bash
# Check CUDA installation
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch for Jetson if needed
```

**❌ "Memory error"**
```bash
# Reduce resolution or enable swap
sudo fallocate -l 4G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**❌ "Camera not found"**
```bash
# Check camera
v4l2-ctl --list-devices
python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**❌ "Performance too slow"**
```bash
# Check power mode
sudo nvpmodel -q

# Set max performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Debug Commands
```bash
# System verification
python3 verify_jetson.py

# Performance benchmark
python3 benchmark_jetson.py --profile

# Resource monitoring
python3 -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"
```

## 📈 Performance Optimization Tips

1. **Power Management**: Use `nvpmodel -m 0` for max performance
2. **Memory**: Enable swap if running out of memory
3. **Cooling**: Ensure adequate cooling for sustained operation
4. **Camera**: Use MJPG format for better performance
5. **Resolution**: Balance between accuracy and speed
6. **Batch Processing**: Process every 2nd or 3rd frame for better FPS

## 🎉 You're Ready!

Your GLASS fabric defection system is now optimized and ready for Jetson Orin Nano deployment. 

### Quick Start Commands:
```bash
# Basic run
glass-detect

# Performance test
glass-benchmark

# System check
glass-verify
```

**Expected Result**: Real-time fabric defect detection at 20-40 FPS with industrial-grade accuracy on edge hardware! 🏭✨