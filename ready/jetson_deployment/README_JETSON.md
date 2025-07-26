# GLASS Fabric Detection - Jetson Orin Nano Deployment

## Overview
Complete deployment package for running GLASS fabric defect detection on NVIDIA Jetson Orin Nano 8GB with optimized performance for edge inference.

## Quick Transfer and Setup

### Step 1: Package the System (On PC)
```bash
# Create deployment package
./create_jetson_package.sh

# This creates: glass_jetson_deploy.tar.gz
```

### Step 2: Transfer to Jetson
```bash
# Copy package to Jetson
scp glass_jetson_deploy.tar.gz jetson@<jetson-ip>:~/

# Or use USB drive, SD card, etc.
```

### Step 3: Install on Jetson
```bash
# On Jetson Orin Nano
tar -xzf glass_jetson_deploy.tar.gz
cd glass_jetson_deploy
sudo ./install_jetson.sh
```

### Step 4: Run System
```bash
# Test installation
python verify_jetson.py

# Run fabric detection
python run_fabric_detection_jetson.py
```

## Jetson Optimizations

### Hardware Configuration
- **Model**: NVIDIA Jetson Orin Nano 8GB
- **Memory**: 8GB shared CPU/GPU (optimized usage < 6GB)
- **Power Mode**: 15W sustained, 25W peak
- **Storage**: 64GB+ eMMC/SD card recommended

### Performance Optimizations
1. **Model Quantization**: FP16 precision for 2x speedup
2. **Memory Management**: Optimized batch processing
3. **Power Management**: Configurable power profiles
4. **TensorRT Integration**: Coming soon (optional)
5. **Input Resolution**: Configurable (224x224 to 288x288)

### Expected Performance
| Mode | Resolution | Inference Time | FPS | Power | Memory |
|------|------------|----------------|-----|-------|--------|
| High Performance | 288x288 | ~50ms | 20 FPS | 20W | 4GB |
| Balanced | 256x256 | ~35ms | 28 FPS | 15W | 3GB |
| Power Save | 224x224 | ~25ms | 40 FPS | 12W | 2.5GB |

## Installation Requirements

### System Requirements
- JetPack 5.1+ (Ubuntu 20.04)
- Python 3.8+
- CUDA 11.4+ (included in JetPack)
- 4GB+ free storage
- Camera (USB/CSI)

### Pre-installed on Jetson
- NVIDIA Container Runtime
- CUDA Toolkit
- cuDNN
- TensorRT (optional)

## Deployment Package Contents

```
glass_jetson_deploy/
├── install_jetson.sh              # Main installation script
├── requirements_jetson.txt         # Jetson-specific dependencies
├── verify_jetson.py              # System verification
├── run_fabric_detection_jetson.py # Optimized main runner
├── jetson_config.yaml            # Jetson-specific configuration
├── models/                        # Optimized models
│   ├── glass_fp16.onnx           # FP16 optimized model
│   └── glass_int8.onnx           # INT8 quantized model (future)
├── utils/                         # Core utilities
├── scripts/                       # Inference scripts
└── docs/                         # Documentation
    ├── PERFORMANCE.md             # Performance tuning guide
    ├── TROUBLESHOOTING.md         # Common issues and solutions
    └── POWER_MANAGEMENT.md        # Power optimization guide
```

## Quick Commands

### System Management
```bash
# Check system status
sudo jtop                          # GPU/CPU/Memory monitoring
sudo tegrastats                    # Real-time stats

# Power management
sudo nvpmodel -m 0                 # Max performance (25W)
sudo nvpmodel -m 1                 # Balanced (15W)
sudo nvpmodel -m 2                 # Power save (10W)

# Set clock speeds
sudo jetson_clocks                 # Max clocks
```

### GLASS System
```bash
# Quick test
python verify_jetson.py

# Performance benchmark
python benchmark_jetson.py

# Run detection (auto-optimized)
python run_fabric_detection_jetson.py --mode balanced

# Run with specific settings
python run_fabric_detection_jetson.py --resolution 256 --power_mode 15W
```

## Network Deployment (Optional)

### Remote Access
```bash
# SSH tunnel for remote display
ssh -X jetson@<jetson-ip>

# VNC setup for GUI access
./setup_vnc.sh
```

### Web Interface (Future)
- HTTP API for remote control
- Web-based monitoring dashboard
- REST endpoints for integration

## Integration with Manufacturing

### GPIO Integration
```python
# Example: Connect detection results to GPIO
import RPi.GPIO as GPIO

def trigger_reject_mechanism(defect_detected):
    if defect_detected:
        GPIO.output(18, GPIO.HIGH)  # Trigger reject mechanism
```

### Industrial Protocols
- Modbus TCP support
- OPC-UA integration ready
- PLC communication protocols

## Monitoring and Maintenance

### Health Monitoring
```bash
# System health check
python health_check_jetson.py

# Performance monitoring
python monitor_performance.py
```

### Update Procedure
```bash
# Update models only
./update_models.sh

# Update full system
./update_system.sh
```

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch size or resolution
2. **Power Throttling**: Check power supply and thermal management
3. **Camera Issues**: Verify V4L2 permissions and drivers
4. **Performance**: Check nvpmodel and jetson_clocks settings

### Debug Commands
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check memory
free -h
nvidia-smi

# Check camera
v4l2-ctl --list-devices
```

## Production Deployment

### Service Setup
```bash
# Install as systemd service
sudo ./install_service.sh

# Control service
sudo systemctl start glass-detection
sudo systemctl enable glass-detection
sudo systemctl status glass-detection
```

### Logging
- System logs: `/var/log/glass-detection/`
- Performance logs: `/opt/glass/logs/`
- Error logs: Syslog integration

## Support and Updates

### Community
- GitHub Issues: Report bugs and feature requests
- Documentation: Online docs and tutorials
- Examples: Sample configurations and integrations

### Professional Support
- Industrial deployment assistance
- Custom optimization services
- 24/7 monitoring solutions

---

**Note**: This deployment package is optimized for Jetson Orin Nano 8GB. For other Jetson models, configurations may need adjustment.