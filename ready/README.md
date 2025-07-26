# Ready-to-Deploy Fabric Defect Detection System

## Overview
Complete GLASS (Gradient-based Localized Anomaly Synthesis Strategy) system for real-time fabric defect detection from camera feed.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run System
```bash
# Run with CUDA acceleration (recommended)
python run_fabric_detection.py --device cuda

# Run with CPU only
python run_fabric_detection.py --device cpu
```

### 3. Controls
- **'q'** - Quit system
- **'d'** - Cycle display modes (All Views/Original+Anomaly/Original+Overlay/Original Only)
- **'s'** - Save current frame
- **'+'**/**'-'** - Adjust detection threshold

### 4. Display Modes
1. **All Views**: 2x2 grid showing original, anomaly map, overlay, and empty panel
2. **Original + Anomaly**: Side-by-side original and anomaly heatmap
3. **Original + Overlay**: Side-by-side original and overlay visualization
4. **Original Only**: Clean view of just the camera feed

## System Components

### Scripts
- `run_fabric_detection.py` - Main system runner with improved display
- `test_system.py` - System verification and diagnostics
- `test_cuda.py` - CUDA functionality and compatibility test
- `benchmark_cuda.py` - Performance benchmarking for CUDA vs CPU
- `demo_display_modes.py` - Interactive demo of display modes
- `scripts/camera_inference.py` - Camera inference engine
- `scripts/predict_single_image.py` - Single image predictor

### Models
- `models/wfdd_grid_cloth/` - Trained fabric defect detection model
  - `glass.onnx` - ONNX format model
  - `glass_simplified.onnx` - Optimized ONNX model
  - `ckpt_best_571.pth` - PyTorch checkpoint

### Utilities
- `utils/glass.py` - Core GLASS algorithm
- `utils/model.py` - Model architectures
- `utils/backbones.py` - Backbone networks
- `utils/common.py` - Common utilities

## Expected Performance
- **Processing Time**: 
  - **CUDA**: ~13ms per frame (75+ FPS)
  - **CPU**: ~100-200ms per frame (5-10 FPS)
- **Detection Accuracy**: >90% for fabric defects  
- **Camera Support**: USB, CSI, IP cameras
- **Resolution**: Optimized for 640x480 input
- **GPU Memory**: ~2GB VRAM usage

## Fabric Defect Types Detected
- Holes and tears
- Stains and discoloration
- Wrinkles and creases
- Thread irregularities
- Pattern defects
- Color variations

## Output
- **Real-time display** with defect highlighting
- **Anomaly score** (0-1 confidence)
- **Color-coded status**: Green=Normal, Red=Defect
- **Processing metrics**: FPS, latency
- **Save functionality** for captured frames

## Hardware Requirements
- **Camera**: USB/CSI camera
- **Memory**: 4GB+ RAM, 2GB+ VRAM (for CUDA)
- **Processing**: CPU or NVIDIA GPU with CUDA support
- **OS**: Linux, Windows, macOS
- **GPU**: NVIDIA GTX 1060+ or RTX series (recommended for real-time performance)

### Edge Deployment
- **Jetson Orin Nano 8GB**: Optimized deployment package available
- **Expected Performance**: 20-40 FPS on Jetson with CUDA acceleration
- **Power Consumption**: 10-25W configurable

## Troubleshooting

### Camera Issues
```bash
# List available cameras
ls /dev/video*

# Test specific camera
python run_fabric_detection.py --camera 1
```

### Performance Issues
- **For best performance**: Use `--device cuda` with NVIDIA GPU
- **CUDA issues**: Run `python test_cuda.py` to diagnose CUDA problems
- **Fallback**: Use `--device cpu` for CPU-only processing
- **Benchmarking**: Run `python benchmark_cuda.py` to compare performance
- Reduce camera resolution if needed
- Close unnecessary applications

### Model Loading Slow
- Initial model loading takes 30-60 seconds (normal)
- Subsequent processing is real-time

## Integration Notes
- System designed for industrial fabric inspection
- Can be integrated with conveyor systems
- Supports automated quality control workflows
- Configurable detection thresholds
- Exportable results for quality reports
- **Edge deployment ready**: Jetson Orin Nano optimized package available

## Jetson Deployment

### Quick Deploy to Jetson Orin Nano
```bash
# Create deployment package (on PC)
cd jetson_deployment
./create_jetson_package.sh

# Transfer to Jetson (211MB package)
scp glass_jetson_deploy.tar.gz jetson@<jetson-ip>:~/

# Install on Jetson
tar -xzf glass_jetson_deploy.tar.gz
cd glass_jetson_deploy
sudo ./install_jetson.sh

# Run optimized system
glass-detect
```

### Jetson Performance
- **Processing**: 35-50ms per frame
- **Throughput**: 20-28 FPS sustained
- **Power**: 15W balanced mode
- **Memory**: <3GB usage
- **Optimizations**: FP16 precision, memory management, power profiles

**See**: [DEPLOY_TO_JETSON.md](DEPLOY_TO_JETSON.md) for complete deployment guide

## Support
- Check camera permissions
- Ensure adequate lighting
- Verify fabric positioning for optimal detection
- Adjust threshold based on fabric type and defect severity