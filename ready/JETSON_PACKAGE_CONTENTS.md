# GLASS Jetson Deployment Package - Contents

## Package Details
- **File**: `glass_jetson_deploy.tar.gz`
- **Size**: 211MB
- **Created**: $(date)

## Fixed Files Included

### Main Runner Scripts (All with Fixes Applied)
1. **`run_fabric_detection_jetson_fixed.py`** - Simple Jetson version with all fixes
   - ✅ Text positioning fixes (no overlapping)
   - ✅ Help overlay system with auto-hide timer
   - ✅ Controls positioned in main camera area
   - ✅ Better system stats spacing

2. **`run_fabric_detection_jetson_complete.py`** - Full PC functionality for Jetson
   - ✅ All display modes (0-3) with proper text positioning
   - ✅ Help overlay system with comprehensive Jetson info
   - ✅ Resolution controls (+/- keys)
   - ✅ Display mode cycling (d key)
   - ✅ Multi-panel layout with no text overlap

3. **`run_fabric_detection_jetson.py`** - Basic optimized runner (generated)

### Core System Files
- **`utils/`** - All GLASS utility modules
- **`scripts/`** - Prediction scripts and helpers
- **`models/`** - Pre-trained model files (.pth and .onnx)

### Jetson-Specific Files
- **`install_jetson.sh`** - Automated installation script
- **`requirements_jetson.txt`** - Jetson-specific dependencies
- **`jetson_config.yaml`** - Optimized configuration
- **`verify_jetson.py`** - System verification script
- **`optimize_model_jetson.py`** - Model optimization for Jetson
- **`diagnose_display.py`** - Display troubleshooting

### Documentation
- **`README.md`** - Jetson-specific installation and usage guide
- **`docs/PERFORMANCE.md`** - Performance optimization tips
- **`docs/TROUBLESHOOTING.md`** - Common issue solutions
- **`docs/POWER_MANAGEMENT.md`** - Jetson power mode guidance

## Key Fixes Applied

### 1. Text Positioning Fixes
- **Problem**: Overlapping control text in various display areas
- **Solution**: 
  - Controls moved to main camera panel area
  - Proper line spacing (22px instead of cramped positioning)
  - Mode-specific placement for different display layouts

### 2. Help Overlay System
- **Feature**: Press 'h' to show comprehensive help
- **Auto-hide**: Help disappears after 5 seconds
- **Content**: 
  - All keyboard controls
  - Current system settings
  - Jetson-specific information
  - Performance tips

### 3. Display Mode Support
- **Mode 0**: 2x2 grid (All Views) - help/controls in main panel
- **Mode 1**: Original + Anomaly Map - help/controls in left panel
- **Mode 2**: Original + Overlay - help/controls in left panel  
- **Mode 3**: Original Only - help/controls at bottom

### 4. Jetson Optimizations
- **CUDA optimizations**: FP16 precision, memory management
- **Temperature monitoring**: Real-time Jetson temperature display
- **Power mode awareness**: Respects Jetson power profiles
- **GStreamer support**: For native Jetson cameras

## Installation Instructions

### 1. Transfer to Jetson
```bash
scp glass_jetson_deploy.tar.gz jetson@<jetson-ip>:~/
```

### 2. Install on Jetson
```bash
tar -xzf glass_jetson_deploy.tar.gz
cd glass_jetson_deploy
sudo ./install_jetson.sh
```

### 3. Verify Installation
```bash
python3 verify_jetson.py
```

### 4. Run Detection
```bash
# Simple fixed version
python3 run_fabric_detection_jetson_fixed.py

# Complete version with all features
python3 run_fabric_detection_jetson_complete.py

# Basic optimized version  
python3 run_fabric_detection_jetson.py
```

## Controls (All Versions)

- **q** - Quit application
- **s** - Save current frame
- **h** - Show/hide help overlay (auto-hides in 5 seconds)
- **d** - Cycle display modes (complete version only)
- **+/-** - Increase/decrease resolution (complete version only)
- **r** - Reset resolution to default (complete version only)

## Requirements

- **Hardware**: NVIDIA Jetson Orin Nano (or compatible)
- **Software**: JetPack 5.1+ with CUDA support
- **Camera**: USB camera or CSI camera compatible with OpenCV
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space

## Package Status
✅ **Ready for deployment** - All fixes applied and tested