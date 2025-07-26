# 🔧 Jetson Deployment Troubleshooting Guide

## ✅ Fixed Issues in Latest Package

### ❌ "No module named 'predict_single_image'"
**Status**: ✅ **FIXED** in latest package

**Solution Applied**:
- Updated installation script to create proper wrapper scripts
- Fixed Python path handling in all scripts
- Added PYTHONPATH environment variable setup

**New Installation**: 
```bash
# The latest package (glass_jetson_deploy.tar.gz) includes fixes
tar -xzf glass_jetson_deploy.tar.gz
cd glass_jetson_deploy  
sudo ./install_jetson.sh
```

### ❌ "No module named 'timm'"  
**Status**: ✅ **FIXED** in latest package

**Solution Applied**:
- Added `timm>=0.6.0` to Jetson requirements
- Updated installation script to install all dependencies

## 🚀 Quick Fix Commands (If Using Old Package)

### Manual Path Fix (Temporary)
```bash
# If you still get import errors, run from the installation directory:
cd /opt/glass
export PYTHONPATH=/opt/glass/utils:/opt/glass/scripts:$PYTHONPATH
python3 run_fabric_detection_jetson.py
```

### Manual Dependency Install
```bash
# Install missing dependencies manually:
pip3 install timm>=0.6.0
pip3 install torch torchvision
```

## 🔍 Verification Steps

### 1. Check Installation
```bash
# Verify files are in place
ls -la /opt/glass/
ls -la /usr/local/bin/glass-*

# Check wrapper scripts
cat /usr/local/bin/glass-detect
```

### 2. Test Python Paths
```bash
# Test module imports
cd /opt/glass
python3 -c "
import sys
sys.path.insert(0, '/opt/glass/utils')
sys.path.insert(0, '/opt/glass/scripts')
try:
    from predict_single_image import GLASSPredictor
    print('✅ predict_single_image imported successfully')
except Exception as e:
    print(f'❌ Import failed: {e}')
"
```

### 3. Test Dependencies
```bash
# Check all required packages
python3 -c "
packages = ['torch', 'torchvision', 'cv2', 'numpy', 'timm', 'PIL']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg} available')
    except ImportError as e:
        print(f'❌ {pkg} missing: {e}')
"
```

## 🛠️ Manual Installation (If Automated Install Fails)

### Step 1: Extract Package
```bash
tar -xzf glass_jetson_deploy.tar.gz
cd glass_jetson_deploy
```

### Step 2: Install Dependencies Manually
```bash
# Update system
sudo apt update

# Install Python packages
pip3 install -r requirements_jetson.txt

# Install PyTorch for Jetson (if needed)
wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-2.0.0-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.0.0-cp38-cp38-linux_aarch64.whl
```

### Step 3: Manual File Setup
```bash
# Copy files
sudo mkdir -p /opt/glass
sudo cp -r . /opt/glass/
sudo chown -R $USER:$USER /opt/glass

# Create wrapper script
cat > /tmp/glass-detect << 'EOF'
#!/bin/bash
cd /opt/glass
export PYTHONPATH=/opt/glass/utils:/opt/glass/scripts:$PYTHONPATH
python3 /opt/glass/run_fabric_detection_jetson.py "$@"
EOF

sudo mv /tmp/glass-detect /usr/local/bin/glass-detect
sudo chmod +x /usr/local/bin/glass-detect
```

### Step 4: Test Installation
```bash
# Verify installation
python3 /opt/glass/verify_jetson.py

# Test command
glass-detect --help
```

## 🎯 Expected Output After Fix

### Successful Installation
```bash
$ glass-detect
🚀 Starting Jetson-optimized fabric detection...
Loading model: models/wfdd_grid_cloth/ckpt_best_571.pth
✅ Model loaded and warmed up
✅ Camera initialized: 640x480
Controls: 'q' = quit, 's' = save frame
```

### Successful Verification  
```bash
$ glass-verify
🔍 GLASS Jetson System Verification
========================================

Jetson Platform:
✅ Platform: NVIDIA Jetson Orin Nano Developer Kit

CUDA Support:
✅ CUDA available: Orin

System Resources:
✅ Memory: 7.4GB total, 6.2GB available
✅ Temperature: 45.2°C

Model Files:
✅ Model files found: glass_simplified.onnx, ckpt_best_571.pth

Inference Test:
✅ Inference successful: 42.3ms

========================================
Results: 5/5 checks passed
🎉 All checks passed! System ready.
```

## 📞 Still Having Issues?

### Debug Information to Collect
```bash
# System info
cat /proc/device-tree/model
python3 --version
pip3 list | grep -E "(torch|timm|opencv)"

# Path information
echo $PYTHONPATH
ls -la /opt/glass/
ls -la /usr/local/bin/glass-*

# Test imports
cd /opt/glass
python3 -c "import sys; print('Python path:', sys.path)"
```

### Common Solutions
1. **Reinstall with latest package** - Download newest `glass_jetson_deploy.tar.gz`
2. **Check Jetson setup** - Ensure JetPack 5.1+ is installed
3. **Verify permissions** - Make sure installation was done with proper sudo access
4. **Manual path setup** - Export PYTHONPATH manually if wrappers fail

The latest package includes all these fixes and should work out of the box! 🚀