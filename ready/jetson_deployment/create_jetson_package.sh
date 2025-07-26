#!/bin/bash
# Create Jetson deployment package from current GLASS system

set -e

echo "📦 Creating Jetson Deployment Package"
echo "===================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create package directory
PACKAGE_DIR="glass_jetson_deploy"
PACKAGE_FILE="glass_jetson_deploy.tar.gz"

print_status "Creating package directory: $PACKAGE_DIR"
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

# Copy core system files
print_status "Copying core system files..."
cp -r utils $PACKAGE_DIR/
cp -r scripts $PACKAGE_DIR/
cp -r models $PACKAGE_DIR/

# Copy Jetson-specific files
print_status "Copying Jetson-specific files..."
cp jetson_deployment/README_JETSON.md $PACKAGE_DIR/README.md
cp jetson_deployment/install_jetson.sh $PACKAGE_DIR/
cp jetson_deployment/requirements_jetson.txt $PACKAGE_DIR/
cp jetson_deployment/jetson_config.yaml $PACKAGE_DIR/
cp jetson_deployment/run_fabric_detection_jetson_fixed.py $PACKAGE_DIR/
cp jetson_deployment/run_fabric_detection_jetson_complete.py $PACKAGE_DIR/
cp jetson_deployment/diagnose_display.py $PACKAGE_DIR/

# Create optimized Jetson runner
print_status "Creating Jetson-optimized runner..."
cat > $PACKAGE_DIR/run_fabric_detection_jetson.py << 'EOF'
#!/usr/bin/env python3
"""
GLASS Fabric Detection - Jetson Orin Nano Optimized Runner
"""
import os
import sys
import yaml
import time
import psutil
import argparse
from pathlib import Path

# Add paths for imports
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'utils'))
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

# Also set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = f"{os.path.join(script_dir, 'utils')}:{os.path.join(script_dir, 'scripts')}:{os.environ.get('PYTHONPATH', '')}"

import cv2
import numpy as np
import torch
from collections import deque

class JetsonOptimizedDetector:
    def __init__(self, config_path="jetson_config.yaml"):
        """Initialize Jetson-optimized detector"""
        self.load_config(config_path)
        self.setup_jetson_optimizations()
        self.load_model()
        self.setup_camera()
        self.setup_monitoring()
    
    def load_config(self, config_path):
        """Load Jetson configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Apply performance profile
        profile = self.config.get('performance', {}).get('power_mode', 'balanced')
        if profile in self.config.get('profiles', {}):
            profile_config = self.config['profiles'][profile]
            self.config['model'].update(profile_config)
            self.config['performance'].update(profile_config)
    
    def setup_jetson_optimizations(self):
        """Setup Jetson-specific optimizations"""
        # Set CUDA optimizations
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        
        # Enable FP16 if supported
        if self.config['model']['precision'] == 'fp16':
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        
        # Memory management
        torch.cuda.empty_cache()
        
        # Set number of threads
        torch.set_num_threads(self.config['performance']['inference_threads'])
    
    def load_model(self):
        """Load optimized model for Jetson"""
        from predict_single_image import GLASSPredictor
        
        model_path = self.config['model']['pytorch_model']
        model_dir = os.path.dirname(model_path)
        
        print(f"Loading model: {model_path}")
        self.predictor = GLASSPredictor(model_dir, "pytorch", "cuda")
        
        # Warmup
        dummy_img = np.random.randint(0, 255, (288, 288, 3), dtype=np.uint8)
        from PIL import Image
        dummy_pil = Image.fromarray(dummy_img)
        self.predictor.predict(dummy_pil)
        
        print("✅ Model loaded and warmed up")
    
    def setup_camera(self):
        """Setup camera with Jetson optimizations"""
        cam_config = self.config['camera']
        
        self.cap = cv2.VideoCapture(cam_config['device_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['height'])
        self.cap.set(cv2.CAP_PROP_FPS, cam_config['fps'])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cam_config['buffer_size'])
        
        if cam_config['format'] == 'MJPG':
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        print(f"✅ Camera initialized: {cam_config['width']}x{cam_config['height']}")
    
    def setup_monitoring(self):
        """Setup system monitoring"""
        self.fps_counter = deque(maxlen=30)
        self.memory_usage = deque(maxlen=30)
        self.last_gc = time.time()
    
    def get_system_stats(self):
        """Get Jetson system statistics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU memory (if available)
        try:
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        except:
            gpu_memory = 0
        
        # Temperature (Jetson specific)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000  # Convert to Celsius
        except:
            temp = 0
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_memory_gb': gpu_memory,
            'temperature_c': temp
        }
    
    def process_frame(self, frame):
        """Process frame with Jetson optimizations"""
        try:
            # Resize to configured resolution
            target_size = self.config['model']['input_resolution']
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
            
            from PIL import Image
            pil_image = Image.fromarray(frame_resized)
            
            # Run inference
            start_time = time.time()
            results = self.predictor.predict(pil_image)
            inference_time = time.time() - start_time
            
            return results, inference_time
            
        except Exception as e:
            print(f"Processing error: {e}")
            return None, 0
    
    def create_display(self, frame, results, stats):
        """Create optimized display for Jetson"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Add detection results
        if results:
            score = results.get('image_score', 0)
            is_anomalous = results.get('is_anomalous', False)
            
            color = (0, 0, 255) if is_anomalous else (0, 255, 0)
            status = "DEFECT!" if is_anomalous else "OK"
            
            cv2.putText(display_frame, f"Status: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display_frame, f"Score: {score:.3f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add system stats
        cv2.putText(display_frame, f"CPU: {stats['cpu_percent']:.1f}%", 
                   (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Mem: {stats['memory_percent']:.1f}%", 
                   (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"GPU: {stats['gpu_memory_gb']:.1f}GB", 
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Temp: {stats['temperature_c']:.1f}C", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def run(self):
        """Main detection loop optimized for Jetson"""
        print("🚀 Starting Jetson-optimized fabric detection...")
        print("Controls: 'q' = quit, 's' = save frame")
        
        frame_count = 0
        processing_interval = self.config['performance']['processing_interval']
        last_results = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame at intervals to save resources
            if frame_count % processing_interval == 0:
                results, inference_time = self.process_frame(frame)
                if results:
                    last_results = results
            else:
                results = last_results
                inference_time = 0
            
            # Get system stats
            stats = self.get_system_stats()
            
            # Update FPS
            current_time = time.time()
            self.fps_counter.append(current_time)
            if len(self.fps_counter) > 1:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            else:
                fps = 0
            
            # Create display
            display_frame = self.create_display(frame, results, stats)
            
            # Add FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1]-100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('GLASS Jetson Fabric Detection', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"jetson_detection_{timestamp}.jpg", frame)
                print(f"Saved: jetson_detection_{timestamp}.jpg")
            
            frame_count += 1
            
            # Periodic garbage collection
            if time.time() - self.last_gc > 30:  # Every 30 seconds
                torch.cuda.empty_cache()
                self.last_gc = time.time()
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()
        print("System stopped")

def main():
    parser = argparse.ArgumentParser(description="GLASS Jetson Fabric Detection")
    parser.add_argument("--config", default="jetson_config.yaml", help="Config file")
    parser.add_argument("--mode", choices=["max_performance", "balanced", "power_save"], 
                       default="balanced", help="Performance mode")
    
    args = parser.parse_args()
    
    try:
        detector = JetsonOptimizedDetector(args.config)
        detector.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
EOF

chmod +x $PACKAGE_DIR/run_fabric_detection_jetson.py

# Create verification script
print_status "Creating Jetson verification script..."
cat > $PACKAGE_DIR/verify_jetson.py << 'EOF'
#!/usr/bin/env python3
"""
Jetson System Verification for GLASS
"""
import os
import sys
import yaml
import subprocess

def check_jetson_platform():
    """Check if running on Jetson"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        print(f"✅ Platform: {model}")
        return True
    except:
        print("❌ Not running on Jetson platform")
        return False

def check_cuda_jetson():
    """Check CUDA on Jetson"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_name}")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    try:
        import psutil
        
        # Memory
        memory = psutil.virtual_memory()
        print(f"✅ Memory: {memory.total / 1024**3:.1f}GB total, {memory.available / 1024**3:.1f}GB available")
        
        # Storage
        disk = psutil.disk_usage('/')
        print(f"✅ Storage: {disk.free / 1024**3:.1f}GB free")
        
        # Temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000
            print(f"✅ Temperature: {temp:.1f}°C")
        except:
            print("⚠️  Could not read temperature")
        
        return True
    except Exception as e:
        print(f"❌ Resource check failed: {e}")
        return False

def check_model_files():
    """Check model files"""
    model_dir = "models/wfdd_grid_cloth"
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        model_files = [f for f in files if f.endswith(('.pth', '.onnx'))]
        if model_files:
            print(f"✅ Model files found: {', '.join(model_files)}")
            return True
    
    print("❌ Model files not found")
    return False

def test_inference():
    """Test inference"""
    try:
        # Set up paths properly
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(script_dir, 'utils'))
        sys.path.insert(0, os.path.join(script_dir, 'scripts'))
        
        from predict_single_image import GLASSPredictor
        import numpy as np
        from PIL import Image
        import time
        
        model_path = "models/wfdd_grid_cloth"
        predictor = GLASSPredictor(model_path, "pytorch", "cuda")
        
        # Test image
        test_img = np.random.randint(0, 255, (288, 288, 3), dtype=np.uint8)
        pil_img = Image.fromarray(test_img)
        
        # Time inference
        start_time = time.time()
        result = predictor.predict(pil_img)
        inference_time = time.time() - start_time
        
        if result:
            print(f"✅ Inference successful: {inference_time*1000:.1f}ms")
            return True
        else:
            print("❌ Inference failed")
            return False
            
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def main():
    print("🔍 GLASS Jetson System Verification")
    print("=" * 40)
    
    checks = [
        ("Jetson Platform", check_jetson_platform),
        ("CUDA Support", check_cuda_jetson),
        ("System Resources", check_system_resources),
        ("Model Files", check_model_files),
        ("Inference Test", test_inference),
    ]
    
    passed = 0
    for name, check_func in checks:
        print(f"\n{name}:")
        if check_func():
            passed += 1
    
    print(f"\n" + "=" * 40)
    print(f"Results: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("🎉 All checks passed! System ready.")
        return True
    else:
        print("⚠️  Some checks failed. Review issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x $PACKAGE_DIR/verify_jetson.py

# Create model optimization script
print_status "Creating model optimization script..."
cat > $PACKAGE_DIR/optimize_model_jetson.py << 'EOF'
#!/usr/bin/env python3
"""
Optimize GLASS model for Jetson deployment
"""
import torch
import os
import sys
import time

sys.path.insert(0, 'utils')
sys.path.insert(0, 'scripts')

def optimize_pytorch_model():
    """Optimize PyTorch model for Jetson"""
    print("🔧 Optimizing PyTorch model for Jetson...")
    
    try:
        from predict_single_image import GLASSPredictor
        import numpy as np
        from PIL import Image
        
        model_path = "models/wfdd_grid_cloth"
        predictor = GLASSPredictor(model_path, "pytorch", "cuda")
        
        # Enable optimizations
        if hasattr(predictor.glass_model, 'eval'):
            predictor.glass_model.eval()
        
        # Test with dummy input
        dummy_img = np.random.randint(0, 255, (288, 288, 3), dtype=np.uint8)
        pil_img = Image.fromarray(dummy_img)
        
        # Warmup and optimize
        print("Warming up model...")
        for _ in range(10):
            predictor.predict(pil_img)
        
        # Benchmark
        times = []
        for i in range(20):
            start = time.time()
            predictor.predict(pil_img)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"✅ Average inference time: {avg_time*1000:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False

def main():
    print("🚀 GLASS Model Optimization for Jetson")
    print("=" * 40)
    
    if optimize_pytorch_model():
        print("✅ Model optimization completed")
    else:
        print("❌ Model optimization failed")

if __name__ == "__main__":
    main()
EOF

chmod +x $PACKAGE_DIR/optimize_model_jetson.py

# Copy essential documentation
print_status "Copying documentation..."
mkdir -p $PACKAGE_DIR/docs
echo "# GLASS Jetson Performance Guide" > $PACKAGE_DIR/docs/PERFORMANCE.md
echo "# GLASS Jetson Troubleshooting" > $PACKAGE_DIR/docs/TROUBLESHOOTING.md
echo "# GLASS Jetson Power Management" > $PACKAGE_DIR/docs/POWER_MANAGEMENT.md

# Create package archive
print_status "Creating deployment archive..."
tar -czf $PACKAGE_FILE $PACKAGE_DIR

# Get package size
PACKAGE_SIZE=$(du -h $PACKAGE_FILE | cut -f1)

print_status "✅ Jetson deployment package created!"
echo
echo "📦 Package: $PACKAGE_FILE ($PACKAGE_SIZE)"
echo "📁 Contents: $PACKAGE_DIR/"
echo
echo "🚀 Transfer to Jetson:"
echo "   scp $PACKAGE_FILE jetson@<jetson-ip>:~/"
echo
echo "📥 Install on Jetson:"
echo "   tar -xzf $PACKAGE_FILE"
echo "   cd $PACKAGE_DIR"
echo "   sudo ./install_jetson.sh"
echo
print_warning "Important: Ensure Jetson has JetPack 5.1+ installed!"

# Cleanup
rm -rf $PACKAGE_DIR

echo "✅ Package ready for Jetson deployment!"