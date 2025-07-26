#!/usr/bin/env python3
"""
Complete System Verification for GLASS Fabric Detection
"""
import os
import sys

def main():
    print("🔍 GLASS Fabric Detection System Verification")
    print("=" * 60)
    
    print("\n1. Testing basic dependencies...")
    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        print("   ✅ Core dependencies available")
    except Exception as e:
        print(f"   ❌ Missing dependencies: {e}")
        return False
    
    print("\n2. Testing camera access...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"   ✅ Camera available: {width}x{height}")
            cap.release()
        else:
            print("   ⚠️  Camera not accessible")
    except Exception as e:
        print(f"   ❌ Camera error: {e}")
    
    print("\n3. Testing CUDA availability...")
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available: {gpu_name}")
        else:
            print("   ⚠️  CUDA not available, will use CPU")
    except Exception as e:
        print(f"   ❌ CUDA check error: {e}")
    
    print("\n4. Testing model files...")
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'wfdd_grid_cloth')
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        onnx_files = [f for f in files if f.endswith('.onnx')]
        pth_files = [f for f in files if f.endswith('.pth')]
        
        if onnx_files or pth_files:
            print(f"   ✅ Model files found:")
            for f in onnx_files + pth_files:
                print(f"      - {f}")
        else:
            print("   ❌ No model files found")
            return False
    else:
        print("   ❌ Model directory not found")
        return False
    
    print("\n5. Testing inference system...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
        
        from predict_single_image import GLASSPredictor
        
        # Test with best available device
        device = "cuda" if cuda_available else "cpu"
        model_type = "pytorch" if device == "cuda" else "onnx"
        
        print(f"   Testing {model_type} model on {device}...")
        predictor = GLASSPredictor(model_path, model_type, device)
        
        # Create test image
        test_img_array = np.random.randint(0, 255, (288, 288, 3), dtype=np.uint8)
        test_img = Image.fromarray(test_img_array)
        
        import time
        start_time = time.time()
        result = predictor.predict(test_img)
        inference_time = time.time() - start_time
        
        if result:
            score = result.get('image_score', 0)
            print(f"   ✅ Inference successful:")
            print(f"      - Score: {score:.3f}")
            print(f"      - Time: {inference_time*1000:.1f}ms")
            print(f"      - Device: {device.upper()}")
        else:
            print("   ❌ Inference failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n6. System ready tests...")
    scripts = [
        'run_fabric_detection.py',
        'test_system.py', 
        'test_cuda.py',
        'benchmark_cuda.py',
        'demo_display_modes.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} missing")
    
    print("\n" + "=" * 60)
    print("🎉 SYSTEM VERIFICATION COMPLETE")
    print("=" * 60)
    
    print(f"\n📊 Performance Summary:")
    if cuda_available:
        print(f"   🚀 CUDA Acceleration: ENABLED")
        print(f"   ⚡ Expected performance: ~13ms per frame (75+ FPS)")
    else:
        print(f"   🔄 CPU Processing: ENABLED") 
        print(f"   ⏱️  Expected performance: ~100-200ms per frame (5-10 FPS)")
    
    print(f"\n🚀 Ready to run:")
    recommended_cmd = f"python run_fabric_detection.py --device {'cuda' if cuda_available else 'cpu'}"
    print(f"   {recommended_cmd}")
    
    print(f"\n🔧 Available tools:")
    print(f"   python test_cuda.py          # Test CUDA functionality")
    print(f"   python benchmark_cuda.py     # Performance benchmark")
    print(f"   python demo_display_modes.py # Demo display modes")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)