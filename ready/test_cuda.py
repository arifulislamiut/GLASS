#!/usr/bin/env python3
"""
Test CUDA functionality for GLASS system
"""
import torch
import onnxruntime as ort
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def test_pytorch_cuda():
    """Test PyTorch CUDA functionality"""
    print("=== PyTorch CUDA Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        
        # Test tensor operations
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✓ PyTorch CUDA operations working")
            return True
        except Exception as e:
            print(f"✗ PyTorch CUDA error: {e}")
            return False
    else:
        print("✗ PyTorch CUDA not available")
        return False

def test_onnx_cuda():
    """Test ONNX Runtime CUDA functionality"""
    print("\n=== ONNX Runtime CUDA Test ===")
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Check if CUDA provider is available
    if 'CUDAExecutionProvider' not in providers:
        print("✗ CUDAExecutionProvider not available")
        return False
    
    # Test ONNX session creation with CUDA
    try:
        # Try to create a session with CUDA provider
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'wfdd_grid_cloth', 'glass_simplified.onnx')
        
        if not os.path.exists(model_path):
            print(f"✗ Model not found at {model_path}")
            return False
        
        print(f"Testing ONNX model: {model_path}")
        
        # Try CUDA first
        cuda_providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        
        session = ort.InferenceSession(model_path, providers=cuda_providers)
        
        # Check which provider is actually being used
        actual_providers = session.get_providers()
        print(f"Session providers: {actual_providers}")
        
        if 'CUDAExecutionProvider' in actual_providers:
            print("✓ ONNX Runtime using CUDA")
            
            # Test inference
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(f"Input: {input_name}, Shape: {input_shape}")
            
            # Create dummy input
            import numpy as np
            dummy_input = np.random.randn(1, 3, 288, 288).astype(np.float32)
            
            result = session.run(None, {input_name: dummy_input})
            print("✓ ONNX CUDA inference test successful")
            return True
        else:
            print("✗ ONNX Runtime falling back to CPU")
            return False
            
    except Exception as e:
        print(f"✗ ONNX CUDA error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_glass_cuda():
    """Test GLASS model with CUDA"""
    print("\n=== GLASS CUDA Test ===")
    
    onnx_works = False
    pytorch_works = False
    
    try:
        from predict_single_image import GLASSPredictor
        from PIL import Image
        import numpy as np
        
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'wfdd_grid_cloth')
        
        # Create dummy image
        dummy_img_array = np.random.randint(0, 255, (288, 288, 3), dtype=np.uint8)
        dummy_img = Image.fromarray(dummy_img_array)
        
        # Test ONNX with CUDA
        print("Testing GLASS ONNX with CUDA...")
        try:
            predictor_onnx = GLASSPredictor(model_path, "onnx", "cuda")
            result_onnx = predictor_onnx.predict(dummy_img)
            
            if result_onnx:
                print("✓ GLASS ONNX CUDA prediction successful")
                print(f"  Score: {result_onnx.get('image_score', 'N/A'):.4f}")
                onnx_works = True
            else:
                print("✗ GLASS ONNX CUDA prediction failed")
        except Exception as e:
            print(f"✗ GLASS ONNX CUDA error: {e}")
        
        # Test PyTorch with CUDA
        print("\nTesting GLASS PyTorch with CUDA...")
        try:
            predictor_pytorch = GLASSPredictor(model_path, "pytorch", "cuda")
            
            import time
            start_time = time.time()
            result_pytorch = predictor_pytorch.predict(dummy_img)
            inference_time = time.time() - start_time
            
            if result_pytorch:
                print("✓ GLASS PyTorch CUDA prediction successful")
                print(f"  Score: {result_pytorch.get('image_score', 'N/A'):.4f}")
                print(f"  Inference time: {inference_time*1000:.1f}ms")
                pytorch_works = True
            else:
                print("✗ GLASS PyTorch CUDA prediction failed")
        except Exception as e:
            print(f"✗ GLASS PyTorch CUDA error: {e}")
            import traceback
            traceback.print_exc()
        
        return onnx_works or pytorch_works
            
    except Exception as e:
        print(f"✗ GLASS CUDA setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("CUDA Functionality Test for GLASS Fabric Detection")
    print("=" * 60)
    
    # Test individual components
    pytorch_ok = test_pytorch_cuda()
    onnx_ok = test_onnx_cuda()
    glass_ok = test_glass_cuda()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"PyTorch CUDA: {'✓ PASS' if pytorch_ok else '✗ FAIL'}")
    print(f"ONNX CUDA: {'✓ PASS' if onnx_ok else '✗ FAIL'}")
    print(f"GLASS CUDA: {'✓ PASS' if glass_ok else '✗ FAIL'}")
    
    if pytorch_ok and glass_ok:
        print("\n🎉 CUDA is ready for GLASS! Using PyTorch backend for acceleration.")
        print("\nTo run with CUDA:")
        print("python run_fabric_detection.py --device cuda")
        if not onnx_ok:
            print("\nNote: ONNX Runtime CUDA has cuDNN version conflicts, but PyTorch CUDA works perfectly.")
            print("The system will automatically use PyTorch mode for CUDA acceleration.")
    elif glass_ok:
        print("\n✅ CUDA partially working. Some acceleration available.")
        print("python run_fabric_detection.py --device cuda")
    else:
        print("\n⚠️  CUDA tests failed. Using CPU mode:")
        print("python run_fabric_detection.py --device cpu")
        
        if not pytorch_ok:
            print("\nTroubleshooting PyTorch CUDA:")
            print("- Reinstall PyTorch with CUDA support")
            print("- Check NVIDIA driver compatibility")
        
        if not onnx_ok:
            print("\nONNX Runtime CUDA issue:")
            print("- Requires cuDNN 9.x but system has cuDNN 8.9")
            print("- PyTorch mode provides CUDA acceleration instead")

if __name__ == "__main__":
    main()