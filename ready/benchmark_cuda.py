#!/usr/bin/env python3
"""
CUDA Performance Benchmark for GLASS Fabric Detection
"""
import time
import numpy as np
import os
import sys
from PIL import Image

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def benchmark_inference(device="cuda", num_tests=20):
    """Benchmark inference performance"""
    print(f"GLASS Inference Benchmark - {device.upper()}")
    print("=" * 50)
    
    try:
        from predict_single_image import GLASSPredictor
        
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'wfdd_grid_cloth')
        
        # Use PyTorch mode for CUDA (due to ONNX Runtime cuDNN conflicts)
        model_type = "pytorch" if device == "cuda" else "onnx"
        
        print(f"Loading model: {model_type} on {device}")
        start_load = time.time()
        predictor = GLASSPredictor(model_path, model_type, device)
        load_time = time.time() - start_load
        print(f"Model loading time: {load_time:.2f}s")
        
        # Create test images
        print(f"\nRunning {num_tests} inference tests...")
        test_images = []
        for i in range(num_tests):
            # Create diverse test images
            if i % 4 == 0:
                # Normal fabric texture
                img_array = np.random.randint(100, 150, (288, 288, 3), dtype=np.uint8)
            elif i % 4 == 1:
                # Fabric with defect pattern
                img_array = np.random.randint(80, 120, (288, 288, 3), dtype=np.uint8)
                img_array[100:150, 100:150] = [255, 0, 0]  # Red defect
            elif i % 4 == 2:
                # Grid pattern
                img_array = np.zeros((288, 288, 3), dtype=np.uint8) + 120
                for j in range(0, 288, 20):
                    img_array[j:j+2, :] = 150
                    img_array[:, j:j+2] = 150
            else:
                # Complex pattern
                img_array = np.random.randint(60, 180, (288, 288, 3), dtype=np.uint8)
            
            test_images.append(Image.fromarray(img_array))
        
        # Warmup
        print("Warming up...")
        for i in range(3):
            predictor.predict(test_images[0])
        
        # Benchmark
        inference_times = []
        scores = []
        
        print("Benchmarking...")
        for i, img in enumerate(test_images):
            start_time = time.time()
            result = predictor.predict(img)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            if result:
                scores.append(result.get('image_score', 0))
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_tests} tests")
        
        # Statistics
        times_ms = [t * 1000 for t in inference_times]
        
        print(f"\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Device: {device.upper()}")
        print(f"Model: {model_type}")
        print(f"Tests: {num_tests}")
        print(f"\nTiming Results:")
        print(f"  Mean: {np.mean(times_ms):.1f}ms")
        print(f"  Median: {np.median(times_ms):.1f}ms")
        print(f"  Min: {np.min(times_ms):.1f}ms")
        print(f"  Max: {np.max(times_ms):.1f}ms")
        print(f"  Std: {np.std(times_ms):.1f}ms")
        print(f"\nThroughput: {1000/np.mean(times_ms):.1f} FPS")
        
        if scores:
            print(f"\nDetection Results:")
            print(f"  Mean score: {np.mean(scores):.3f}")
            print(f"  Score range: {np.min(scores):.3f} - {np.max(scores):.3f}")
        
        return np.mean(times_ms)
        
    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_cuda_cpu():
    """Compare CUDA vs CPU performance"""
    print("CUDA vs CPU Performance Comparison")
    print("=" * 60)
    
    # Test CUDA
    cuda_time = benchmark_inference("cuda", 10)
    
    print("\n" + "="*60)
    
    # Test CPU
    cpu_time = benchmark_inference("cpu", 10)
    
    if cuda_time and cpu_time:
        speedup = cpu_time / cuda_time
        print(f"\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"CUDA: {cuda_time:.1f}ms average")
        print(f"CPU:  {cpu_time:.1f}ms average")
        print(f"Speedup: {speedup:.1f}x faster with CUDA")
        print(f"Performance gain: {(speedup-1)*100:.1f}%")
        
        if speedup > 2:
            print("🚀 Excellent CUDA acceleration!")
        elif speedup > 1.5:
            print("✅ Good CUDA acceleration")
        else:
            print("⚠️  Modest CUDA acceleration")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GLASS CUDA Benchmark")
    parser.add_argument("--device", choices=["cuda", "cpu", "compare"], default="compare",
                       help="Device to benchmark (default: compare both)")
    parser.add_argument("--tests", type=int, default=20, help="Number of inference tests")
    
    args = parser.parse_args()
    
    if args.device == "compare":
        compare_cuda_cpu()
    else:
        benchmark_inference(args.device, args.tests)

if __name__ == "__main__":
    main()