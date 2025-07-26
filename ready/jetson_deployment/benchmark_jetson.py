#!/usr/bin/env python3
"""
Jetson Performance Benchmark for GLASS
"""
import time
import os
import sys
import psutil
import yaml
import numpy as np
from PIL import Image

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'utils'))
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

# Also set PYTHONPATH environment variable  
os.environ['PYTHONPATH'] = f"{os.path.join(script_dir, 'utils')}:{os.path.join(script_dir, 'scripts')}:{os.environ.get('PYTHONPATH', '')}"

def get_jetson_stats():
    """Get Jetson-specific system stats"""
    stats = {}
    
    # CPU usage
    stats['cpu_percent'] = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    stats['memory_percent'] = memory.percent
    stats['memory_total_gb'] = memory.total / 1024**3
    stats['memory_available_gb'] = memory.available / 1024**3
    
    # GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            stats['gpu_memory_gb'] = torch.cuda.memory_allocated() / 1024**3
            stats['gpu_memory_cached_gb'] = torch.cuda.memory_reserved() / 1024**3
    except:
        pass
    
    # Temperature
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            stats['temperature_c'] = int(f.read().strip()) / 1000
    except:
        stats['temperature_c'] = 0
    
    # Power (if available)
    try:
        with open('/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio_device/in_power0_input', 'r') as f:
            stats['power_mw'] = int(f.read().strip())
    except:
        stats['power_mw'] = 0
    
    return stats

def benchmark_inference_jetson(model_path, num_tests=20, resolution=256):
    """Benchmark inference performance on Jetson"""
    print(f"🚀 Jetson Inference Benchmark")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Tests: {num_tests}")
    print("=" * 50)
    
    try:
        from predict_single_image import GLASSPredictor
        
        # Load model
        print("Loading PyTorch model...")
        start_load = time.time()
        predictor = GLASSPredictor(model_path, "pytorch", "cuda")
        load_time = time.time() - start_load
        print(f"Model load time: {load_time:.2f}s")
        
        # Get initial stats
        initial_stats = get_jetson_stats()
        print(f"Initial temperature: {initial_stats['temperature_c']:.1f}°C")
        print(f"Initial memory: {initial_stats['memory_percent']:.1f}%")
        
        # Create test images
        test_images = []
        for i in range(num_tests):
            if i % 3 == 0:
                # Normal fabric
                img_array = np.random.randint(100, 150, (resolution, resolution, 3), dtype=np.uint8)
            elif i % 3 == 1:
                # Fabric with defect
                img_array = np.random.randint(80, 120, (resolution, resolution, 3), dtype=np.uint8)
                img_array[50:100, 50:100] = [255, 0, 0]  # Red defect
            else:
                # Grid pattern
                img_array = np.zeros((resolution, resolution, 3), dtype=np.uint8) + 120
                for j in range(0, resolution, 20):
                    img_array[j:j+2, :] = 150
                    img_array[:, j:j+2] = 150
            
            test_images.append(Image.fromarray(img_array))
        
        # Warmup
        print("Warming up...")
        for i in range(5):
            predictor.predict(test_images[0])
        
        # Clear GPU cache
        import torch
        torch.cuda.empty_cache()
        
        # Benchmark
        print("Running benchmark...")
        inference_times = []
        system_stats = []
        
        for i, img in enumerate(test_images):
            # Get pre-inference stats
            pre_stats = get_jetson_stats()
            
            # Run inference
            start_time = time.time()
            result = predictor.predict(img)
            inference_time = time.time() - start_time
            
            # Get post-inference stats
            post_stats = get_jetson_stats()
            
            inference_times.append(inference_time)
            system_stats.append({
                'inference_time': inference_time,
                'temperature': post_stats['temperature_c'],
                'memory_percent': post_stats['memory_percent'],
                'cpu_percent': post_stats['cpu_percent'],
                'gpu_memory_gb': post_stats.get('gpu_memory_gb', 0)
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_tests} tests")
        
        # Results
        times_ms = [t * 1000 for t in inference_times]
        
        print(f"\n" + "=" * 50)
        print("📊 BENCHMARK RESULTS")
        print("=" * 50)
        
        # Timing results
        print(f"Inference Performance:")
        print(f"  Mean: {np.mean(times_ms):.1f}ms")
        print(f"  Median: {np.median(times_ms):.1f}ms")
        print(f"  Min: {np.min(times_ms):.1f}ms")
        print(f"  Max: {np.max(times_ms):.1f}ms")
        print(f"  Std: {np.std(times_ms):.1f}ms")
        print(f"  Throughput: {1000/np.mean(times_ms):.1f} FPS")
        
        # System performance
        avg_temp = np.mean([s['temperature'] for s in system_stats])
        max_temp = np.max([s['temperature'] for s in system_stats])
        avg_memory = np.mean([s['memory_percent'] for s in system_stats])
        max_memory = np.max([s['memory_percent'] for s in system_stats])
        avg_cpu = np.mean([s['cpu_percent'] for s in system_stats])
        avg_gpu_mem = np.mean([s['gpu_memory_gb'] for s in system_stats])
        
        print(f"\nSystem Performance:")
        print(f"  Temperature: {avg_temp:.1f}°C avg, {max_temp:.1f}°C max")
        print(f"  Memory: {avg_memory:.1f}% avg, {max_memory:.1f}% max")
        print(f"  CPU: {avg_cpu:.1f}% avg")
        print(f"  GPU Memory: {avg_gpu_mem:.2f}GB avg")
        
        # Performance rating
        if np.mean(times_ms) < 30:
            rating = "🚀 EXCELLENT"
        elif np.mean(times_ms) < 50:
            rating = "✅ GOOD"
        elif np.mean(times_ms) < 100:
            rating = "⚠️  ACCEPTABLE"
        else:
            rating = "❌ NEEDS OPTIMIZATION"
        
        print(f"\nPerformance Rating: {rating}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if max_temp > 70:
            print(f"  - Temperature is high ({max_temp:.1f}°C), consider cooling")
        if max_memory > 85:
            print(f"  - Memory usage is high ({max_memory:.1f}%), reduce batch size")
        if np.mean(times_ms) > 50:
            print(f"  - Consider reducing resolution or using INT8 quantization")
        if avg_cpu > 80:
            print(f"  - High CPU usage, optimize preprocessing")
        
        return np.mean(times_ms)
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def profile_different_resolutions():
    """Profile performance at different resolutions"""
    print("🔍 Resolution Performance Profile")
    print("=" * 40)
    
    model_path = "models/wfdd_grid_cloth"
    resolutions = [224, 256, 288]
    results = {}
    
    for res in resolutions:
        print(f"\nTesting {res}x{res}...")
        avg_time = benchmark_inference_jetson(model_path, num_tests=10, resolution=res)
        if avg_time:
            results[res] = avg_time
    
    if results:
        print(f"\n" + "=" * 40)
        print("📈 RESOLUTION COMPARISON")
        print("=" * 40)
        for res, time_ms in results.items():
            fps = 1000 / time_ms
            print(f"{res}x{res}: {time_ms:.1f}ms ({fps:.1f} FPS)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Jetson Performance Benchmark")
    parser.add_argument("--resolution", type=int, default=256, help="Input resolution")
    parser.add_argument("--tests", type=int, default=20, help="Number of test iterations")
    parser.add_argument("--profile", action="store_true", help="Profile different resolutions")
    
    args = parser.parse_args()
    
    print("🔧 GLASS Jetson Performance Benchmark")
    print("=" * 50)
    
    # Check if running on Jetson
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        print(f"Platform: {model}")
    except:
        print("⚠️  Not running on Jetson platform")
    
    if args.profile:
        profile_different_resolutions()
    else:
        model_path = "models/wfdd_grid_cloth"
        benchmark_inference_jetson(model_path, args.tests, args.resolution)

if __name__ == "__main__":
    main()