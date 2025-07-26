#!/usr/bin/env python3
"""Test resolution change impact on inference performance"""

import os
import sys
import time
import numpy as np
from PIL import Image

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'utils'))
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

from scripts.predict_single_image import GLASSPredictor

def test_resolution_performance():
    """Test inference performance at different resolutions"""
    print("🧪 Testing Resolution Impact on Inference Performance")
    print("=" * 60)
    
    model_path = "models/wfdd_grid_cloth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Initialize predictor with default resolution
    print("Loading model...")
    predictor = GLASSPredictor(model_path, "pytorch", "cuda")
    print("✅ Model loaded")
    
    # Test resolutions
    test_resolutions = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
    
    # Create test image
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    pil_img = Image.fromarray(test_img)
    
    print("\nTesting inference performance at different resolutions:")
    print("Resolution | Avg Time (ms) | Min Time (ms) | Max Time (ms)")
    print("-" * 60)
    
    results = []
    
    for resolution in test_resolutions:
        # Update predictor resolution
        predictor.set_image_size(resolution)
        
        # Warmup
        for _ in range(3):
            predictor.predict(pil_img)
        
        # Benchmark
        times = []
        for i in range(10):
            start_time = time.time()
            result = predictor.predict(pil_img)
            inference_time = time.time() - start_time
            times.append(inference_time * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        results.append({
            'resolution': resolution,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time
        })
        
        print(f"{resolution:>4}x{resolution:>3}   | {avg_time:>9.1f}   | {min_time:>9.1f}   | {max_time:>9.1f}")
    
    print("\n" + "=" * 60)
    
    # Analysis
    baseline = results[2]  # 288x288
    print(f"Baseline (288x288): {baseline['avg_time']:.1f}ms")
    print("\nRelative Performance:")
    
    for result in results:
        relative = result['avg_time'] / baseline['avg_time']
        status = "🟢" if relative < 1.2 else "🟡" if relative < 2.0 else "🔴"
        print(f"{status} {result['resolution']:>3}x{result['resolution']:>3}: {relative:.2f}x slower")
    
    print(f"\n✅ Resolution change test completed!")
    print(f"   Fastest: {min(results, key=lambda x: x['avg_time'])['resolution']}x{min(results, key=lambda x: x['avg_time'])['resolution']} ({min(results, key=lambda x: x['avg_time'])['avg_time']:.1f}ms)")
    print(f"   Slowest: {max(results, key=lambda x: x['avg_time'])['resolution']}x{max(results, key=lambda x: x['avg_time'])['resolution']} ({max(results, key=lambda x: x['avg_time'])['avg_time']:.1f}ms)")

if __name__ == "__main__":
    test_resolution_performance()