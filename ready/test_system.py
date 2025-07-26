#!/usr/bin/env python3
"""Quick system test"""
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def test_imports():
    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        print("✓ Dependencies OK")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_camera():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera OK")
            cap.release()
            return True
        else:
            print("✗ Camera failed")
            return False
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return False

def test_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'wfdd_grid_cloth')
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        onnx_files = [f for f in files if f.endswith('.onnx')]
        if onnx_files:
            print(f"✓ Model OK: {onnx_files[0]}")
            return True
    print("✗ Model failed")
    return False

def test_predictor():
    try:
        from predict_single_image import GLASSPredictor
        print("✓ Predictor import OK")
        return True
    except Exception as e:
        print(f"✗ Predictor error: {e}")
        return False

if __name__ == "__main__":
    print("System Test")
    print("=" * 20)
    
    all_ok = True
    all_ok &= test_imports()
    all_ok &= test_camera()
    all_ok &= test_model()
    all_ok &= test_predictor()
    
    if all_ok:
        print("\n✓ All tests passed! System ready to run:")
        print("python run_fabric_detection.py")
    else:
        print("\n✗ Some tests failed")