#!/usr/bin/env python3
"""
Quick Camera Demo for GLASS Anomaly Detection
"""
import cv2
import os
import sys

def main():
    print("GLASS Camera Inference Quick Start")
    print("=" * 40)
    
    # Check available models
    model_base = "results/models/backbone_0"
    available_models = []
    
    if os.path.exists(model_base):
        for model_dir in os.listdir(model_base):
            model_path = os.path.join(model_base, model_dir)
            if os.path.isdir(model_path):
                # Check for PyTorch models
                if os.path.exists(os.path.join(model_path, "ckpt_best.pth")) or \
                   any(f.endswith("_best.pth") for f in os.listdir(model_path) if f.startswith("ckpt")):
                    available_models.append((model_dir, "pytorch"))
                # Check for ONNX models
                if any(f.endswith(".onnx") for f in os.listdir(model_path)):
                    available_models.append((model_dir, "onnx"))
    
    print(f"Available models:")
    for i, (model, type_) in enumerate(available_models):
        print(f"  {i+1}. {model} ({type_})")
    
    if not available_models:
        print("No trained models found!")
        return
    
    # Check camera
    print(f"\nTesting camera access...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera!")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera: {width}x{height} @ {fps:.1f}FPS")
    cap.release()
    
    # Recommend fabric model for fabric defect detection
    fabric_models = [(model, type_) for model, type_ in available_models 
                     if 'wfdd' in model.lower() or 'cloth' in model.lower() or 'fabric' in model.lower()]
    
    if fabric_models:
        recommended = fabric_models[0]
        print(f"\nRecommended for fabric defect detection: {recommended[0]} ({recommended[1]})")
    else:
        recommended = available_models[0]
        print(f"\nUsing available model: {recommended[0]} ({recommended[1]})")
    
    model_path = os.path.join(model_base, recommended[0])
    
    print(f"\nTo run camera inference:")
    print(f"python camera_inference.py --model_path {model_path} --model_type {recommended[1]} --device cpu --camera 0")
    
    print(f"\nControls when running:")
    print("  'q' - Quit")
    print("  'a' - Toggle anomaly map display") 
    print("  'o' - Toggle overlay display")
    print("  's' - Save current frame")
    print("  '+'/'-' - Adjust threshold")
    
    print(f"\nNote: Model loading may take 30-60 seconds initially...")
    
    # Ask if user wants to start
    response = input("\nStart camera inference now? (y/n): ")
    if response.lower() == 'y':
        print("Starting camera inference...")
        os.system(f"python camera_inference.py --model_path {model_path} --model_type {recommended[1]} --device cpu --camera 0")

if __name__ == "__main__":
    main()