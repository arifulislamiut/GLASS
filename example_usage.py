#!/usr/bin/env python3
"""
Example usage of GLASS model for anomaly detection
This script demonstrates how to use the GLASS predictor on your own images.
"""

import os
from predict_single_image import GLASSPredictor

def main():
    # Example 1: Using ONNX model (faster)
    print("=== Example 1: Using ONNX Model ===")
    
    # Initialize predictor with ONNX model
    model_path = "results/models/backbone_0/wfdd_grid_cloth"
    
    try:
        predictor = GLASSPredictor(
            model_path=model_path,
            model_type="onnx",  # Use ONNX for faster inference
            device="cuda"       # Use GPU if available
        )
        
        # Example image path (replace with your image)
        image_path = "your_image.jpg"  # Replace with your image path
        
        if os.path.exists(image_path):
            # Run prediction
            results = predictor.predict(image_path)
            
            # Print results
            print(f"Image Score: {results['image_score']:.4f}")
            print(f"Is Anomalous: {results['is_anomalous']}")
            
            # Save visualization
            output_path = "prediction_result.png"
            predictor.visualize_results(image_path, results, output_path)
        else:
            print(f"Image {image_path} not found. Please provide a valid image path.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the trained model in the specified path.")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Using PyTorch model (more flexible)
    print("=== Example 2: Using PyTorch Model ===")
    
    try:
        predictor_pytorch = GLASSPredictor(
            model_path=model_path,
            model_type="pytorch",  # Use PyTorch for more flexibility
            device="cuda"
        )
        
        # You can also use a PIL Image object
        from PIL import Image
        import numpy as np
        
        # Create a dummy image for demonstration
        dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        
        # Run prediction
        results = predictor_pytorch.predict(dummy_image)
        
        print(f"Image Score: {results['image_score']:.4f}")
        print(f"Is Anomalous: {results['is_anomalous']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 