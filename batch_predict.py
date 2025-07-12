#!/usr/bin/env python3
"""
Batch prediction script for GLASS model
Process multiple images and save results
"""

import argparse
import os
import glob
from pathlib import Path
import pandas as pd
from predict_single_image import GLASSPredictor
import cv2
import numpy as np

def process_batch(input_dir, output_dir, model_path, model_type="onnx", device="cuda"):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        model_path: Path to trained model
        model_type: "pytorch" or "onnx"
        device: "cuda" or "cpu"
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize predictor
    print(f"Loading model from {model_path}")
    predictor = GLASSPredictor(model_path, model_type, device)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Run prediction
            prediction_results = predictor.predict(image_path)
            
            # Save visualization
            image_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{image_name}_prediction.png")
            predictor.visualize_results(image_path, prediction_results, output_path)
            
            # Save anomaly map
            anomaly_map_path = os.path.join(output_dir, f"{image_name}_anomaly_map.png")
            anomaly_map = prediction_results['anomaly_map']
            anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            anomaly_map_colored = cv2.applyColorMap((anomaly_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(anomaly_map_path, anomaly_map_colored)
            
            # Store results
            results.append({
                'image_name': os.path.basename(image_path),
                'image_score': prediction_results['image_score'],
                'is_anomalous': prediction_results['is_anomalous'],
                'anomaly_map_path': anomaly_map_path,
                'prediction_path': output_path
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_name': os.path.basename(image_path),
                'image_score': -1,
                'is_anomalous': False,
                'anomaly_map_path': '',
                'prediction_path': '',
                'error': str(e)
            })
    
    # Save summary CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "prediction_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {output_dir}")
    print(f"Summary saved to {csv_path}")
    
    # Print summary
    successful_predictions = df[df['image_score'] != -1]
    if len(successful_predictions) > 0:
        print(f"\nSummary:")
        print(f"Total images: {len(df)}")
        print(f"Successfully processed: {len(successful_predictions)}")
        print(f"Anomalous images: {len(successful_predictions[successful_predictions['is_anomalous']])}")
        print(f"Average score: {successful_predictions['image_score'].mean():.4f}")
        print(f"Max score: {successful_predictions['image_score'].max():.4f}")
        print(f"Min score: {successful_predictions['image_score'].min():.4f}")

def main():
    parser = argparse.ArgumentParser(description="Batch process images with GLASS model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_path", type=str, 
                       default="results/models/backbone_0/wfdd_grid_cloth",
                       help="Path to trained model directory")
    parser.add_argument("--model_type", type=str, choices=["pytorch", "onnx"], 
                       default="onnx", help="Model type to use")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                       default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found")
        return
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} not found")
        return
    
    # Process batch
    process_batch(args.input_dir, args.output_dir, args.model_path, args.model_type, args.device)

if __name__ == "__main__":
    main() 