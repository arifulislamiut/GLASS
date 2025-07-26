#!/usr/bin/env python3
"""
Keyboard Anomaly Detection Inference Script
Uses the trained GLASS model to detect anomalies in keyboard images
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from glass import GLASS
from datasets.keyboard import KeyboardDataset
from torchvision import transforms

class KeyboardInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize keyboard inference with trained model
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model()
        
        # No need for separate inference wrapper, use model directly
        
        # Get transforms
        self.transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded from: {model_path}")
        print(f"Device: {device}")
    
    def _load_model(self):
        """Load the trained GLASS model"""
        # Load backbone
        from backbones import load as load_backbone
        backbone = load_backbone("wideresnet50")
        backbone.name = "wideresnet50"
        
        # Initialize GLASS model
        model = GLASS(self.device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=self.device,
            input_shape=(3, 288, 288),  # Standard input size
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'discriminator' in checkpoint:
            model.discriminator.load_state_dict(checkpoint['discriminator'])
            if "pre_projection" in checkpoint:
                model.pre_projection.load_state_dict(checkpoint["pre_projection"])
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_image(self, image_path, save_result=True, output_dir='keyboard_results'):
        """
        Predict anomaly for a single image
        
        Args:
            image_path: Path to input image
            save_result: Whether to save the result
            output_dir: Directory to save results
            
        Returns:
            dict: Prediction results with scores and visualization
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            scores, masks = self.model._predict(input_tensor)
        
        # Get scores (take first image from batch)
        image_score = scores[0] if isinstance(scores, list) else scores[0].item()
        anomaly_map = masks[0] if isinstance(masks, list) else masks[0]
        
        # Convert to numpy if needed
        if hasattr(anomaly_map, 'cpu'):
            anomaly_map = anomaly_map.cpu().numpy()
        if hasattr(image_score, 'cpu'):
            image_score = image_score.cpu().numpy()
        
        # Resize anomaly map to original image size
        anomaly_map_resized = cv2.resize(
            anomaly_map, 
            (original_size[0], original_size[1]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(
            (anomaly_map_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Convert original image to numpy for overlay
        original_np = np.array(image)
        original_np = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(original_np, 0.7, heatmap, 0.3, 0)
        
        # Add text with score
        text = f"Anomaly Score: {image_score:.4f}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        
        # Save results if requested
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save original image
            original_path = os.path.join(output_dir, f"original_{Path(image_path).stem}.png")
            cv2.imwrite(original_path, original_np)
            
            # Save anomaly map
            anomaly_path = os.path.join(output_dir, f"anomaly_map_{Path(image_path).stem}.png")
            cv2.imwrite(anomaly_path, (anomaly_map_resized * 255).astype(np.uint8))
            
            # Save overlay
            overlay_path = os.path.join(output_dir, f"overlay_{Path(image_path).stem}.png")
            cv2.imwrite(overlay_path, overlay)
            
            print(f"Results saved to {output_dir}/")
        
        return {
            'image_score': image_score,
            'anomaly_map': anomaly_map_resized,
            'overlay': overlay,
            'is_anomaly': image_score > 0.5  # Threshold can be adjusted
        }
    
    def predict_batch(self, image_dir, output_dir='keyboard_results'):
        """
        Predict anomalies for all images in a directory
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        results = []
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            try:
                result = self.predict_image(str(image_path), save_result=True, output_dir=output_dir)
                results.append({
                    'image': image_path.name,
                    'score': result['image_score'],
                    'is_anomaly': result['is_anomaly']
                })
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
        
        # Print summary
        print("\n" + "="*50)
        print("BATCH PREDICTION SUMMARY")
        print("="*50)
        for result in results:
            status = "ANOMALY" if result['is_anomaly'] else "NORMAL"
            print(f"{result['image']}: {result['score']:.4f} ({status})")
        
        anomalies = [r for r in results if r['is_anomaly']]
        print(f"\nTotal images: {len(results)}")
        print(f"Anomalies detected: {len(anomalies)}")
        print(f"Anomaly rate: {len(anomalies)/len(results)*100:.1f}%")
    
    def real_time_inference(self, camera_id=0, threshold=0.5):
        """
        Real-time inference using webcam
        
        Args:
            camera_id: Camera device ID
            threshold: Anomaly detection threshold
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("Real-time keyboard anomaly detection started")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Transform image
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                scores, masks = self.model._predict(input_tensor)
            
            # Get scores (take first image from batch)
            image_score = scores[0] if isinstance(scores, list) else scores[0].item()
            anomaly_map = masks[0] if isinstance(masks, list) else masks[0]
            
            # Convert to numpy if needed
            if hasattr(anomaly_map, 'cpu'):
                anomaly_map = anomaly_map.cpu().numpy()
            if hasattr(image_score, 'cpu'):
                image_score = image_score.cpu().numpy()
            
            # Resize anomaly map to frame size
            h, w = frame.shape[:2]
            anomaly_map_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Create heatmap
            heatmap = cv2.applyColorMap(
                (anomaly_map_resized * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            
            # Overlay heatmap
            overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
            
            # Add text
            status = "ANOMALY" if image_score > threshold else "NORMAL"
            color = (0, 0, 255) if image_score > threshold else (0, 255, 0)
            
            cv2.putText(overlay, f"Score: {image_score:.4f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(overlay, f"Status: {status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                start_time = time.time()
            
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Keyboard Anomaly Detection', overlay)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f"keyboard_capture_{timestamp}.png", frame)
                cv2.imwrite(f"keyboard_overlay_{timestamp}.png", overlay)
                print(f"Saved frame with score {image_score:.4f}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Keyboard Anomaly Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input image or directory')
    parser.add_argument('--camera', action='store_true',
                       help='Run real-time camera inference')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--output', type=str, default='keyboard_results',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Anomaly detection threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize inference
    try:
        inference = KeyboardInference(args.model, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run inference based on input type
    if args.camera:
        inference.real_time_inference(args.camera_id, args.threshold)
    elif args.input:
        if os.path.isfile(args.input):
            # Single image
            result = inference.predict_image(args.input, save_result=True, output_dir=args.output)
            print(f"Image Score: {result['image_score']:.4f}")
            print(f"Anomaly: {result['is_anomaly']}")
        elif os.path.isdir(args.input):
            # Directory of images
            inference.predict_batch(args.input, args.output)
        else:
            print(f"Error: {args.input} is not a valid file or directory")
    else:
        print("Please specify --input for image/directory or --camera for real-time inference")

if __name__ == '__main__':
    main() 