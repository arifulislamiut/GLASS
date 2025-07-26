#!/usr/bin/env python3
"""
GLASS Model Inference Script
Run anomaly detection on single images using the trained GLASS model.

Usage:
    python predict_single_image.py --image path/to/your/image.jpg --model_type pytorch
    python predict_single_image.py --image path/to/your/image.jpg --model_type onnx
"""

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import GLASS components
import backbones
import glass
import utils

# ONNX runtime for faster inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime-gpu")


class GLASSPredictor:
    def __init__(self, model_path, model_type="pytorch", device="cuda", image_size=288):
        """
        Initialize GLASS predictor
        
        Args:
            model_path: Path to the trained model checkpoint
            model_type: "pytorch" or "onnx"
            device: "cuda" or "cpu"
            image_size: Input image size (default: 288)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        
        # Image preprocessing parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.image_size = image_size
        
        if model_type == "onnx" and ONNX_AVAILABLE:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
    
    def _load_onnx_model(self):
        """Load ONNX model for faster inference"""
        print(f"Loading ONNX model from {self.model_path}")
        onnx_path = os.path.join(self.model_path, "glass_simplified.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print("ONNX model loaded successfully")
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        print(f"Loading PyTorch model from {self.model_path}")
        
        # Load backbone
        backbone = backbones.load("wideresnet50")
        backbone.name = "wideresnet50"
        
        # Initialize GLASS
        self.glass_model = glass.GLASS(self.device)
        self.glass_model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=self.device,
            input_shape=(3, self.image_size, self.image_size),
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
        
        # Load trained weights
        ckpt_path = os.path.join(self.model_path, "ckpt_best_571.pth")
        if not os.path.exists(ckpt_path):
            # Try to find any checkpoint
            ckpt_files = [f for f in os.listdir(self.model_path) if f.startswith("ckpt_best_")]
            if ckpt_files:
                ckpt_path = os.path.join(self.model_path, ckpt_files[0])
            else:
                raise FileNotFoundError(f"No checkpoint found in {self.model_path}")
        
        state_dict = torch.load(ckpt_path, map_location=self.device)
        if 'discriminator' in state_dict:
            self.glass_model.discriminator.load_state_dict(state_dict['discriminator'])
            if "pre_projection" in state_dict:
                self.glass_model.pre_projection.load_state_dict(state_dict["pre_projection"])
        else:
            self.glass_model.load_state_dict(state_dict, strict=False)
        
        print("PyTorch model loaded successfully")
    
    def set_image_size(self, new_size):
        """Update the image size for dynamic resolution changes"""
        self.image_size = new_size
        print(f"🔍 Model image size updated to {new_size}x{new_size}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load and resize image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_path)
        else:
            image = image_path
        
        # Resize to model input size
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        image = (image - self.mean) / self.std
        image = image.transpose((2, 0, 1))  # HWC to CHW
        
        return image
    
    def predict_onnx(self, image):
        """Predict using ONNX model"""
        # Prepare input
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        image_batch = np.repeat(image_batch, 8, axis=0)  # Batch size 8 as in training
        
        # Run inference
        output = self.session.run(None, {"input": image_batch})[0]
        
        # Process output
        output = np.expand_dims(output, axis=1)[0]  # Take first image
        output = output.transpose((1, 2, 0))
        output = cv2.resize(output, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        output = cv2.GaussianBlur(output, (33, 33), 4)
        
        # Ensure output is properly formatted
        if len(output.shape) > 2:
            output = np.squeeze(output)
        output = np.ascontiguousarray(output)
        
        return output
    
    def predict_pytorch(self, image):
        """Predict using PyTorch model"""
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            scores, masks = self.glass_model._predict(image_tensor)
        
        # Convert to numpy
        mask = masks[0]  # Take first image
        score = scores[0]  # Take first image
        
        # Convert to numpy if they're tensors
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()
        if hasattr(score, 'numpy'):
            score = score.numpy()
        
        return mask, score
    
    def predict(self, image_path):
        """Main prediction function"""
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Run prediction
        if self.model_type == "onnx" and ONNX_AVAILABLE:
            anomaly_map = self.predict_onnx(image)
            # For ONNX, we don't have direct access to image-level score
            # We can compute it from the anomaly map
            image_score = np.max(anomaly_map)
        else:
            anomaly_map, image_score = self.predict_pytorch(image)
            # Ensure anomaly_map is numpy array
            if hasattr(anomaly_map, 'numpy'):
                anomaly_map = anomaly_map.numpy()
        
        # Ensure anomaly_map is properly formatted
        if isinstance(anomaly_map, np.ndarray):
            # Make sure it's 2D and contiguous
            if len(anomaly_map.shape) > 2:
                anomaly_map = np.squeeze(anomaly_map)
            anomaly_map = np.ascontiguousarray(anomaly_map)
        else:
            print(f"Warning: anomaly_map is not numpy array, type: {type(anomaly_map)}")
            # Create a dummy anomaly map if prediction failed
            anomaly_map = np.zeros((288, 288), dtype=np.float32)
            image_score = 0.0
        
        return {
            'anomaly_map': anomaly_map,
            'image_score': image_score,
            'is_anomalous': image_score > 0.5  # Threshold can be adjusted
        }
    
    def visualize_results(self, image_path, results, output_path=None):
        """Visualize prediction results"""
        # Load original image
        if isinstance(image_path, str):
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = np.array(image_path)
        
        # Resize original image to match anomaly map
        original_image = cv2.resize(original_image, (self.image_size, self.image_size))
        
        # Create anomaly map visualization
        anomaly_map = results['anomaly_map']
        
        # Ensure anomaly_map is properly formatted
        if not isinstance(anomaly_map, np.ndarray):
            print(f"Warning: anomaly_map is not numpy array, type: {type(anomaly_map)}")
            anomaly_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Ensure it's 2D and contiguous
        if len(anomaly_map.shape) > 2:
            anomaly_map = np.squeeze(anomaly_map)
        anomaly_map = np.ascontiguousarray(anomaly_map)
        
        # Normalize
        anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        
        # Apply colormap
        anomaly_map_colored = cv2.applyColorMap((anomaly_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        anomaly_map_colored = cv2.cvtColor(anomaly_map_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(original_image, 0.7, anomaly_map_colored, 0.3, 0)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Anomaly map
        axes[0, 1].imshow(anomaly_map, cmap='hot')
        axes[0, 1].set_title('Anomaly Map')
        axes[0, 1].axis('off')
        
        # Colored anomaly map
        axes[1, 0].imshow(anomaly_map_colored)
        axes[1, 0].set_title('Anomaly Map (Colored)')
        axes[1, 0].axis('off')
        
        # Overlay
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title(f'Overlay (Score: {results["image_score"]:.3f})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to {output_path}")
        else:
            plt.show()
        
        return fig


def main():
    parser = argparse.ArgumentParser(description="Run GLASS model inference on single images")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, 
                       default="results/models/backbone_0/wfdd_grid_cloth",
                       help="Path to trained model directory")
    parser.add_argument("--model_type", type=str, choices=["pytorch", "onnx"], 
                       default="onnx", help="Model type to use")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                       default="cuda", help="Device to use")
    parser.add_argument("--output", type=str, help="Output path for visualization")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Anomaly threshold")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found")
        return
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} not found")
        return
    
    # Initialize predictor
    try:
        predictor = GLASSPredictor(args.model_path, args.model_type, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run prediction
    print(f"Running prediction on {args.image}")
    results = predictor.predict(args.image)
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"Image Score: {results['image_score']:.4f}")
    print(f"Is Anomalous: {results['is_anomalous']}")
    print(f"Anomaly Map Shape: {results['anomaly_map'].shape}")
    
    # Visualize results
    if args.output:
        output_path = args.output
    else:
        # Create default output path
        input_name = Path(args.image).stem
        output_path = f"prediction_{input_name}.png"
    
    predictor.visualize_results(args.image, results, output_path)


if __name__ == "__main__":
    main() 