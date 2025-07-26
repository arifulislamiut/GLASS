#!/usr/bin/env python3
"""
Convert trained keyboard GLASS model to ONNX format
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from glass import GLASS
from backbones import load as load_backbone

class KeyboardONNXConverter:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize ONNX converter for keyboard model
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run conversion on
        """
        self.device = device
        self.model_path = model_path
        self.input_shape = (3, 288, 288)  # Standard input size
        
        # Load model
        self.model = self._load_model()
        
        print(f"Model loaded from: {model_path}")
        print(f"Device: {device}")
    
    def _load_model(self):
        """Load the trained GLASS model"""
        # Load backbone
        backbone = load_backbone("wideresnet50")
        backbone.name = "wideresnet50"
        
        # Initialize GLASS model
        model = GLASS(self.device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=self.device,
            input_shape=self.input_shape,
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
    
    def create_onnx_model(self, output_path):
        """
        Create ONNX model from the trained GLASS model
        
        Args:
            output_path: Path to save the ONNX model
        """
        print("Creating ONNX model...")
        
        # Create dummy input
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)
        
        # Create a wrapper class for ONNX export
        class GLASSWrapper(torch.nn.Module):
            def __init__(self, glass_model):
                super(GLASSWrapper, self).__init__()
                self.glass_model = glass_model
            
            def forward(self, x):
                # Run the embedding and prediction pipeline manually to avoid numpy conversion
                self.glass_model.forward_modules.eval()
                
                if self.glass_model.pre_proj > 0:
                    self.glass_model.pre_projection.eval()
                self.glass_model.discriminator.eval()
                
                # Get patch features
                patch_features, patch_shapes = self.glass_model._embed(x, provide_patch_shapes=True, evaluation=True)
                
                # Apply pre-projection if available
                if self.glass_model.pre_proj > 0:
                    patch_features = self.glass_model.pre_projection(patch_features)
                    patch_features = patch_features[0] if len(patch_features) == 2 else patch_features
                
                # Get discriminator scores
                patch_scores = self.glass_model.discriminator(patch_features)
                
                # Unpatch scores
                patch_scores_unpatched = self.glass_model.patch_maker.unpatch_scores(patch_scores, batchsize=x.shape[0])
                scales = patch_shapes[0]
                patch_scores_reshaped = patch_scores_unpatched.reshape(x.shape[0], scales[0], scales[1])
                
                # Convert to segmentation
                masks = self.glass_model.anomaly_segmentor.convert_to_segmentation(patch_scores_reshaped)
                # Ensure outputs are tensors
                if isinstance(masks, list):
                    masks = [torch.from_numpy(m).to(x.device, dtype=torch.float32) if isinstance(m, np.ndarray) else m for m in masks]
                    masks = torch.stack(masks) if len(masks) > 0 else torch.zeros(1, 288, 288, device=x.device, dtype=torch.float32)
                
                # Get image scores
                image_scores_unpatched = self.glass_model.patch_maker.unpatch_scores(patch_scores, batchsize=x.shape[0])
                image_scores = self.glass_model.patch_maker.score(image_scores_unpatched)
                
                # Ensure outputs are tensors
                if not isinstance(image_scores, torch.Tensor):
                    image_scores = torch.tensor(image_scores, device=x.device, dtype=torch.float32)
                
                return image_scores, masks
        
        # Create wrapper model
        wrapper_model = GLASSWrapper(self.model)
        wrapper_model.eval()
        
        # Export to ONNX
        print(f"Exporting to ONNX: {output_path}")
        
        try:
            torch.onnx.export(
                wrapper_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['scores', 'masks'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'scores': {0: 'batch_size'},
                    'masks': {0: 'batch_size'}
                },
                verbose=False
            )
            print(f"✅ ONNX model saved to: {output_path}")
            
            # Verify the ONNX model
            self._verify_onnx_model(output_path, dummy_input)
            
        except Exception as e:
            print(f"❌ Error exporting to ONNX: {e}")
            raise
    
    def _verify_onnx_model(self, onnx_path, dummy_input):
        """Verify the ONNX model works correctly"""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model is valid")
            
            # Test inference with ONNX Runtime
            ort_session = ort.InferenceSession(onnx_path)
            
            # Prepare input
            input_data = dummy_input.cpu().numpy()
            
            # Run inference
            outputs = ort_session.run(None, {'input': input_data})
            scores, masks = outputs
            
            print(f"✅ ONNX inference successful")
            print(f"   Scores shape: {scores.shape}")
            print(f"   Masks shape: {masks.shape}")
            print(f"   Sample score: {scores[0]:.4f}")
            
        except ImportError:
            print("⚠️  ONNX or ONNX Runtime not available for verification")
        except Exception as e:
            print(f"⚠️  ONNX verification failed: {e}")
    
    def create_simplified_onnx(self, output_path):
        """
        Create a simplified ONNX model that only outputs the anomaly map
        This is useful for deployment scenarios where you only need the segmentation
        """
        print("Creating simplified ONNX model...")
        
        # Create dummy input
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)
        
        # Create a simplified wrapper
        class SimplifiedGLASSWrapper(torch.nn.Module):
            def __init__(self, glass_model):
                super(SimplifiedGLASSWrapper, self).__init__()
                self.glass_model = glass_model
            
            def forward(self, x):
                # Run the embedding and prediction pipeline manually
                self.glass_model.forward_modules.eval()
                
                if self.glass_model.pre_proj > 0:
                    self.glass_model.pre_projection.eval()
                self.glass_model.discriminator.eval()
                
                # Get patch features
                patch_features, patch_shapes = self.glass_model._embed(x, provide_patch_shapes=True, evaluation=True)
                
                # Apply pre-projection if available
                if self.glass_model.pre_proj > 0:
                    patch_features = self.glass_model.pre_projection(patch_features)
                    patch_features = patch_features[0] if len(patch_features) == 2 else patch_features
                
                # Get discriminator scores
                patch_scores = self.glass_model.discriminator(patch_features)
                
                # Unpatch scores
                patch_scores_unpatched = self.glass_model.patch_maker.unpatch_scores(patch_scores, batchsize=x.shape[0])
                scales = patch_shapes[0]
                patch_scores_reshaped = patch_scores_unpatched.reshape(x.shape[0], scales[0], scales[1])
                
                # Convert to segmentation
                masks = self.glass_model.anomaly_segmentor.convert_to_segmentation(patch_scores_reshaped)
                if isinstance(masks, list):
                    masks = [torch.from_numpy(m).to(x.device, dtype=torch.float32) if isinstance(m, np.ndarray) else m for m in masks]
                    masks = torch.stack(masks) if len(masks) > 0 else torch.zeros(1, 288, 288, device=x.device, dtype=torch.float32)
                return masks
        
        # Create wrapper model
        wrapper_model = SimplifiedGLASSWrapper(self.model)
        wrapper_model.eval()
        
        # Export to ONNX
        simplified_path = output_path.replace('.onnx', '_simplified.onnx')
        print(f"Exporting simplified model to: {simplified_path}")
        
        try:
            torch.onnx.export(
                wrapper_model,
                dummy_input,
                simplified_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['anomaly_map'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'anomaly_map': {0: 'batch_size'}
                },
                verbose=False
            )
            print(f"✅ Simplified ONNX model saved to: {simplified_path}")
            
        except Exception as e:
            print(f"❌ Error exporting simplified ONNX: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Convert Keyboard GLASS Model to ONNX')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for ONNX model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--simplified', action='store_true',
                       help='Also create simplified ONNX model')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Set output path
    if args.output is None:
        model_dir = os.path.dirname(args.model)
        model_name = Path(args.model).stem
        args.output = os.path.join(model_dir, f"{model_name}.onnx")
    
    # Create converter
    converter = KeyboardONNXConverter(args.model, device=device)
    
    # Convert to ONNX
    converter.create_onnx_model(args.output)
    
    # Create simplified version if requested
    if args.simplified:
        converter.create_simplified_onnx(args.output)
    
    print("\n🎉 ONNX conversion completed!")
    print(f"Model saved to: {args.output}")
    if args.simplified:
        simplified_path = args.output.replace('.onnx', '_simplified.onnx')
        print(f"Simplified model saved to: {simplified_path}")

if __name__ == '__main__':
    main() 