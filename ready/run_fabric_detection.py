#!/usr/bin/env python3
"""
Ready-to-Deploy Fabric Defect Detection System
GLASS Anomaly Detection with Camera Interface

Usage: python run_fabric_detection.py
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import cv2
import numpy as np
import time
import argparse
from collections import deque

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        from PIL import Image
        print("✓ All dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def check_camera(camera_id=0):
    """Check camera access"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"✗ Cannot access camera {camera_id}")
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Camera {camera_id}: {width}x{height} @ {fps:.1f}FPS")
    cap.release()
    return True

def check_model():
    """Check if model files exist"""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'wfdd_grid_cloth')
    
    if not os.path.exists(model_path):
        print(f"✗ Model directory not found: {model_path}")
        return False
    
    # Check for ONNX model
    onnx_files = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
    if onnx_files:
        print(f"✓ ONNX model found: {onnx_files[0]}")
        return True, 'onnx', model_path  # Return directory path for ONNX
    
    # Check for PyTorch model
    pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    if pth_files:
        print(f"✓ PyTorch model found: {pth_files[0]}")
        return True, 'pytorch', model_path
    
    print("✗ No model files found")
    return False, None, None

class FabricDefectDetector:
    def __init__(self, model_path, model_type, device='cpu', camera_id=0):
        """Initialize fabric defect detector"""
        print(f"Loading GLASS model from {model_path}")
        print("Please wait, this may take 30-60 seconds...")
        
        # Import predictor
        from predict_single_image import GLASSPredictor
        
        self.predictor = GLASSPredictor(model_path, model_type, device)
        print("✓ Model loaded successfully!")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        
        # Settings
        self.threshold = 0.5
        self.show_overlay = True
        self.show_anomaly_map = True
        self.display_mode = 0  # 0=all, 1=original+anomaly, 2=original+overlay, 3=original only
        
        # Cache for smooth display
        self.last_results = None
        self.last_processing_time = 0
        
        print("✓ System ready!")
    
    def process_frame(self, frame):
        """Process single frame for defect detection"""
        try:
            # Convert and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (288, 288))
            
            from PIL import Image
            pil_image = Image.fromarray(frame_resized)
            
            # Run prediction
            start_time = time.time()
            results = self.predictor.predict(pil_image)
            processing_time = time.time() - start_time
            
            self.processing_times.append(processing_time)
            
            return results, processing_time
            
        except Exception as e:
            print(f"Processing error: {e}")
            return None, 0
    
    def create_visualization(self, frame, results):
        """Create clean visualization display"""
        # Standard display size
        display_width = 960
        display_height = 540
        panel_width = 480
        panel_height = 360
        
        # Always resize main frame consistently
        main_frame = cv2.resize(frame, (panel_width, panel_height))
        
        if results is None:
            # Show only main frame when no results
            display_frame = cv2.resize(main_frame, (display_width, display_height))
            # Add "Processing..." text
            cv2.putText(display_frame, "Processing...", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            return display_frame
        
        # Get and process anomaly map
        anomaly_map = results['anomaly_map']
        if len(anomaly_map.shape) > 2:
            anomaly_map = np.squeeze(anomaly_map)
        
        # Normalize anomaly map
        anomaly_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        anomaly_resized = cv2.resize(anomaly_normalized, (panel_width, panel_height))
        
        # Create colored anomaly map
        anomaly_colored = cv2.applyColorMap((anomaly_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(main_frame, 0.6, anomaly_colored, 0.4, 0)
        
        # Create layout based on display mode
        if self.display_mode == 0:  # Show all views
            # 2x2 grid layout
            top_row = np.hstack([main_frame, anomaly_colored])
            bottom_row = np.hstack([overlay, np.zeros_like(main_frame)])
            combined = np.vstack([top_row, bottom_row])
            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Anomaly Map", (panel_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Overlay", (10, panel_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif self.display_mode == 1:  # Original + Anomaly
            combined = np.hstack([main_frame, anomaly_colored])
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Anomaly Map", (panel_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif self.display_mode == 2:  # Original + Overlay
            combined = np.hstack([main_frame, overlay])
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Overlay", (panel_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:  # Original only (mode 3)
            combined = main_frame
            cv2.putText(combined, "Live Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return combined
    
    def add_info_overlay(self, frame, results, fps):
        """Add clean information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent info panel
        info_panel = frame.copy()
        
        if results:
            score = results['image_score']
            is_anomalous = results['is_anomalous']
            
            # Status color and text
            color = (0, 0, 255) if is_anomalous else (0, 255, 0)
            bg_color = (0, 0, 100) if is_anomalous else (0, 100, 0)
            status = "DEFECT DETECTED!" if is_anomalous else "FABRIC OK"
            
            # Status background
            cv2.rectangle(info_panel, (10, 50), (300, 120), bg_color, -1)
            cv2.rectangle(info_panel, (10, 50), (300, 120), color, 2)
            
            # Status text
            cv2.putText(info_panel, status, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_panel, f"Score: {score:.3f}", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No results yet
            cv2.rectangle(info_panel, (10, 50), (200, 90), (100, 100, 0), -1)
            cv2.putText(info_panel, "PROCESSING...", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Performance info background
        info_y = h - 120
        cv2.rectangle(info_panel, (10, info_y), (350, h - 10), (50, 50, 50), -1)
        
        # Performance text
        avg_time = np.mean(self.processing_times) if self.processing_times else self.last_processing_time
        
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (20, info_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Processing: {avg_time*1000:.1f}ms", (120, info_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Threshold: {self.threshold:.2f}", (20, info_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display mode info
        modes = ["All Views", "Orig+Anomaly", "Orig+Overlay", "Original Only"]
        cv2.putText(info_panel, f"Mode: {modes[self.display_mode]}", (150, info_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(info_panel, "Controls: q=quit, d=display mode, s=save", (20, info_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(info_panel, "+/- = adjust threshold", (20, info_y + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return info_panel
    
    def run(self):
        """Main detection loop"""
        print("\n" + "="*50)
        print("FABRIC DEFECT DETECTION SYSTEM ACTIVE")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'd' - Cycle display modes (All/Original+Anomaly/Original+Overlay/Original only)")
        print("  's' - Save current frame")
        print("  '+'/'-' - Adjust threshold")
        print("="*50)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frames with better continuity
            if frame_count % 2 == 0:  # Process every 2nd frame instead of 3rd
                results, processing_time = self.process_frame(frame)
                if results is not None:
                    self.last_results = results
                    self.last_processing_time = processing_time
            
            # Always use cached results for smooth display
            current_results = self.last_results
            
            # Create visualization with consistent results
            display_frame = self.create_visualization(frame, current_results)
            
            # Calculate FPS
            current_time = time.time()
            self.fps_counter.append(current_time)
            if len(self.fps_counter) > 1:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            else:
                fps = 0
            
            # Add information overlay
            display_frame = self.add_info_overlay(display_frame, current_results, fps)
            
            # Display
            cv2.imshow('Fabric Defect Detection - GLASS', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.display_mode = (self.display_mode + 1) % 4
                modes = ["All Views", "Original + Anomaly", "Original + Overlay", "Original Only"]
                print(f"Display mode: {modes[self.display_mode]}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"fabric_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif key == ord('+'):
                self.threshold = min(1.0, self.threshold + 0.05)
                print(f"Threshold: {self.threshold:.2f}")
            elif key == ord('-'):
                self.threshold = max(0.0, self.threshold - 0.05)
                print(f"Threshold: {self.threshold:.2f}")
            
            frame_count += 1
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")

def main():
    parser = argparse.ArgumentParser(description="Fabric Defect Detection System")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", 
                       help="Processing device")
    
    args = parser.parse_args()
    
    print("GLASS Fabric Defect Detection System")
    print("="*40)
    
    # System checks
    print("Performing system checks...")
    
    if not check_dependencies():
        print("Please install missing dependencies")
        return
    
    if not check_camera(args.camera):
        print("Camera check failed")
        return
    
    model_available, model_type, model_path = check_model()
    if not model_available:
        print("Model check failed")
        return
    
    # Force PyTorch mode for CUDA (ONNX Runtime has cuDNN version conflicts)
    if args.device == "cuda":
        print("Note: Using PyTorch mode for CUDA acceleration (ONNX Runtime cuDNN conflict)")
        model_type = "pytorch"
    
    print("\nAll checks passed! Initializing system...")
    
    try:
        # Initialize detector
        detector = FabricDefectDetector(
            model_path=model_path,
            model_type=model_type,
            device=args.device,
            camera_id=args.camera
        )
        
        # Run detection
        detector.run()
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()