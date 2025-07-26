#!/usr/bin/env python3
"""
Real-time GLASS Anomaly Detection from Camera Feed
Run anomaly detection on live video stream from camera
"""

import cv2
import numpy as np
import time
import argparse
from predict_single_image import GLASSPredictor
import threading
from collections import deque

class CameraAnomalyDetector:
    def __init__(self, model_path, model_type="onnx", device="cuda", camera_id=0):
        """
        Initialize camera-based anomaly detector
        
        Args:
            model_path: Path to trained model
            model_type: "pytorch" or "onnx"
            device: "cuda" or "cpu"
            camera_id: Camera device ID (usually 0 for default camera)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.camera_id = camera_id
        
        # Initialize GLASS predictor
        print(f"Loading GLASS model from {model_path}")
        self.predictor = GLASSPredictor(model_path, model_type, device)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        
        # Display settings
        self.show_anomaly_map = True
        self.show_overlay = True
        self.threshold = 0.5
        self.auto_threshold = False
        
        # Threading for smooth display
        self.latest_frame = None
        self.latest_results = None
        self.running = False
        

    
    def run_prediction(self, frame):
        """Run GLASS prediction on frame"""
        try:
            # Convert BGR to RGB and resize for preprocessing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (288, 288))
            
            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(frame_resized)
            
            # Run prediction
            start_time = time.time()
            results = self.predictor.predict(pil_image)
            processing_time = time.time() - start_time
            
            # Validate results
            if results and 'anomaly_map' in results:
                anomaly_map = results['anomaly_map']
                if not isinstance(anomaly_map, np.ndarray):
                    print(f"Warning: anomaly_map is not numpy array, type: {type(anomaly_map)}")
                    return None, processing_time
                
                # Ensure anomaly_map is 2D
                if len(anomaly_map.shape) > 2:
                    results['anomaly_map'] = np.squeeze(anomaly_map)
            
            return results, processing_time
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def create_visualization(self, frame, results):
        """Create visualization for display"""
        if results is None:
            return frame
        
        # Resize frame for display
        display_frame = cv2.resize(frame, (640, 480))
        
        # Create anomaly map visualization
        anomaly_map = results['anomaly_map']
        
        # Ensure anomaly_map is 2D and contiguous
        if len(anomaly_map.shape) > 2:
            anomaly_map = np.squeeze(anomaly_map)
        anomaly_map = np.ascontiguousarray(anomaly_map)
        
        # Resize anomaly map
        anomaly_map_resized = cv2.resize(anomaly_map, (320, 240))
        
        # Normalize and apply colormap
        anomaly_map_normalized = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)
        anomaly_map_colored = cv2.applyColorMap((anomaly_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create overlay
        frame_resized = cv2.resize(frame, (320, 240))
        overlay = cv2.addWeighted(frame_resized, 0.7, anomaly_map_colored, 0.3, 0)
        
        # Create combined display
        if self.show_anomaly_map and self.show_overlay:
            # 2x2 grid
            top_row = np.hstack([cv2.resize(frame, (320, 240)), anomaly_map_colored])
            bottom_row = np.hstack([cv2.resize(frame, (320, 240)), overlay])
            combined = np.vstack([top_row, bottom_row])
        elif self.show_anomaly_map:
            # Side by side
            combined = np.hstack([cv2.resize(frame, (320, 240)), anomaly_map_colored])
        elif self.show_overlay:
            # Side by side with overlay
            combined = np.hstack([cv2.resize(frame, (320, 240)), overlay])
        else:
            # Just original frame
            combined = cv2.resize(frame, (640, 480))
        
        return combined
    
    def add_text_overlay(self, frame, results, fps, processing_time):
        """Add text information to frame"""
        # Score and status
        if results:
            score = results['image_score']
            is_anomalous = results['is_anomalous']
            
            # Color based on anomaly status
            color = (0, 0, 255) if is_anomalous else (0, 255, 0)
            status = "ANOMALY DETECTED!" if is_anomalous else "Normal"
            
            # Add text
            cv2.putText(frame, f"Score: {score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Performance info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Processing: {processing_time*1000:.1f}ms", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Threshold: {self.threshold:.2f}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main camera loop"""
        print("Starting camera inference...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'a' - Toggle anomaly map display")
        print("  'o' - Toggle overlay display")
        print("  't' - Toggle auto threshold")
        print("  '+' - Increase threshold")
        print("  '-' - Decrease threshold")
        print("  's' - Save current frame")
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Run prediction (every few frames for performance)
            if frame_count % 3 == 0:  # Process every 3rd frame
                results, processing_time = self.run_prediction(frame)
                self.latest_results = results
                self.processing_times.append(processing_time)
            else:
                results = self.latest_results
                processing_time = 0
            
            # Create visualization
            display_frame = self.create_visualization(frame, results)
            
            # Add text overlay
            current_time = time.time()
            self.fps_counter.append(current_time)
            if len(self.fps_counter) > 1:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            else:
                fps = 0
            
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            display_frame = self.add_text_overlay(display_frame, results, fps, avg_processing_time)
            
            # Show frame
            cv2.imshow('GLASS Anomaly Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.show_anomaly_map = not self.show_anomaly_map
                print(f"Anomaly map display: {'ON' if self.show_anomaly_map else 'OFF'}")
            elif key == ord('o'):
                self.show_overlay = not self.show_overlay
                print(f"Overlay display: {'ON' if self.show_overlay else 'OFF'}")
            elif key == ord('t'):
                self.auto_threshold = not self.auto_threshold
                print(f"Auto threshold: {'ON' if self.auto_threshold else 'OFF'}")
            elif key == ord('+'):
                self.threshold = min(1.0, self.threshold + 0.05)
                print(f"Threshold: {self.threshold:.2f}")
            elif key == ord('-'):
                self.threshold = max(0.0, self.threshold - 0.05)
                print(f"Threshold: {self.threshold:.2f}")
            elif key == ord('s'):
                # Save current frame with results
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"camera_frame_{timestamp}.jpg", frame)
                if results:
                    self.predictor.visualize_results(frame, results, f"camera_result_{timestamp}.png")
                print(f"Saved frame and results: camera_frame_{timestamp}.jpg, camera_result_{timestamp}.png")
            
            frame_count += 1
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera inference stopped")

def main():
    parser = argparse.ArgumentParser(description="Real-time GLASS anomaly detection from camera")
    parser.add_argument("--model_path", type=str, 
                       default="results/models/backbone_0/wfdd_grid_cloth",
                       help="Path to trained model directory")
    parser.add_argument("--model_type", type=str, choices=["pytorch", "onnx"], 
                       default="onnx", help="Model type to use")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                       default="cuda", help="Device to use")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera device ID (default: 0)")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} not found")
        return
    
    try:
        # Initialize and run camera detector
        detector = CameraAnomalyDetector(
            model_path=args.model_path,
            model_type=args.model_type,
            device=args.device,
            camera_id=args.camera
        )
        detector.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import os
    main() 