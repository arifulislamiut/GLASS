#!/usr/bin/env python3
"""
Video File GLASS Anomaly Detection
Run anomaly detection on video files
"""

import cv2
import numpy as np
import time
import argparse
from predict_single_image import GLASSPredictor
import os

class VideoAnomalyDetector:
    def __init__(self, model_path, model_type="onnx", device="cuda"):
        """
        Initialize video-based anomaly detector
        
        Args:
            model_path: Path to trained model
            model_type: "pytorch" or "onnx"
            device: "cuda" or "cpu"
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        
        # Initialize GLASS predictor
        print(f"Loading GLASS model from {model_path}")
        self.predictor = GLASSPredictor(model_path, model_type, device)
        
    def process_video(self, video_path, output_path=None, save_frames=False):
        """
        Process video file for anomaly detection
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            save_frames: Whether to save individual frames with anomalies
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {frame_width}x{frame_height} @ {fps:.1f} FPS ({total_frames} frames)")
        
        # Setup output video writer
        output_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
        
        # Process frames
        frame_count = 0
        anomaly_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                try:
                    # Run prediction
                    start_time = time.time()
                    results = self.predictor.predict(frame)
                    processing_time = time.time() - start_time
                    
                    # Create visualization
                    display_frame = self.create_visualization(frame, results, processing_time)
                    
                    # Check for anomalies
                    if results and results['is_anomalous']:
                        anomaly_frames.append(frame_count)
                        print(f"Anomaly detected at frame {frame_count}: score={results['image_score']:.3f}")
                        
                        # Save frame if requested
                        if save_frames:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            cv2.imwrite(f"anomaly_frame_{frame_count}_{timestamp}.jpg", frame)
                            self.predictor.visualize_results(frame, results, f"anomaly_result_{frame_count}_{timestamp}.png")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    display_frame = cv2.resize(frame, (640, 480))
                    cv2.putText(display_frame, "Processing Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Skip frame, just resize for display
                display_frame = cv2.resize(frame, (640, 480))
                cv2.putText(display_frame, "Skipped", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Write to output video
            if output_writer:
                output_writer.write(display_frame)
            
            # Show frame
            cv2.imshow('Video Anomaly Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"video_frame_{frame_count}_{timestamp}.jpg", frame)
                if results:
                    self.predictor.visualize_results(frame, results, f"video_result_{frame_count}_{timestamp}.png")
                print(f"Saved frame {frame_count}")
        
        # Cleanup
        cap.release()
        if output_writer:
            output_writer.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Anomaly frames detected: {len(anomaly_frames)}")
        if anomaly_frames:
            print(f"Anomaly frame numbers: {anomaly_frames}")
        
        return anomaly_frames
    
    def create_visualization(self, frame, results, processing_time):
        """Create visualization for display"""
        if results is None:
            return cv2.resize(frame, (640, 480))
        
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
        
        # Create 2x2 grid
        top_row = np.hstack([cv2.resize(frame, (320, 240)), anomaly_map_colored])
        bottom_row = np.hstack([cv2.resize(frame, (320, 240)), overlay])
        combined = np.vstack([top_row, bottom_row])
        
        # Add text overlay
        score = results['image_score']
        is_anomalous = results['is_anomalous']
        
        # Color based on anomaly status
        color = (0, 0, 255) if is_anomalous else (0, 255, 0)
        status = "ANOMALY DETECTED!" if is_anomalous else "Normal"
        
        # Add text
        cv2.putText(combined, f"Score: {score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(combined, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(combined, f"Processing: {processing_time*1000:.1f}ms", (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return combined

def main():
    parser = argparse.ArgumentParser(description="Video file GLASS anomaly detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--model_path", type=str, 
                       default="results/models/backbone_0/wfdd_grid_cloth",
                       help="Path to trained model directory")
    parser.add_argument("--model_type", type=str, choices=["pytorch", "onnx"], 
                       default="onnx", help="Model type to use")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                       default="cuda", help="Device to use")
    parser.add_argument("--save_frames", action="store_true", 
                       help="Save individual frames with anomalies")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} not found")
        return
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} not found")
        return
    
    try:
        # Initialize and run video detector
        detector = VideoAnomalyDetector(
            model_path=args.model_path,
            model_type=args.model_type,
            device=args.device
        )
        detector.process_video(args.video, args.output, args.save_frames)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 