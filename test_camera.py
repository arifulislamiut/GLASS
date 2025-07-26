#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os
import time

def test_camera_basic():
    """Test basic camera functionality"""
    print("Testing basic camera access...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return False
    
    print("Camera opened successfully!")
    print(f"Frame width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"Frame height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Capture a few frames
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: {frame.shape}")
            cv2.imshow('Camera Test', frame)
            cv2.waitKey(1000)  # Show for 1 second
        else:
            print(f"Failed to capture frame {i+1}")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def test_glass_camera():
    """Test GLASS with camera"""
    print("\nTesting GLASS with camera...")
    
    # Check if model exists
    model_path = "results/models/backbone_0/wfdd_grid_cloth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model path {model_path} not found")
        return False
    
    try:
        from predict_single_image import GLASSPredictor
        print("Loading GLASS predictor...")
        
        # Try PyTorch model first
        predictor = GLASSPredictor(model_path, "pytorch", "cpu")
        print("GLASS predictor loaded successfully!")
        
        # Test with camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return False
        
        print("Starting camera inference...")
        print("Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process every 10th frame to avoid overload
            if frame_count % 10 == 0:
                try:
                    # Convert and resize frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (288, 288))
                    
                    from PIL import Image
                    pil_image = Image.fromarray(frame_resized)
                    
                    # Run prediction
                    start_time = time.time()
                    results = predictor.predict(pil_image)
                    processing_time = time.time() - start_time
                    
                    if results:
                        score = results.get('image_score', 0)
                        is_anomalous = results.get('is_anomalous', False)
                        
                        # Add text to frame
                        color = (0, 0, 255) if is_anomalous else (0, 255, 0)
                        status = "ANOMALY!" if is_anomalous else "Normal"
                        
                        cv2.putText(frame, f"Score: {score:.3f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, status, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frame, f"Time: {processing_time*1000:.1f}ms", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        print(f"Frame {frame_count}: Score={score:.3f}, Status={status}, Time={processing_time*1000:.1f}ms")
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    cv2.putText(frame, "Prediction Error", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('GLASS Camera Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"camera_test_{timestamp}.jpg", frame)
                print(f"Saved frame: camera_test_{timestamp}.jpg")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("GLASS Camera Test")
    print("=================")
    
    # Test basic camera first
    if not test_camera_basic():
        print("Basic camera test failed!")
        sys.exit(1)
    
    # Test GLASS with camera
    if not test_glass_camera():
        print("GLASS camera test failed!")
        sys.exit(1)
    
    print("All tests completed successfully!")