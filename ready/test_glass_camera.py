#!/usr/bin/env python3
"""
Test GLASS with camera preview first, then model loading
"""
import sys
import os
import time

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import cv2
import numpy as np

def test_camera_first():
    """Test camera before loading heavy models"""
    print("🔍 Testing camera before loading GLASS model...")
    
    # Try V4L2 backend (which works from our test)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Fallback
        
        if not cap.isOpened():
            print("❌ Camera failed to open")
            return False
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test frame capture
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read frame from camera")
            cap.release()
            return False
        
        print(f"✅ Camera working: {frame.shape}")
        
        # Show preview for 3 seconds
        cv2.namedWindow('Camera Test', cv2.WINDOW_AUTOSIZE)
        
        start_time = time.time()
        frame_count = 0
        
        print("📹 Showing 3-second camera preview...")
        
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Add overlay
            cv2.putText(frame, "Camera Test - Loading GLASS...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Camera preview successful: {frame_count} frames")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("🤖 Testing GLASS model loading...")
    
    try:
        # Check if model exists
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'wfdd_grid_cloth')
        if not os.path.exists(model_path):
            print(f"❌ Model directory not found: {model_path}")
            return False
        
        print(f"✅ Model directory found: {model_path}")
        
        # Try to import predictor
        from predict_single_image import GLASSPredictor
        print("✅ GLASSPredictor imported successfully")
        
        # Try to load model (this might take time)
        print("⏳ Loading GLASS model (this may take 30-60 seconds)...")
        start_time = time.time()
        
        predictor = GLASSPredictor(model_path, "pytorch", "cuda")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        # Test prediction with dummy image
        dummy_img = np.random.randint(0, 255, (288, 288, 3), dtype=np.uint8)
        from PIL import Image
        pil_img = Image.fromarray(dummy_img)
        
        print("🧪 Testing inference with dummy image...")
        start_time = time.time()
        result = predictor.predict(pil_img)
        inference_time = time.time() - start_time
        
        print(f"✅ Inference successful in {inference_time*1000:.1f}ms")
        print(f"Result keys: {list(result.keys()) if result else 'None'}")
        
        return True, predictor
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def run_glass_with_camera(predictor):
    """Run GLASS with working camera and model"""
    print("🚀 Starting GLASS with camera preview...")
    
    try:
        # Open camera (we know this works)
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        cv2.namedWindow('GLASS Fabric Detection', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        
        print("Controls: 'q' = quit, 's' = save frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Lost camera connection")
                break
            
            frame_count += 1
            
            # Process every 3rd frame to save resources
            if frame_count % 3 == 0:
                try:
                    # Resize for processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (288, 288))
                    
                    from PIL import Image
                    pil_image = Image.fromarray(frame_resized)
                    
                    # Run prediction
                    start_time = time.time()
                    results = predictor.predict(pil_image)
                    processing_time = time.time() - start_time
                    
                    # Add results to display
                    if results:
                        score = results.get('image_score', 0)
                        is_anomalous = results.get('is_anomalous', False)
                        
                        color = (0, 0, 255) if is_anomalous else (0, 255, 0)
                        status = "DEFECT!" if is_anomalous else "OK"
                        
                        cv2.putText(frame, f"Status: {status}", (10, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(frame, f"Score: {score:.3f}", (10, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.putText(frame, f"Time: {processing_time*1000:.1f}ms", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    cv2.putText(frame, "Processing Error", (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Always show frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (frame.shape[1]-200, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('GLASS Fabric Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"glass_test_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ GLASS test completed successfully")
        
    except Exception as e:
        print(f"❌ GLASS runtime error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔍 GLASS Camera + Model Test")
    print("=" * 50)
    
    # Step 1: Test camera first
    if not test_camera_first():
        print("❌ Camera test failed. Fix camera issues first.")
        return
    
    # Step 2: Test model loading
    model_success, predictor = test_model_loading()
    if not model_success:
        print("❌ Model loading failed. Check model files.")
        return
    
    # Step 3: Run combined system
    print("\n🎉 Both camera and model working! Starting combined system...")
    run_glass_with_camera(predictor)

if __name__ == "__main__":
    main()