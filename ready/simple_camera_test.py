#!/usr/bin/env python3
"""
Simple Camera Test - Minimal camera preview to isolate issues
"""
import cv2
import time

def simple_camera_test():
    print("🔍 Simple Camera Test")
    print("Press 'q' to quit")
    
    # Try different methods in order
    methods = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY, "Default"),
        (0, "Index only")
    ]
    
    for method, name in methods:
        print(f"\nTrying {name}...")
        
        try:
            if isinstance(method, int):
                cap = cv2.VideoCapture(method)
            else:
                cap = cv2.VideoCapture(0, method)
            
            if not cap.isOpened():
                print(f"❌ {name}: Could not open camera")
                continue
            
            # Set basic properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ {name}: Camera opened")
            
            # Test frame capture
            ret, frame = cap.read()
            if not ret:
                print(f"❌ {name}: Could not read frame")
                cap.release()
                continue
            
            print(f"✅ {name}: Frame captured {frame.shape}")
            
            # Show preview
            window_name = f"Camera Test - {name}"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Add simple overlay
                cv2.putText(frame, f"Method: {name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"✅ {name}: Showed {frame_count} frames at {fps:.1f} FPS")
            print(f"✅ {name}: Camera preview working!")
            return True
            
        except Exception as e:
            print(f"❌ {name}: Exception - {e}")
            try:
                cap.release()
            except:
                pass
    
    print("❌ All camera methods failed")
    return False

if __name__ == "__main__":
    if simple_camera_test():
        print("\n🎉 Camera is working! The issue might be in the main application.")
        print("Check if the main app is using the correct camera backend.")
    else:
        print("\n❌ Camera is not working. Check hardware and permissions.")