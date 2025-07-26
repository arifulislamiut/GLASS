#!/usr/bin/env python3
"""
Camera Diagnostic Tool for GLASS Fabric Detection
Tests various camera access methods and configurations
"""
import cv2
import numpy as np
import time
import os
import platform
import subprocess

def print_header(title):
    print(f"\n{'='*60}")
    print(f"📹 {title}")
    print('='*60)

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m[INFO]\033[0m",
        "SUCCESS": "\033[92m[SUCCESS]\033[0m", 
        "WARNING": "\033[93m[WARNING]\033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m"
    }
    print(f"{colors.get(status, '[INFO]')} {message}")

def check_camera_devices():
    """Check available camera devices"""
    print_header("Camera Device Detection")
    
    system = platform.system()
    
    if system == "Linux":
        # Check /dev/video* devices
        try:
            video_devices = []
            for i in range(10):  # Check video0 to video9
                device = f"/dev/video{i}"
                if os.path.exists(device):
                    video_devices.append(i)
            
            if video_devices:
                print_status(f"Found video devices: {video_devices}", "SUCCESS")
                
                # Get detailed info with v4l2-ctl if available
                try:
                    result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print("Device details:")
                        print(result.stdout)
                except:
                    print_status("v4l2-ctl not available for detailed info", "WARNING")
                    
            else:
                print_status("No /dev/video* devices found", "ERROR")
                
        except Exception as e:
            print_status(f"Device check failed: {e}", "ERROR")
    
    elif system == "Darwin":  # macOS
        print_status("macOS detected - cameras should be accessible via index", "INFO")
        
    elif system == "Windows":
        print_status("Windows detected - using DirectShow detection", "INFO")
    
    return video_devices if system == "Linux" else [0, 1, 2]  # Default indices

def test_opencv_backends():
    """Test different OpenCV camera backends"""
    print_header("OpenCV Backend Testing")
    
    backends = []
    
    # Add backends based on platform
    if platform.system() == "Linux":
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_GSTREAMER, "GStreamer"),
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_ANY, "Default")
        ]
    elif platform.system() == "Windows":
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Default")
        ]
    else:  # macOS
        backends = [
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (cv2.CAP_ANY, "Default")
        ]
    
    working_backends = []
    
    for backend_id, backend_name in backends:
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print_status(f"{backend_name}: Working ({w}x{h})", "SUCCESS")
                    working_backends.append((backend_id, backend_name))
                else:
                    print_status(f"{backend_name}: Opened but no frames", "WARNING")
                cap.release()
            else:
                print_status(f"{backend_name}: Failed to open", "ERROR")
        except Exception as e:
            print_status(f"{backend_name}: Exception - {e}", "ERROR")
    
    return working_backends

def test_camera_resolutions(backend_id=cv2.CAP_ANY):
    """Test different camera resolutions"""
    print_header("Camera Resolution Testing")
    
    resolutions = [
        (320, 240),
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1920, 1080)
    ]
    
    working_resolutions = []
    
    try:
        cap = cv2.VideoCapture(0, backend_id)
        if not cap.isOpened():
            print_status("Could not open camera for resolution testing", "ERROR")
            return working_resolutions
        
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Give camera time to adjust
            time.sleep(0.1)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_h, frame_w = frame.shape[:2]
                if frame_w >= width * 0.8 and frame_h >= height * 0.8:  # Allow some tolerance
                    print_status(f"{width}x{height}: Working (actual: {frame_w}x{frame_h})", "SUCCESS")
                    working_resolutions.append((width, height))
                else:
                    print_status(f"{width}x{height}: Different size ({frame_w}x{frame_h})", "WARNING")
            else:
                print_status(f"{width}x{height}: No frame captured", "ERROR")
        
        cap.release()
        
    except Exception as e:
        print_status(f"Resolution testing failed: {e}", "ERROR")
    
    return working_resolutions

def test_camera_preview(backend_id=cv2.CAP_ANY, duration=5):
    """Test camera preview window"""
    print_header(f"Camera Preview Test ({duration}s)")
    
    try:
        cap = cv2.VideoCapture(0, backend_id)
        if not cap.isOpened():
            print_status("Could not open camera", "ERROR")
            return False
        
        # Set reasonable defaults
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print_status("Camera opened, starting preview...", "SUCCESS")
        print_status("Press 'q' to quit early, or wait for auto-close", "INFO")
        
        # Test display creation
        window_name = "Camera Preview Test"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print_status("Failed to read frame", "ERROR")
                break
            
            frame_count += 1
            
            # Add frame info
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {w}x{h}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {time.time() - start_time:.1f}s", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print_status("Preview stopped by user", "INFO")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            fps = frame_count / (time.time() - start_time)
            print_status(f"Preview successful: {frame_count} frames, {fps:.1f} FPS", "SUCCESS")
            return True
        else:
            print_status("No frames captured during preview", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Preview test failed: {e}", "ERROR")
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        return False

def test_jetson_specific():
    """Test Jetson-specific camera methods"""
    print_header("Jetson-Specific Camera Testing")
    
    # Check if running on Jetson
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        if "Jetson" not in model:
            print_status("Not running on Jetson platform", "INFO")
            return
        print_status(f"Detected: {model}", "SUCCESS")
    except:
        print_status("Cannot detect Jetson platform", "WARNING")
        return
    
    # Test nvgstcapture
    try:
        result = subprocess.run(['which', 'nvgstcapture'], capture_output=True)
        if result.returncode == 0:
            print_status("nvgstcapture available", "SUCCESS")
        else:
            print_status("nvgstcapture not found", "WARNING")
    except:
        print_status("Cannot check for nvgstcapture", "WARNING")
    
    # Test GStreamer pipeline
    try:
        gst_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        
        print_status("Testing GStreamer pipeline...", "INFO")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print_status("GStreamer pipeline working", "SUCCESS")
                h, w = frame.shape[:2]
                print_status(f"GStreamer frame size: {w}x{h}", "INFO")
            else:
                print_status("GStreamer pipeline opened but no frames", "WARNING")
            cap.release()
        else:
            print_status("GStreamer pipeline failed to open", "ERROR")
            
    except Exception as e:
        print_status(f"GStreamer test failed: {e}", "ERROR")

def recommend_solutions():
    """Provide recommendations based on test results"""
    print_header("Recommendations")
    
    system = platform.system()
    
    if system == "Linux":
        print("🐧 Linux Solutions:")
        print("1. Check camera permissions:")
        print("   sudo usermod -a -G video $USER")
        print("   # Then logout and login again")
        print()
        print("2. Check camera is not in use:")
        print("   sudo lsof /dev/video0")
        print()
        print("3. Install v4l-utils for camera tools:")
        print("   sudo apt install v4l-utils")
        print()
        print("4. Test camera manually:")
        print("   v4l2-ctl --list-devices")
        print("   v4l2-ctl --list-formats-ext")
        print()
        
    elif system == "Windows":
        print("🪟 Windows Solutions:")
        print("1. Check camera permissions in Settings > Privacy > Camera")
        print("2. Update camera drivers in Device Manager")
        print("3. Close other applications using camera (Skype, Teams, etc.)")
        print("4. Try running as Administrator")
        print()
        
    elif system == "Darwin":
        print("🍎 macOS Solutions:")
        print("1. Grant camera permissions in System Preferences > Security & Privacy")
        print("2. Check if camera is in use by other apps")
        print("3. Try different camera indices (0, 1, 2)")
        print()
    
    print("🔧 General Solutions:")
    print("1. Try different camera backends in your app")
    print("2. Use lower resolution (320x240 or 640x480)")
    print("3. Check USB connection and try different ports")
    print("4. Restart the camera service/reconnect camera")
    print("5. For Jetson: Use GStreamer pipeline for better performance")

def main():
    print("🔍 GLASS Camera Diagnostic Tool")
    print("This tool will test your camera setup comprehensively")
    
    # Run all tests
    devices = check_camera_devices()
    working_backends = test_opencv_backends()
    
    if working_backends:
        best_backend_id, best_backend_name = working_backends[0]
        print_status(f"Using {best_backend_name} for further tests", "INFO")
        
        working_resolutions = test_camera_resolutions(best_backend_id)
        
        # Test preview if we have a working backend
        print_status("Starting 5-second camera preview test...", "INFO")
        preview_success = test_camera_preview(best_backend_id, duration=5)
        
        if not preview_success:
            print_status("Preview failed, trying with default backend", "WARNING")
            test_camera_preview(cv2.CAP_ANY, duration=3)
    else:
        print_status("No working camera backends found", "ERROR")
    
    # Platform-specific tests
    if platform.system() == "Linux":
        test_jetson_specific()
    
    # Provide recommendations
    recommend_solutions()
    
    print_header("Diagnostic Complete")
    print_status("If camera preview worked, your camera setup is OK", "INFO")
    print_status("If not, follow the recommendations above", "INFO")

if __name__ == "__main__":
    main()