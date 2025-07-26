#!/usr/bin/env python3
"""
Jetson Display Diagnostic Tool for GLASS
"""
import os
import sys
import subprocess
import time

def print_header(title):
    print(f"\n{'='*50}")
    print(f"🔍 {title}")
    print('='*50)

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m[INFO]\033[0m",
        "SUCCESS": "\033[92m[SUCCESS]\033[0m", 
        "WARNING": "\033[93m[WARNING]\033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m"
    }
    print(f"{colors.get(status, '[INFO]')} {message}")

def run_command(cmd, description=""):
    """Run a command and return the result"""
    try:
        if description:
            print(f"Running: {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_display_environment():
    """Check display environment variables"""
    print_header("Display Environment Check")
    
    # Check DISPLAY variable
    display = os.environ.get('DISPLAY', None)
    if display:
        print_status(f"DISPLAY variable: {display}", "SUCCESS")
    else:
        print_status("DISPLAY variable not set", "WARNING")
        print_status("Setting DISPLAY=:0", "INFO")
        os.environ['DISPLAY'] = ':0'
    
    # Check other relevant variables
    vars_to_check = ['WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 'GDMSESSION']
    for var in vars_to_check:
        value = os.environ.get(var, None)
        if value:
            print_status(f"{var}: {value}", "INFO")
        else:
            print_status(f"{var}: not set", "INFO")

def check_x11_display():
    """Check X11 display server"""
    print_header("X11 Display Server Check")
    
    # Check if X11 is running
    success, output, error = run_command("ps aux | grep -i 'xorg\\|x11' | grep -v grep", "Checking X11 processes")
    if success and output:
        print_status("X11 server is running", "SUCCESS")
        print(f"   Processes: {output}")
    else:
        print_status("X11 server not found", "WARNING")
    
    # Check display access
    success, output, error = run_command("xdpyinfo", "Testing display access")
    if success:
        print_status("Display access successful", "SUCCESS")
        print(f"   Display info: {output.split()[0] if output else 'Available'}")
    else:
        print_status("Display access failed", "ERROR")
        print_status(f"Error: {error}", "ERROR")

def check_opencv_backends():
    """Check OpenCV backend capabilities"""
    print_header("OpenCV Backend Check")
    
    try:
        import cv2
        print_status("OpenCV imported successfully", "SUCCESS")
        print_status(f"OpenCV version: {cv2.__version__}", "INFO")
        
        # Check GUI support
        try:
            # Create a test window
            test_img = cv2.imread('/dev/null')  # Will be None
            if test_img is None:
                # Create a dummy image
                import numpy as np
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Test window creation
            cv2.namedWindow('Test Window', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Test Window', test_img)
            cv2.waitKey(1)
            cv2.destroyWindow('Test Window')
            print_status("OpenCV GUI test successful", "SUCCESS")
            
        except Exception as e:
            print_status(f"OpenCV GUI test failed: {e}", "ERROR")
            
        # Check video capture backends
        print_status("Available VideoCapture backends:", "INFO")
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_GSTREAMER, "GStreamer"),
            (cv2.CAP_FFMPEG, "FFMPEG"),
        ]
        
        for backend_id, backend_name in backends:
            try:
                cap = cv2.VideoCapture(0, backend_id)
                if cap.isOpened():
                    print_status(f"  ✅ {backend_name} - Working", "SUCCESS")
                    cap.release()
                else:
                    print_status(f"  ❌ {backend_name} - Failed", "WARNING")
            except:
                print_status(f"  ❌ {backend_name} - Not available", "WARNING")
                
    except ImportError as e:
        print_status(f"OpenCV import failed: {e}", "ERROR")

def check_camera_devices():
    """Check available camera devices"""
    print_header("Camera Device Check")
    
    # Check /dev/video* devices
    success, output, error = run_command("ls -la /dev/video*", "Listing video devices")
    if success and output:
        print_status("Video devices found:", "SUCCESS")
        for line in output.split('\n'):
            if 'video' in line:
                print(f"   {line}")
    else:
        print_status("No video devices found", "WARNING")
    
    # Check v4l2 information
    success, output, error = run_command("v4l2-ctl --list-devices", "V4L2 device list")
    if success and output:
        print_status("V4L2 devices:", "SUCCESS")
        print(f"   {output}")
    else:
        print_status("V4L2 info not available", "WARNING")

def check_jetson_specific():
    """Check Jetson-specific display components"""
    print_header("Jetson Specific Checks")
    
    # Check if running on Jetson
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        print_status(f"Platform: {model}", "SUCCESS")
        
        if "Jetson" not in model:
            print_status("Not running on Jetson platform", "WARNING")
            return
            
    except:
        print_status("Could not detect platform", "WARNING")
        return
    
    # Check Jetson display manager
    success, output, error = run_command("systemctl status gdm3", "Checking display manager")
    if success:
        print_status("Display manager (gdm3) is running", "SUCCESS")
    else:
        success, output, error = run_command("systemctl status lightdm", "Checking LightDM")
        if success:
            print_status("Display manager (lightdm) is running", "SUCCESS")
        else:
            print_status("No display manager found", "WARNING")
    
    # Check for nvgstcapture (Jetson camera tool)
    success, output, error = run_command("which nvgstcapture", "Checking nvgstcapture")
    if success:
        print_status("nvgstcapture available (Jetson camera tool)", "SUCCESS")
    else:
        print_status("nvgstcapture not found", "INFO")

def test_display_solutions():
    """Test various display solutions"""
    print_header("Display Solution Tests")
    
    try:
        import cv2
        import numpy as np
        
        # Test 1: Basic window creation
        print_status("Test 1: Basic window creation", "INFO")
        try:
            test_img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (50, 50), (150, 150), (0, 255, 0), 2)
            cv2.putText(test_img, 'TEST', (70, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.namedWindow('Display Test 1', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Display Test 1', test_img)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyWindow('Display Test 1')
            print_status("  ✅ Basic window test passed", "SUCCESS")
            
        except Exception as e:
            print_status(f"  ❌ Basic window test failed: {e}", "ERROR")
        
        # Test 2: Camera capture test
        print_status("Test 2: Camera capture test", "INFO")
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print_status("  ✅ Camera capture successful", "SUCCESS")
                    
                    # Try to display frame
                    cv2.namedWindow('Camera Test', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Camera Test', frame)
                    cv2.waitKey(2000)
                    cv2.destroyWindow('Camera Test')
                    print_status("  ✅ Camera display test passed", "SUCCESS")
                else:
                    print_status("  ❌ Could not read camera frame", "ERROR")
                cap.release()
            else:
                print_status("  ❌ Could not open camera", "ERROR")
                
        except Exception as e:
            print_status(f"  ❌ Camera test failed: {e}", "ERROR")
            
    except ImportError:
        print_status("OpenCV not available for display tests", "ERROR")

def provide_solutions():
    """Provide solutions for common display issues"""
    print_header("Common Solutions")
    
    solutions = [
        {
            "issue": "DISPLAY variable not set",
            "solution": "export DISPLAY=:0"
        },
        {
            "issue": "Permission denied for display",
            "solution": "xhost +local: (run from desktop session)"
        },
        {
            "issue": "No display server running", 
            "solution": "Start desktop session or use VNC/X11 forwarding"
        },
        {
            "issue": "OpenCV GUI not working",
            "solution": "Install: sudo apt-get install python3-opencv libopencv-dev"
        },
        {
            "issue": "Jetson camera not detected",
            "solution": "Check camera connection and permissions: ls -la /dev/video*"
        },
        {
            "issue": "Display over SSH",
            "solution": "Use SSH with X11 forwarding: ssh -X user@jetson"
        },
        {
            "issue": "Headless operation needed",
            "solution": "Use --headless flag or set headless_mode: true in config"
        }
    ]
    
    for i, sol in enumerate(solutions, 1):
        print(f"{i}. {sol['issue']}")
        print(f"   Solution: {sol['solution']}")
        print()

def main():
    print("🔧 GLASS Jetson Display Diagnostic Tool")
    print("This tool will help diagnose display issues on Jetson")
    
    check_display_environment()
    check_x11_display()
    check_opencv_backends()
    check_camera_devices()
    check_jetson_specific()
    test_display_solutions()
    provide_solutions()
    
    print_header("Diagnostic Complete")
    print_status("Run the GLASS system with:", "INFO")
    print_status("  glass-detect --headless     # For headless mode", "INFO")
    print_status("  glass-detect                # For GUI mode (if display works)", "INFO")
    print()
    print_status("If display still doesn't work, try:", "INFO")
    print_status("  1. Run from desktop session (not SSH)", "INFO")
    print_status("  2. Use VNC to connect to Jetson desktop", "INFO")
    print_status("  3. Use headless mode for production", "INFO")

if __name__ == "__main__":
    main()