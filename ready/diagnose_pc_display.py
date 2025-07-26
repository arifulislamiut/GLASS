#!/usr/bin/env python3
"""
PC Display Diagnostic Tool for GLASS Fabric Detection
"""
import os
import sys
import subprocess
import platform
import time

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m[INFO]\033[0m",
        "SUCCESS": "\033[92m[SUCCESS]\033[0m", 
        "WARNING": "\033[93m[WARNING]\033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m"
    }
    print(f"{colors.get(status, '[INFO]')} {message}")

def run_command(cmd, description="", timeout=10):
    """Run a command and return the result"""
    try:
        if description:
            print(f"Running: {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_system_info():
    """Check basic system information"""
    print_header("System Information")
    
    # Platform info
    print_status(f"Operating System: {platform.system()} {platform.release()}", "INFO")
    print_status(f"Architecture: {platform.machine()}", "INFO")
    print_status(f"Python Version: {platform.python_version()}", "INFO")
    
    # Desktop environment (Linux)
    if platform.system() == "Linux":
        desktop = os.environ.get('XDG_CURRENT_DESKTOP', 'Unknown')
        session = os.environ.get('XDG_SESSION_TYPE', 'Unknown')
        print_status(f"Desktop Environment: {desktop}", "INFO")
        print_status(f"Session Type: {session}", "INFO")

def check_display_environment():
    """Check display environment variables"""
    print_header("Display Environment Check")
    
    # Check DISPLAY variable
    display = os.environ.get('DISPLAY', None)
    if display:
        print_status(f"DISPLAY variable: {display}", "SUCCESS")
    else:
        print_status("DISPLAY variable not set", "WARNING")
        if platform.system() != "Windows":
            print_status("This may cause GUI issues on Linux/macOS", "WARNING")
    
    # Check other relevant variables
    vars_to_check = [
        'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 'GDMSESSION', 
        'QT_X11_NO_MITSHM', 'XAUTHORITY'
    ]
    
    for var in vars_to_check:
        value = os.environ.get(var, None)
        if value:
            print_status(f"{var}: {value}", "INFO")

def check_gui_libraries():
    """Check GUI library availability"""
    print_header("GUI Library Check")
    
    # Check OpenCV
    try:
        import cv2
        print_status(f"OpenCV version: {cv2.__version__}", "SUCCESS")
        
        # Check OpenCV build info
        build_info = cv2.getBuildInformation()
        if "GUI" in build_info:
            gui_backends = []
            for line in build_info.split('\n'):
                if 'QT' in line and 'YES' in line:
                    gui_backends.append('Qt')
                elif 'GTK' in line and 'YES' in line:
                    gui_backends.append('GTK')
                elif 'Win32 UI' in line and 'YES' in line:
                    gui_backends.append('Win32')
            
            if gui_backends:
                print_status(f"OpenCV GUI backends: {', '.join(gui_backends)}", "SUCCESS")
            else:
                print_status("OpenCV GUI backends: None found", "WARNING")
        
    except ImportError as e:
        print_status(f"OpenCV not available: {e}", "ERROR")
        return False
    
    # Check other important libraries
    libraries = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('torch', 'PyTorch'),
        ('psutil', 'psutil')
    ]
    
    for lib_name, display_name in libraries:
        try:
            lib = __import__(lib_name)
            version = getattr(lib, '__version__', 'Unknown')
            print_status(f"{display_name}: {version}", "SUCCESS")
        except ImportError:
            print_status(f"{display_name}: Not available", "WARNING")
    
    return True

def check_display_server():
    """Check display server (X11/Wayland)"""
    print_header("Display Server Check")
    
    if platform.system() == "Windows":
        print_status("Windows - Native display system", "SUCCESS")
        return True
    elif platform.system() == "Darwin":
        print_status("macOS - Native display system", "SUCCESS")
        # Check if XQuartz is available for X11 forwarding
        success, output, error = run_command("which xquartz", "Checking XQuartz")
        if success:
            print_status("XQuartz available for X11 applications", "SUCCESS")
        return True
    else:  # Linux
        # Check X11
        success, output, error = run_command("ps aux | grep -i '[x]org\\|[x]11' | head -5", "Checking X11 processes")
        if success and output:
            print_status("X11 server is running", "SUCCESS")
            print(f"   Processes: {output.split()[0] if output else 'Running'}")
        else:
            print_status("X11 server not found", "WARNING")
        
        # Check Wayland
        wayland_display = os.environ.get('WAYLAND_DISPLAY')
        if wayland_display:
            print_status(f"Wayland display: {wayland_display}", "SUCCESS")
        
        # Test display access
        success, output, error = run_command("xdpyinfo -display :0 2>/dev/null | head -3", "Testing display access")
        if success and output:
            print_status("Display access successful", "SUCCESS")
        else:
            print_status("Display access failed", "WARNING")
            if error:
                print_status(f"Error: {error}", "INFO")

def check_camera_system():
    """Check camera system"""
    print_header("Camera System Check")
    
    if platform.system() == "Windows":
        # Windows - check DirectShow devices
        print_status("Checking Windows camera system...", "INFO")
        
    elif platform.system() == "Linux":
        # Linux - check V4L2 devices
        success, output, error = run_command("ls -la /dev/video* 2>/dev/null", "Listing video devices")
        if success and output:
            print_status("Video devices found:", "SUCCESS")
            for line in output.split('\n'):
                if 'video' in line:
                    print(f"   {line}")
        else:
            print_status("No video devices found", "WARNING")
        
        # Check v4l2 info
        success, output, error = run_command("which v4l2-ctl", "Checking v4l2-ctl")
        if success:
            success, output, error = run_command("v4l2-ctl --list-devices 2>/dev/null", "V4L2 device list")
            if success and output:
                print_status("V4L2 devices:", "SUCCESS")
                print(f"   {output}")
    
    elif platform.system() == "Darwin":
        # macOS - basic check
        print_status("macOS camera system - checking basic availability", "INFO")

def test_opencv_display():
    """Test OpenCV display functionality"""
    print_header("OpenCV Display Test")
    
    try:
        import cv2
        import numpy as np
        
        # Test 1: Basic window creation
        print_status("Test 1: Basic window creation", "INFO")
        try:
            test_img = np.zeros((200, 300, 3), dtype=np.uint8)
            # Add some content to the test image
            cv2.rectangle(test_img, (50, 50), (250, 150), (0, 255, 0), 2)
            cv2.putText(test_img, 'DISPLAY TEST', (70, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.namedWindow('OpenCV Display Test', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('OpenCV Display Test', test_img)
            print_status("  Window created, waiting 3 seconds...", "INFO")
            
            # Wait and check for window
            key = cv2.waitKey(3000)  # Wait 3 seconds
            cv2.destroyWindow('OpenCV Display Test')
            
            if key != -1:
                print_status("  ✅ Display test passed (key pressed)", "SUCCESS")
            else:
                print_status("  ✅ Display test passed (timeout)", "SUCCESS")
                
        except Exception as e:
            print_status(f"  ❌ Display test failed: {e}", "ERROR")
            return False
        
        # Test 2: Camera capture test
        print_status("Test 2: Camera capture test", "INFO")
        try:
            # Try different camera backends
            backends = [(cv2.CAP_ANY, "Default")]
            
            if platform.system() == "Windows":
                backends.append((cv2.CAP_DSHOW, "DirectShow"))
            elif platform.system() == "Linux":
                backends.append((cv2.CAP_V4L2, "V4L2"))
            
            camera_working = False
            for backend_id, backend_name in backends:
                try:
                    cap = cv2.VideoCapture(0, backend_id)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            print_status(f"  ✅ Camera working with {backend_name}: {frame.shape}", "SUCCESS")
                            
                            # Try to display camera frame
                            cv2.namedWindow('Camera Test', cv2.WINDOW_AUTOSIZE)
                            cv2.imshow('Camera Test', frame)
                            cv2.waitKey(2000)  # Show for 2 seconds
                            cv2.destroyWindow('Camera Test')
                            print_status(f"  ✅ Camera display test passed", "SUCCESS")
                            
                            camera_working = True
                            cap.release()
                            break
                        else:
                            print_status(f"  ❌ {backend_name}: Could not read frame", "WARNING")
                        cap.release()
                    else:
                        print_status(f"  ❌ {backend_name}: Could not open camera", "WARNING")
                except Exception as e:
                    print_status(f"  ❌ {backend_name}: {e}", "WARNING")
            
            if not camera_working:
                print_status("  ❌ No camera backends working", "ERROR")
                
        except Exception as e:
            print_status(f"  ❌ Camera test failed: {e}", "ERROR")
        
        return True
        
    except ImportError:
        print_status("OpenCV not available for testing", "ERROR")
        return False

def test_glass_system():
    """Test GLASS system components"""
    print_header("GLASS System Test")
    
    # Check model files
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'wfdd_grid_cloth')
    if os.path.exists(model_path):
        print_status(f"Model directory found: {model_path}", "SUCCESS")
        
        # Check for model files
        files = os.listdir(model_path)
        onnx_files = [f for f in files if f.endswith('.onnx')]
        pth_files = [f for f in files if f.endswith('.pth')]
        
        if onnx_files:
            print_status(f"ONNX models: {', '.join(onnx_files)}", "SUCCESS")
        if pth_files:
            print_status(f"PyTorch models: {', '.join(pth_files)}", "SUCCESS")
        
        if not onnx_files and not pth_files:
            print_status("No model files found", "WARNING")
    else:
        print_status(f"Model directory not found: {model_path}", "ERROR")
    
    # Check GLASS components
    components = [
        'utils/glass.py',
        'utils/model.py',
        'utils/common.py',
        'scripts/predict_single_image.py'
    ]
    
    for component in components:
        if os.path.exists(component):
            print_status(f"Component found: {component}", "SUCCESS")
        else:
            print_status(f"Component missing: {component}", "WARNING")

def check_cuda_system():
    """Check CUDA system"""
    print_header("CUDA System Check")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print_status("CUDA is available", "SUCCESS")
            print_status(f"CUDA version: {torch.version.cuda}", "INFO")
            print_status(f"Device count: {torch.cuda.device_count()}", "INFO")
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_status(f"Device {i}: {device_name} ({device_memory:.1f}GB)", "INFO")
        else:
            print_status("CUDA not available - will use CPU", "WARNING")
            
    except ImportError:
        print_status("PyTorch not available", "ERROR")

def provide_solutions():
    """Provide solutions for common issues"""
    print_header("Common Solutions")
    
    solutions = {
        "Display not working (Linux)": [
            "export DISPLAY=:0",
            "xhost +local:root (if running as root)",
            "Install: sudo apt-get install python3-opencv",
            "For SSH: ssh -X username@hostname"
        ],
        "Display not working (Windows)": [
            "Check camera permissions in Privacy settings",
            "Install Visual C++ Redistributable",
            "Try running as Administrator"
        ],
        "Camera not detected": [
            "Linux: Check permissions: ls -la /dev/video*",
            "Windows: Check Device Manager",
            "Try different camera backends",
            "Disconnect/reconnect camera"
        ],
        "Performance issues": [
            "Use CUDA if available",
            "Lower processing interval",
            "Use headless mode for production",
            "Close other applications"
        ],
        "Import errors": [
            "pip install -r requirements.txt",
            "Check Python path and virtual environment",
            "Install missing dependencies"
        ]
    }
    
    for issue, solutions_list in solutions.items():
        print(f"\n{issue}:")
        for i, solution in enumerate(solutions_list, 1):
            print(f"  {i}. {solution}")

def main():
    print("🔧 GLASS PC Display Diagnostic Tool")
    print("This tool will help diagnose display and camera issues")
    
    # Run all checks
    check_system_info()
    check_display_environment()
    check_gui_libraries()
    check_display_server()
    check_camera_system()
    test_opencv_display()
    test_glass_system()
    check_cuda_system()
    provide_solutions()
    
    print_header("Diagnostic Complete")
    print_status("Next steps:", "INFO")
    print_status("1. Fix any ERROR items above", "INFO")
    print_status("2. Try running: python run_fabric_detection_fixed.py", "INFO")
    print_status("3. Use --headless flag if display issues persist", "INFO")
    print_status("4. Use --diagnose flag in the main script for quick checks", "INFO")

if __name__ == "__main__":
    main()