#!/usr/bin/env python3
"""
GLASS Fabric Detection - Jetson Orin Nano Complete Runner with Full PC Functionality
"""
import os
import sys
import yaml
import time
import psutil
import argparse
from pathlib import Path

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'utils'))
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

# Also set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = f"{os.path.join(script_dir, 'utils')}:{os.path.join(script_dir, 'scripts')}:{os.environ.get('PYTHONPATH', '')}"

import cv2
import numpy as np
import torch
from collections import deque

class JetsonCompleteFabricDetector:
    def __init__(self, config_path="jetson_config.yaml", headless=False):
        """Initialize Jetson-optimized detector with complete PC functionality"""
        self.headless = headless
        self.load_config(config_path)
        
        if not headless:
            self.setup_display_fixes()
        
        self.setup_jetson_optimizations()
        self.load_model()
        self.setup_camera()
        self.setup_monitoring()
    
    def load_config(self, config_path):
        """Load Jetson configuration with PC-like defaults"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print("⚠️  Config file not found, using defaults")
            self.config = self.get_default_config()
        
        # Apply performance profile
        profile = self.config.get('performance', {}).get('power_mode', 'balanced')
        if profile in self.config.get('profiles', {}):
            profile_config = self.config['profiles'][profile]
            self.config['model'].update(profile_config)
            self.config['performance'].update(profile_config)
    
    def get_default_config(self):
        """Get default configuration matching PC functionality"""
        return {
            'model': {
                'pytorch_model': 'models/wfdd_grid_cloth/ckpt_best_571.pth',
                'input_resolution': 256,
                'precision': 'fp16'
            },
            'camera': {
                'device_id': 0,
                'width': 640,
                'height': 480,
                'fps': 30,
                'buffer_size': 1,
                'format': 'MJPG'
            },
            'performance': {
                'processing_interval': 1,  # Process every frame like PC
                'inference_threads': 4
            },
            'display': {
                'enable_gui': True,
                'headless_mode': False,
                'fallback_to_headless': True,
                'window_name': 'GLASS Jetson Fabric Detection',
                'save_frames': False,
                'default_mode': 0  # All views like PC
            }
        }
    
    def setup_display_fixes(self):
        """Setup display fixes for Jetson"""
        print("🔧 Setting up display fixes...")
        
        # Set display environment variables
        if 'DISPLAY' not in os.environ:
            os.environ['DISPLAY'] = ':0'
            print(f"Set DISPLAY to {os.environ['DISPLAY']}")
        
        # Set Qt backend to avoid conflicts
        os.environ['QT_X11_NO_MITSHM'] = '1'
        
        # Force OpenCV to use specific backend
        try:
            # Try different backends in order of preference
            backends = [
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_GSTREAMER, "GStreamer"), 
                (cv2.CAP_FFMPEG, "FFMPEG")
            ]
            
            self.camera_backend = cv2.CAP_V4L2  # Default
            print("Available camera backends tested")
            
        except Exception as e:
            print(f"Backend setup warning: {e}")
        
        # Test display capabilities
        self.has_display = self.test_display()
        
        if not self.has_display and self.config['display']['fallback_to_headless']:
            print("⚠️  Display not available, switching to headless mode")
            self.headless = True
    
    def test_display(self):
        """Test if display is working"""
        try:
            # Test creating a simple window
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Display Test', test_img)
            cv2.waitKey(1)
            cv2.destroyWindow('Display Test')
            print("✅ Display test passed")
            return True
        except Exception as e:
            print(f"⚠️  Display test failed: {e}")
            print("Will use headless mode...")
            return False

    def setup_jetson_optimizations(self):
        """Setup Jetson-specific optimizations"""
        # Set CUDA optimizations
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        
        # Enable FP16 if supported
        if self.config['model']['precision'] == 'fp16':
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        
        # Memory management
        torch.cuda.empty_cache()
        
        # Set number of threads
        torch.set_num_threads(self.config['performance']['inference_threads'])
    
    def load_model(self):
        """Load optimized model for Jetson"""
        from predict_single_image import GLASSPredictor
        
        model_path = self.config['model']['pytorch_model']
        model_dir = os.path.dirname(model_path)
        
        print(f"Loading model: {model_path}")
        self.predictor = GLASSPredictor(model_dir, "pytorch", "cuda")
        
        # Warmup
        dummy_img = np.random.randint(0, 255, (288, 288, 3), dtype=np.uint8)
        from PIL import Image
        dummy_pil = Image.fromarray(dummy_img)
        self.predictor.predict(dummy_pil)
        
        print("✅ Model loaded and warmed up")
    
    def setup_camera(self):
        """Setup camera with Jetson optimizations and display fixes"""
        cam_config = self.config['camera']
        
        # Try different camera initialization methods
        camera_initialized = False
        
        # Method 1: Try with specific backend
        try:
            self.cap = cv2.VideoCapture(cam_config['device_id'], self.camera_backend)
            if self.cap.isOpened():
                camera_initialized = True
                print(f"✅ Camera opened with backend")
        except Exception as e:
            print(f"Backend method failed: {e}")
        
        # Method 2: Try default method
        if not camera_initialized:
            try:
                self.cap = cv2.VideoCapture(cam_config['device_id'])
                if self.cap.isOpened():
                    camera_initialized = True
                    print("✅ Camera opened with default method")
            except Exception as e:
                print(f"Default method failed: {e}")
        
        # Method 3: Try GStreamer pipeline (for Jetson cameras)
        if not camera_initialized:
            try:
                gst_pipeline = f"nvarguscamerasrc sensor-id={cam_config['device_id']} ! video/x-raw(memory:NVMM), width={cam_config['width']}, height={cam_config['height']}, format=(string)NV12, framerate=(fraction){cam_config['fps']}/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    camera_initialized = True
                    print("✅ Camera opened with GStreamer pipeline")
            except Exception as e:
                print(f"GStreamer method failed: {e}")
        
        if not camera_initialized:
            raise RuntimeError("❌ Could not initialize camera with any method")
        
        # Configure camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['height'])
        self.cap.set(cv2.CAP_PROP_FPS, cam_config['fps'])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cam_config['buffer_size'])
        
        if cam_config['format'] == 'MJPG':
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Test frame capture
        ret, test_frame = self.cap.read()
        if ret:
            print(f"✅ Camera test successful: {test_frame.shape}")
        else:
            print("⚠️  Camera test failed - no frame received")
        
        print(f"✅ Camera initialized: {cam_config['width']}x{cam_config['height']}")
    
    def setup_monitoring(self):
        """Setup system monitoring with PC-like features"""
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.memory_usage = deque(maxlen=30)
        self.last_gc = time.time()
        
        # Display settings - same as PC
        self.display_mode = self.config['display'].get('default_mode', 0)  # 0=all, 1=original+anomaly, 2=original+overlay, 3=original only
        self.show_overlay = True
        self.last_results = None
        
        # Runtime resolution control
        self.available_resolutions = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
        self.current_resolution = self.config['model'].get('input_resolution', 288)
        try:
            self.resolution_index = self.available_resolutions.index(self.current_resolution)
        except ValueError:
            self.current_resolution = 288
            self.resolution_index = self.available_resolutions.index(288)
        
        # Help display state
        self.show_help = False
        self.help_start_time = 0
        self.help_duration = 5.0  # Show help for 5 seconds
    
    def get_system_stats(self):
        """Get Jetson system statistics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU memory and utilization (if available)
        try:
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
            
            # Try to get GPU utilization on Jetson
            try:
                # Jetson-specific GPU utilization
                with open('/sys/devices/gpu.0/load', 'r') as f:
                    gpu_utilization = int(f.read().strip())
            except:
                # Fallback to nvidia-ml-py if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = utilization.gpu
                except:
                    gpu_utilization = 0
        except:
            gpu_memory = 0
            gpu_memory_cached = 0
            gpu_utilization = 0
        
        # Temperature (Jetson specific)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000  # Convert to Celsius
        except:
            temp = 0
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available_gb': memory.available / 1024**3,
            'gpu_memory_gb': gpu_memory,
            'gpu_memory_cached_gb': gpu_memory_cached,
            'gpu_utilization': gpu_utilization,
            'temperature_c': temp
        }
    
    def process_frame(self, frame):
        """Process frame with Jetson optimizations and runtime resolution"""
        try:
            # Convert to RGB (predictor will handle resizing to current_resolution)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Run inference - predictor will resize to self.image_size internally
            start_time = time.time()
            results = self.predictor.predict(pil_image)
            inference_time = time.time() - start_time
            
            self.processing_times.append(inference_time)
            
            return results, inference_time
            
        except Exception as e:
            print(f"Processing error: {e}")
            return None, 0
    
    def increase_resolution(self):
        """Increase processing resolution"""
        if self.resolution_index < len(self.available_resolutions) - 1:
            self.resolution_index += 1
            self.current_resolution = self.available_resolutions[self.resolution_index]
            # Update model image size
            self.predictor.set_image_size(self.current_resolution)
            print(f"🔍 Resolution increased to {self.current_resolution}x{self.current_resolution}")
            return True
        else:
            print(f"⚠️  Already at maximum resolution ({self.current_resolution}x{self.current_resolution})")
            return False
    
    def decrease_resolution(self):
        """Decrease processing resolution"""
        if self.resolution_index > 0:
            self.resolution_index -= 1
            self.current_resolution = self.available_resolutions[self.resolution_index]
            # Update model image size
            self.predictor.set_image_size(self.current_resolution)
            print(f"🔍 Resolution decreased to {self.current_resolution}x{self.current_resolution}")
            return True
        else:
            print(f"⚠️  Already at minimum resolution ({self.current_resolution}x{self.current_resolution})")
            return False
    
    def create_display_frame(self, frame, results, stats, fps):
        """Create enhanced display frame with full PC functionality"""
        if self.display_mode == 3:  # Original only
            display_frame = frame.copy()
            # Show help overlay if requested, otherwise show controls
            if self.show_help:
                display_frame = self.create_help_overlay(display_frame)
            else:
                # Add control text for original only mode
                control_y = display_frame.shape[0] - 25
                cv2.putText(display_frame, "Controls: q:Quit s:Save d:Display +/-:Res h:Help", 
                           (10, control_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        else:
            # Standard multi-panel display - same as PC
            display_width = 960
            display_height = 540
            panel_width = display_width // 2  # 480
            panel_height = display_height // 2  # 270
            
            # Always resize main frame consistently
            main_frame = cv2.resize(frame, (panel_width, panel_height))
            
            if results is None:
                # Show only main frame when no results
                display_frame = cv2.resize(main_frame, (display_width, display_height))
                cv2.putText(display_frame, "Processing...", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                return display_frame
            
            # Create panels based on display mode
            if self.display_mode == 0:  # All views - exactly like PC
                # Create 2x2 layout
                display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                
                # Panel positions
                positions = [(0, 0), (panel_width, 0), (0, panel_height), (panel_width, panel_height)]
                
                # Ensure all panels are exactly the right size
                try:
                    # Original frame with control text or help overlay
                    main_panel = cv2.resize(main_frame, (panel_width, panel_height))
                    
                    # Show help overlay if requested
                    if self.show_help:
                        main_panel = self.create_help_overlay(main_panel)
                    else:
                        # Add control text directly to main panel before placing it
                        control_y = panel_height - 35
                        cv2.putText(main_panel, "Controls:", (10, control_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        cv2.putText(main_panel, "q:Quit s:Save d:Display +/-:Res h:Help", 
                                   (10, control_y + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
                    
                    display_frame[positions[0][1]:positions[0][1]+panel_height, 
                                 positions[0][0]:positions[0][0]+panel_width] = main_panel
                    
                    # Anomaly map
                    if 'anomaly_map' in results:
                        anomaly_panel = self.create_anomaly_panel(results['anomaly_map'], panel_width, panel_height)
                        display_frame[positions[1][1]:positions[1][1]+panel_height,
                                     positions[1][0]:positions[1][0]+panel_width] = anomaly_panel
                    
                    # Overlay
                    overlay_panel = self.create_overlay_panel(main_panel, results)  # Use resized main_panel
                    display_frame[positions[2][1]:positions[2][1]+panel_height,
                                 positions[2][0]:positions[2][0]+panel_width] = overlay_panel
                    
                    # Info panel - enhanced with Jetson-specific info
                    info_panel = self.create_info_panel(results, stats, fps, panel_width, panel_height)
                    display_frame[positions[3][1]:positions[3][1]+panel_height,
                                 positions[3][0]:positions[3][0]+panel_width] = info_panel
                
                except Exception as e:
                    print(f"Display creation error: {e}")
                    # Fallback to simple display
                    display_frame = cv2.resize(main_frame, (display_width, display_height))
                    cv2.putText(display_frame, "Display Error - Simple Mode", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
            elif self.display_mode == 1:  # Original + Anomaly
                try:
                    display_frame = np.zeros((panel_height, display_width, 3), dtype=np.uint8)
                    main_panel = cv2.resize(main_frame, (panel_width, panel_height))
                    
                    # Show help overlay if requested, otherwise show controls
                    if self.show_help:
                        main_panel = self.create_help_overlay(main_panel)
                    else:
                        # Add control text to main panel
                        control_y = panel_height - 35
                        cv2.putText(main_panel, "Controls: q:Quit s:Save d:Display +/-:Res h:Help", 
                                   (10, control_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
                    
                    display_frame[:, :panel_width] = main_panel
                    if 'anomaly_map' in results:
                        anomaly_panel = self.create_anomaly_panel(results['anomaly_map'], panel_width, panel_height)
                        display_frame[:, panel_width:] = anomaly_panel
                except Exception as e:
                    print(f"Mode 1 display error: {e}")
                    display_frame = cv2.resize(main_frame, (display_width, display_height))
                    
            elif self.display_mode == 2:  # Original + Overlay
                try:
                    display_frame = np.zeros((panel_height, display_width, 3), dtype=np.uint8)
                    main_panel = cv2.resize(main_frame, (panel_width, panel_height))
                    
                    # Show help overlay if requested, otherwise show controls
                    if self.show_help:
                        main_panel = self.create_help_overlay(main_panel)
                    else:
                        # Add control text to main panel
                        control_y = panel_height - 35
                        cv2.putText(main_panel, "Controls: q:Quit s:Save d:Display +/-:Res h:Help", 
                                   (10, control_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
                    
                    display_frame[:, :panel_width] = main_panel
                    overlay_panel = self.create_overlay_panel(main_panel, results)
                    display_frame[:, panel_width:] = overlay_panel
                except Exception as e:
                    print(f"Mode 2 display error: {e}")
                    display_frame = cv2.resize(main_frame, (display_width, display_height))
        
        # Add status information
        self.add_status_overlay(display_frame, results, stats, fps)
        
        return display_frame
    
    def create_anomaly_panel(self, anomaly_map, width, height):
        """Create anomaly map visualization panel - same as PC"""
        try:
            # Normalize and convert anomaly map
            anomaly_normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
            anomaly_colored = cv2.applyColorMap(anomaly_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            anomaly_resized = cv2.resize(anomaly_colored, (width, height))
            
            # Add title
            cv2.putText(anomaly_resized, 'Anomaly Map', (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return anomaly_resized
        except:
            # Fallback - black panel with text
            panel = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(panel, 'Anomaly Map', (10, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(panel, 'Not Available', (10, height//2 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            return panel
    
    def create_overlay_panel(self, base_frame, results):
        """Create overlay panel with detection results - same as PC"""
        overlay_frame = base_frame.copy()
        
        if results and results.get('is_anomalous', False):
            # Add red overlay for anomaly
            red_overlay = np.zeros_like(overlay_frame)
            red_overlay[:, :] = [0, 0, 255]
            overlay_frame = cv2.addWeighted(overlay_frame, 0.7, red_overlay, 0.3, 0)
            
            # Add anomaly text
            cv2.putText(overlay_frame, 'DEFECT DETECTED!', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Add green overlay for normal
            green_overlay = np.zeros_like(overlay_frame)
            green_overlay[:, :] = [0, 255, 0]
            overlay_frame = cv2.addWeighted(overlay_frame, 0.9, green_overlay, 0.1, 0)
            
            cv2.putText(overlay_frame, 'OK', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return overlay_frame
    
    def create_help_overlay(self, base_frame):
        """Create help overlay panel for Jetson complete version"""
        help_frame = base_frame.copy()
        
        # Create semi-transparent dark overlay
        overlay = np.zeros_like(help_frame)
        help_frame = cv2.addWeighted(help_frame, 0.3, overlay, 0.7, 0)
        
        # Help text content for Jetson
        help_lines = [
            "GLASS Jetson Fabric Detection - Help",
            "",
            "Controls:",
            "q - Quit application",
            "s - Save current frame",
            "d - Cycle display modes:",
            "    0: All Views (2x2 grid)",
            "    1: Original + Anomaly Map", 
            "    2: Original + Overlay",
            "    3: Original Only",
            "+ / = - Increase resolution",
            "- / _ - Decrease resolution", 
            "r - Reset to default resolution",
            "h - Show/hide this help",
            "",
            f"Current Resolution: {self.current_resolution}x{self.current_resolution}",
            f"Display Mode: {self.display_mode}",
            f"Precision: {self.config['model']['precision']}",
            "",
            "Help will auto-hide in 5 seconds..."
        ]
        
        # Calculate text positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        line_height = 20
        start_y = 30
        
        # Draw help text
        for i, line in enumerate(help_lines):
            y_pos = start_y + (i * line_height)
            if y_pos > help_frame.shape[0] - 10:  # Stop if we run out of space
                break
                
            # Choose color based on line type
            if line == "GLASS Jetson Fabric Detection - Help":
                color = (0, 255, 255)  # Yellow for title
                thickness = 2
            elif line.startswith("    "):
                color = (200, 200, 200)  # Light gray for indented items
                thickness = 1
            elif line in ["Controls:", "Current Resolution:", "Display Mode:", "Precision:"]:
                color = (255, 255, 255)  # White for section headers
                thickness = 1
            else:
                color = (220, 220, 220)  # Light gray for regular text
                thickness = 1
            
            cv2.putText(help_frame, line, (10, y_pos), font, font_scale, color, thickness)
        
        return help_frame
    
    def create_info_panel(self, results, stats, fps, width, height):
        """Create information panel with Jetson-specific enhancements"""
        info_panel = np.zeros((height, width, 3), dtype=np.uint8) + 20
        
        # Title
        cv2.putText(info_panel, 'Jetson System Info', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = 50
        line_height = 22
        
        # Detection results
        if results:
            score = results.get('image_score', 0)
            is_anomalous = results.get('is_anomalous', False)
            status = "DEFECT" if is_anomalous else "OK"
            color = (0, 0, 255) if is_anomalous else (0, 255, 0)
            
            cv2.putText(info_panel, f"Status: {status}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(info_panel, f"Score: {score:.3f}", (10, y_offset + line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height * 2
        
        # Performance stats
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"CPU: {stats['cpu_percent']:.1f}%", (10, y_offset + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Memory: {stats['memory_percent']:.1f}%", (10, y_offset + line_height*2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"GPU: {stats['gpu_memory_gb']:.1f}GB", (10, y_offset + line_height*3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Temp: {stats['temperature_c']:.1f}C", (10, y_offset + line_height*4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # GPU utilization if available
        if stats.get('gpu_utilization', 0) > 0:
            cv2.putText(info_panel, f"GPU Load: {stats['gpu_utilization']:.0f}%", (10, y_offset + line_height*5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Processing resolution
        cv2.putText(info_panel, f"Resolution: {self.current_resolution}x{self.current_resolution}", (10, y_offset + line_height*5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        # Processing time
        if self.processing_times:
            avg_time = np.mean(list(self.processing_times)) * 1000  # ms
            cv2.putText(info_panel, f"Inference: {avg_time:.1f}ms", (10, y_offset + line_height*6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls - same as PC with resolution control
        y_offset = height - 130
        cv2.putText(info_panel, "Controls:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, "q: Quit", (10, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(info_panel, "s: Save", (10, y_offset + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(info_panel, "d: Display", (10, y_offset + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(info_panel, "+/-: Resolution", (10, y_offset + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(info_panel, "h: Help", (10, y_offset + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        
        return info_panel
    
    def add_status_overlay(self, frame, results, stats, fps):
        """Add status overlay to frame - same as PC"""
        if results:
            score = results.get('image_score', 0)
            is_anomalous = results.get('is_anomalous', False)
            color = (0, 0, 255) if is_anomalous else (0, 255, 0)
            status = "DEFECT!" if is_anomalous else "OK"
            
            # Main status
            cv2.putText(frame, f"Status: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # FPS in corner
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-100, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def save_frame(self, frame, results):
        """Save frame with timestamp and results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        status = "defect" if results and results.get('is_anomalous', False) else "normal"
        filename = f"jetson_detection_{timestamp}_{status}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"💾 Saved: {filename}")
        return filename
    
    def show_help(self):
        """Show help information with resolution controls"""
        print("\n" + "="*60)
        print("🎮 GLASS Jetson Fabric Detection Controls")
        print("="*60)
        print("q - Quit application")
        print("s - Save current frame")
        print("d - Cycle display modes:")
        print("    0: All Views (2x2 grid)")
        print("    1: Original + Anomaly Map")
        print("    2: Original + Overlay")  
        print("    3: Original Only")
        print("+ / = - Increase processing resolution")
        print("- / _ - Decrease processing resolution")
        print("r - Reset to default resolution (288x288)")
        print("h - Show this help")
        print()
        print("💡 Resolution Control for Jetson:")
        print(f"   Current: {self.current_resolution}x{self.current_resolution}")
        print(f"   Available: {', '.join(map(str, self.available_resolutions))}")
        print("   Higher resolution = better accuracy, more GPU load")
        print("   Lower resolution = faster processing, less GPU load")
        print("   Monitor GPU Load % and Temperature!")
        print("="*60)
    
    def run_headless(self):
        """Run in headless mode (no display)"""
        print("🖥️  Running in headless mode - no GUI display")
        print("   Results will be printed to console")
        print("   Press Ctrl+C to stop")
        
        frame_count = 0
        processing_interval = self.config['performance']['processing_interval']
        last_results = None
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Process frame at intervals
                if frame_count % processing_interval == 0:
                    results, inference_time = self.process_frame(frame)
                    if results:
                        last_results = results
                        
                        # Print results to console
                        score = results.get('image_score', 0)
                        is_anomalous = results.get('is_anomalous', False)
                        status = "DEFECT!" if is_anomalous else "OK"
                        
                        stats = self.get_system_stats()
                        print(f"Frame {frame_count}: {status} (Score: {score:.3f}) | "
                              f"Inference: {inference_time*1000:.1f}ms | "
                              f"CPU: {stats['cpu_percent']:.1f}% | "
                              f"Temp: {stats['temperature_c']:.1f}°C")
                
                frame_count += 1
                
                # Auto-save on defect detection
                if last_results and last_results.get('is_anomalous', False):
                    if frame_count % (processing_interval * 30) == 0:  # Save every 30 detections
                        self.save_frame(frame, last_results)
                
                # Periodic garbage collection
                if time.time() - self.last_gc > 30:
                    torch.cuda.empty_cache()
                    self.last_gc = time.time()
                
                time.sleep(0.01)  # Small delay to prevent overwhelming CPU
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
    
    def run_with_display(self):
        """Run with GUI display - full PC functionality"""
        print("🚀 Starting Jetson fabric detection with complete display...")
        print("Controls: 'q'=quit, 's'=save, 'd'=display mode, 'h'=help overlay")
        
        window_name = self.config['display']['window_name']
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        processing_interval = self.config['performance']['processing_interval']
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame at intervals
            if frame_count % processing_interval == 0:
                results, inference_time = self.process_frame(frame)
                if results:
                    self.last_results = results
            else:
                results = self.last_results
            
            # Get system stats
            stats = self.get_system_stats()
            
            # Update FPS
            current_time = time.time()
            self.fps_counter.append(current_time)
            if len(self.fps_counter) > 1:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            else:
                fps = 0
            
            # Check help auto-hide timer
            if self.show_help and current_time - self.help_start_time > self.help_duration:
                self.show_help = False
            
            # Create display - same as PC
            display_frame = self.create_display_frame(frame, results, stats, fps)
            
            # Show frame with error handling
            try:
                cv2.imshow(window_name, display_frame)
            except Exception as e:
                print(f"Display error: {e}")
                print("Switching to headless mode...")
                cv2.destroyAllWindows()
                return self.run_headless()
            
            # Handle keys - same as PC with resolution controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_frame(frame, results)
            elif key == ord('d'):
                self.display_mode = (self.display_mode + 1) % 4
                modes = ["All Views", "Original+Anomaly", "Original+Overlay", "Original Only"]
                print(f"Display mode: {modes[self.display_mode]}")
            elif key == ord('h'):
                self.show_help = not self.show_help
                if self.show_help:
                    self.help_start_time = time.time()
                print(f"Help display: {'ON' if self.show_help else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.increase_resolution()
            elif key == ord('-') or key == ord('_'):
                self.decrease_resolution()
            elif key == ord('r'):
                print(f"🔄 Reset to default resolution (288x288)")
                self.current_resolution = 288
                self.resolution_index = self.available_resolutions.index(288)
            
            frame_count += 1
            
            # Periodic garbage collection
            if time.time() - self.last_gc > 30:  # Every 30 seconds
                torch.cuda.empty_cache()
                self.last_gc = time.time()
        
        self.cleanup()
    
    def run(self):
        """Main detection loop with display fallback"""
        # Check if we should force headless mode
        force_headless = self.config.get('display', {}).get('headless_mode', False)
        
        if force_headless or self.headless or not self.has_display:
            self.run_headless()
        else:
            try:
                self.run_with_display()
            except Exception as e:
                print(f"Display mode failed: {e}")
                print("Falling back to headless mode...")
                self.run_headless()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()
        print("System stopped")

def main():
    parser = argparse.ArgumentParser(description="GLASS Jetson Fabric Detection - Complete PC Functionality")
    parser.add_argument("--config", default="jetson_config.yaml", help="Config file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--mode", choices=["max_performance", "balanced", "power_save"], 
                       default="balanced", help="Performance mode")
    
    args = parser.parse_args()
    
    try:
        detector = JetsonCompleteFabricDetector(args.config, args.headless)
        detector.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()