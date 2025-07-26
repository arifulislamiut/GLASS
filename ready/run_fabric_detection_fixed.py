#!/usr/bin/env python3
"""
Enhanced Fabric Defect Detection System with Display Fixes
GLASS Anomaly Detection with Camera Interface and Headless Support

Usage: 
  python run_fabric_detection_fixed.py                # GUI mode
  python run_fabric_detection_fixed.py --headless     # Headless mode
  python run_fabric_detection_fixed.py --diagnose     # Display diagnostic
"""

import sys
import os
import argparse
import time
import yaml
import psutil
from collections import deque

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import cv2
import numpy as np

class DisplayFixedFabricDetector:
    def __init__(self, camera_id=0, headless=False, config_file=None):
        """Initialize fabric defect detector with display fixes"""
        self.headless = headless
        self.config = self.load_config(config_file)
        
        if not headless:
            self.setup_display_fixes()
        
        self.load_model()
        self.setup_camera(camera_id)
        self.setup_monitoring()
        
        print("✓ System ready!")
    
    def load_config(self, config_file=None):
        """Load configuration with defaults"""
        default_config = {
            'model': {
                'backend': 'pytorch',
                'device': 'cuda' if self.check_cuda() else 'cpu',
                'model_path': 'models/wfdd_grid_cloth'
            },
            'camera': {
                'width': 640,
                'height': 480,
                'fps': 30,
                'buffer_size': 1
            },
            'display': {
                'enabled': True,
                'headless_mode': False,
                'fallback_to_headless': True,
                'window_name': 'GLASS Fabric Detection',
                'save_frames': False
            },
            'performance': {
                'processing_interval': 1,
                'target_fps': 30
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Merge configs
                self.deep_update(default_config, user_config)
            except Exception as e:
                print(f"⚠️  Config load failed: {e}, using defaults")
        
        return default_config
    
    def deep_update(self, base_dict, update_dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self.deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def check_cuda(self):
        """Check CUDA availability"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def setup_display_fixes(self):
        """Setup display fixes for various environments"""
        print("🔧 Setting up display fixes...")
        
        # Set display environment variables if not set
        if 'DISPLAY' not in os.environ:
            os.environ['DISPLAY'] = ':0'
            print(f"Set DISPLAY to {os.environ['DISPLAY']}")
        
        # Set Qt backend to avoid conflicts
        os.environ['QT_X11_NO_MITSHM'] = '1'
        
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
            return False
    
    def load_model(self):
        """Load the GLASS model"""
        model_path = self.config['model']['model_path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check for available models
        onnx_files = [f for f in os.listdir(model_path) if f.endswith('.onnx')]
        pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        
        if self.config['model']['backend'] == 'onnx' and onnx_files:
            model_type = 'onnx'
            print(f"✓ Using ONNX model: {onnx_files[0]}")
        elif pth_files:
            model_type = 'pytorch'
            print(f"✓ Using PyTorch model: {pth_files[0]}")
        else:
            raise FileNotFoundError("No compatible model files found")
        
        print("Loading GLASS model... (this may take 30-60 seconds)")
        
        from predict_single_image import GLASSPredictor
        device = self.config['model']['device']
        
        self.predictor = GLASSPredictor(model_path, model_type, device)
        print(f"✓ Model loaded successfully on {device}!")
    
    def setup_camera(self, camera_id):
        """Setup camera with enhanced initialization"""
        cam_config = self.config['camera']
        
        print(f"🔍 Setting up camera {camera_id}...")
        
        # Try different camera initialization methods in order of preference
        camera_methods = []
        
        if os.name == 'posix':  # Linux/macOS
            camera_methods = [
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_ANY, "Default"),
                (cv2.CAP_GSTREAMER, "GStreamer")
            ]
        elif os.name == 'nt':  # Windows
            camera_methods = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Default")
            ]
        else:
            camera_methods = [(cv2.CAP_ANY, "Default")]
        
        camera_initialized = False
        working_method = None
        
        for backend, method_name in camera_methods:
            try:
                print(f"   Trying {method_name}...")
                self.cap = cv2.VideoCapture(camera_id, backend)
                
                if self.cap.isOpened():
                    # Test if we can actually read frames
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        camera_initialized = True
                        working_method = method_name
                        print(f"✅ Camera opened with {method_name}: {test_frame.shape}")
                        break
                    else:
                        print(f"   {method_name}: Opened but no frames")
                        self.cap.release()
                else:
                    print(f"   {method_name}: Failed to open")
                    
            except Exception as e:
                print(f"   {method_name}: Exception - {e}")
                try:
                    self.cap.release()
                except:
                    pass
        
        if not camera_initialized:
            raise RuntimeError(f"❌ Could not initialize camera {camera_id} with any method")
        
        print(f"✅ Using {working_method} for camera access")
        
        # Configure camera settings
        original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Original settings: {original_width}x{original_height} @ {original_fps:.1f}FPS")
        
        # Set desired settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['height'])
        self.cap.set(cv2.CAP_PROP_FPS, cam_config['fps'])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cam_config['buffer_size'])
        
        # Verify actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Configured to: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")
        
        # Final test
        for i in range(3):  # Try a few frames to ensure stability
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                if i == 2:  # Last test
                    h, w = test_frame.shape[:2]
                    print(f"✅ Camera ready: {w}x{h}")
                    return
            else:
                print(f"   Frame test {i+1}/3 failed")
        
        # If we get here, camera is unstable
        print("⚠️  Camera seems unstable but will proceed")
    
    def setup_monitoring(self):
        """Setup performance monitoring"""
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.last_results = None
        self.display_mode = 0  # 0=all, 1=original+anomaly, 2=original+overlay, 3=original only
        self.show_overlay = True
        
        # Help display state
        self.show_help = False
        self.help_start_time = 0
        self.help_duration = 5.0  # Show help for 5 seconds
        
        # Runtime resolution control
        self.available_resolutions = [224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
        self.current_resolution = 288  # Default
        self.resolution_index = self.available_resolutions.index(self.current_resolution)
    
    def get_system_stats(self):
        """Get system performance statistics"""
        try:
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                stats = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / 1024**3
                }
            except ImportError:
                # Fallback when psutil is not available
                print("⚠️  psutil not available, using basic stats")
                stats = {
                    'cpu_percent': 0,
                    'memory_percent': 0,
                    'memory_available_gb': 0
                }
            
            # GPU memory and utilization if available
            try:
                import torch
                if torch.cuda.is_available():
                    stats['gpu_memory_gb'] = torch.cuda.memory_allocated() / 1024**3
                    stats['gpu_memory_cached_gb'] = torch.cuda.memory_reserved() / 1024**3
                    
                    # Try to get GPU utilization
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        stats['gpu_utilization'] = utilization.gpu
                        stats['gpu_memory_utilization'] = utilization.memory
                    except:
                        stats['gpu_utilization'] = 0
                        stats['gpu_memory_utilization'] = 0
            except:
                stats['gpu_memory_gb'] = 0
                stats['gpu_memory_cached_gb'] = 0
                stats['gpu_utilization'] = 0
                stats['gpu_memory_utilization'] = 0
            
            return stats
            
        except Exception as e:
            print(f"Stats error: {e}")
            return {'cpu_percent': 0, 'memory_percent': 0, 'memory_available_gb': 0, 'gpu_memory_gb': 0}
    
    def process_frame(self, frame):
        """Process frame for defect detection with runtime resolution"""
        try:
            # Convert and resize to current resolution
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.current_resolution, self.current_resolution))
            
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
    
    def increase_resolution(self):
        """Increase processing resolution"""
        if self.resolution_index < len(self.available_resolutions) - 1:
            self.resolution_index += 1
            self.current_resolution = self.available_resolutions[self.resolution_index]
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
            print(f"🔍 Resolution decreased to {self.current_resolution}x{self.current_resolution}")
            return True
        else:
            print(f"⚠️  Already at minimum resolution ({self.current_resolution}x{self.current_resolution})")
            return False
    
    def create_display_frame(self, frame, results, stats, fps):
        """Create enhanced display frame"""
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
            # Standard multi-panel display
            display_width = 960
            display_height = 540
            panel_width = display_width // 2  # 480
            panel_height = display_height // 2  # 270 - this was the bug!
            
            # Always resize main frame consistently
            main_frame = cv2.resize(frame, (panel_width, panel_height))
            
            if results is None:
                # Show only main frame when no results
                display_frame = cv2.resize(main_frame, (display_width, display_height))
                cv2.putText(display_frame, "Processing...", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                return display_frame
            
            # Create panels based on display mode
            if self.display_mode == 0:  # All views
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
                    
                    # Info panel
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
        """Create anomaly map visualization panel"""
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
        """Create overlay panel with detection results"""
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
        """Create help overlay panel"""
        help_frame = base_frame.copy()
        
        # Create semi-transparent dark overlay
        overlay = np.zeros_like(help_frame)
        help_frame = cv2.addWeighted(help_frame, 0.3, overlay, 0.7, 0)
        
        # Help text content
        help_lines = [
            "GLASS Fabric Detection - Help",
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
            if line == "GLASS Fabric Detection - Help":
                color = (0, 255, 255)  # Yellow for title
                thickness = 2
            elif line.startswith("    "):
                color = (200, 200, 200)  # Light gray for indented items
                thickness = 1
            elif line in ["Controls:", "Current Resolution:", "Display Mode:"]:
                color = (255, 255, 255)  # White for section headers
                thickness = 1
            else:
                color = (220, 220, 220)  # Light gray for regular text
                thickness = 1
            
            cv2.putText(help_frame, line, (10, y_pos), font, font_scale, color, thickness)
        
        return help_frame
    
    def create_info_panel(self, results, stats, fps, width, height):
        """Create information panel"""
        info_panel = np.zeros((height, width, 3), dtype=np.uint8) + 20
        
        # Title
        cv2.putText(info_panel, 'System Info', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 60
        line_height = 25
        
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
        
        # Update y_offset to account for GPU info section
        y_offset += line_height * 3  # Account for FPS, CPU, Memory
        
        if stats.get('gpu_memory_gb', 0) > 0:
            cv2.putText(info_panel, f"GPU: {stats['gpu_memory_gb']:.1f}GB", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # GPU utilization if available
            if stats.get('gpu_utilization', 0) > 0:
                cv2.putText(info_panel, f"GPU Load: {stats['gpu_utilization']:.0f}%", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height
        
        # Processing resolution with proper spacing
        cv2.putText(info_panel, f"Resolution: {self.current_resolution}x{self.current_resolution}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        y_offset += line_height
        
        # Processing time with proper spacing
        if self.processing_times:
            avg_time = np.mean(list(self.processing_times)) * 1000  # ms
            cv2.putText(info_panel, f"Inference: {avg_time:.1f}ms", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls text removed from info panel - now displayed on main camera area
        
        return info_panel
    
    def add_status_overlay(self, frame, results, stats, fps):
        """Add status overlay to frame"""
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
        filename = f"detection_{timestamp}_{status}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"💾 Saved: {filename}")
        return filename
    
    def run_headless(self):
        """Run in headless mode without GUI"""
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
                              f"Memory: {stats['memory_percent']:.1f}%")
                
                frame_count += 1
                
                # Auto-save on defect detection
                if last_results and last_results.get('is_anomalous', False):
                    if frame_count % (processing_interval * 30) == 0:  # Save every 30 detections
                        self.save_frame(frame, last_results)
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
    
    def run_with_display(self):
        """Run with GUI display"""
        print("🚀 Starting fabric detection with enhanced display...")
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
            
            # Create display
            display_frame = self.create_display_frame(frame, results, stats, fps)
            
            # Show frame with error handling
            try:
                cv2.imshow(window_name, display_frame)
            except Exception as e:
                print(f"Display error: {e}")
                print("Switching to headless mode...")
                cv2.destroyAllWindows()
                return self.run_headless()
            
            # Handle keys
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
        
        self.cleanup()
    
    def print_help_to_console(self):
        """Print help information to console"""
        print("\n" + "="*60)
        print("🎮 GLASS Fabric Detection Controls")
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
        print("h - Show/hide help overlay")
        print()
        print("💡 Resolution Control:")
        print(f"   Current: {self.current_resolution}x{self.current_resolution}")
        print(f"   Available: {', '.join(map(str, self.available_resolutions))}")
        print("   Higher resolution = better accuracy, more GPU load")
        print("   Lower resolution = faster processing, less GPU load")
        print("="*60)
    
    def run(self):
        """Main detection loop with display fallback"""
        if self.headless:
            self.run_headless()
        else:
            try:
                self.run_with_display()
            except Exception as e:
                print(f"Display mode failed: {e}")
                if self.config['display']['fallback_to_headless']:
                    print("Falling back to headless mode...")
                    self.run_headless()
                else:
                    raise
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped")

def diagnose_display():
    """Run display diagnostic"""
    print("🔍 GLASS PC Display Diagnostic")
    print("="*50)
    
    # Check display environment
    display = os.environ.get('DISPLAY', None)
    print(f"DISPLAY: {display if display else 'Not set'}")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test display
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Test', test_img)
            cv2.waitKey(100)
            cv2.destroyWindow('Test')
            print("✅ OpenCV display working")
        except Exception as e:
            print(f"❌ OpenCV display failed: {e}")
            
    except ImportError:
        print("❌ OpenCV not available")
    
    # Check camera
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera working: {frame.shape}")
            else:
                print("❌ Camera not capturing")
            cap.release()
        else:
            print("❌ Camera not accessible")
    except Exception as e:
        print(f"❌ Camera error: {e}")
    
    print("="*50)
    print("Solutions:")
    print("1. For headless: use --headless flag")
    print("2. For SSH: use ssh -X for X11 forwarding")
    print("3. For Windows: ensure camera permissions")
    print("4. For Linux: check /dev/video* permissions")

def main():
    parser = argparse.ArgumentParser(description="Enhanced GLASS Fabric Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--diagnose", action="store_true", help="Run display diagnostic")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_display()
        return
    
    try:
        detector = DisplayFixedFabricDetector(
            camera_id=args.camera,
            headless=args.headless,
            config_file=args.config
        )
        detector.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()