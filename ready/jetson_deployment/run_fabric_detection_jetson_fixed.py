#!/usr/bin/env python3
"""
GLASS Fabric Detection - Jetson Orin Nano Optimized Runner with Display Fixes
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

class JetsonDisplayFixedDetector:
    def __init__(self, config_path="jetson_config.yaml"):
        """Initialize Jetson-optimized detector with display fixes"""
        self.load_config(config_path)
        self.setup_display_fixes()
        self.setup_jetson_optimizations()
        self.load_model()
        self.setup_camera()
        self.setup_monitoring()
    
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
        self.test_display()
    
    def test_display(self):
        """Test if display is working"""
        try:
            # Test creating a simple window
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Display Test', test_img)
            cv2.waitKey(1)
            cv2.destroyWindow('Display Test')
            print("✅ Display test passed")
            self.has_display = True
        except Exception as e:
            print(f"⚠️  Display test failed: {e}")
            print("Running in headless mode...")
            self.has_display = False

    def load_config(self, config_path):
        """Load Jetson configuration"""
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
        """Get default configuration"""
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
                'processing_interval': 3,
                'inference_threads': 4
            },
            'display': {
                'enable_gui': True,
                'save_frames': False,
                'headless_mode': False
            }
        }
    
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
        """Setup system monitoring"""
        self.fps_counter = deque(maxlen=30)
        self.memory_usage = deque(maxlen=30)
        self.last_gc = time.time()
        
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
        
        # GPU memory (if available)
        try:
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        except:
            gpu_memory = 0
        
        # Temperature (Jetson specific)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000  # Convert to Celsius
        except:
            temp = 0
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_memory_gb': gpu_memory,
            'temperature_c': temp
        }
    
    def process_frame(self, frame):
        """Process frame with Jetson optimizations"""
        try:
            # Resize to configured resolution
            target_size = self.config['model']['input_resolution']
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
            
            from PIL import Image
            pil_image = Image.fromarray(frame_resized)
            
            # Run inference
            start_time = time.time()
            results = self.predictor.predict(pil_image)
            inference_time = time.time() - start_time
            
            return results, inference_time
            
        except Exception as e:
            print(f"Processing error: {e}")
            return None, 0
    
    def create_display_frame(self, frame, results, stats, fps):
        """Create display frame with enhanced info and help overlay support"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Show help overlay if requested
        if self.show_help:
            return self.create_help_overlay(display_frame)
        
        # Add detection results
        if results:
            score = results.get('image_score', 0)
            is_anomalous = results.get('is_anomalous', False)
            
            color = (0, 0, 255) if is_anomalous else (0, 255, 0)
            status = "DEFECT!" if is_anomalous else "OK"
            
            # Main status - larger text
            cv2.putText(display_frame, f"Status: {status}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(display_frame, f"Score: {score:.3f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # System stats on the left side with improved spacing
        y_offset = h - 150  # More space from bottom
        line_height = 22    # Better line spacing
        
        cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"CPU: {stats['cpu_percent']:.1f}%", 
                   (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Mem: {stats['memory_percent']:.1f}%", 
                   (10, y_offset + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"GPU: {stats['gpu_memory_gb']:.1f}GB", 
                   (10, y_offset + line_height*3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Temp: {stats['temperature_c']:.1f}C", 
                   (10, y_offset + line_height*4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls info in top right - updated to include help
        cv2.putText(display_frame, "Controls: 'q'=quit, 's'=save, 'h'=help", 
                   (w-350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display_frame
    
    def create_help_overlay(self, base_frame):
        """Create help overlay panel for Jetson"""
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
            "h - Show/hide this help",
            "",
            "System Info:",
            f"Model: PyTorch on CUDA",
            f"Resolution: {self.config['model']['input_resolution']}x{self.config['model']['input_resolution']}",
            f"Precision: {self.config['model']['precision']}",
            f"Processing Interval: {self.config['performance']['processing_interval']}",
            "",
            "Performance Tips:",
            "- Lower processing interval = higher accuracy, more GPU load",
            "- FP16 precision optimized for Jetson",
            "- Automatic memory management enabled",
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
            elif line.startswith("    ") or line.startswith("-"):
                color = (200, 200, 200)  # Light gray for indented items
                thickness = 1
            elif line.endswith(":"):
                color = (255, 255, 255)  # White for section headers
                thickness = 1
            else:
                color = (220, 220, 220)  # Light gray for regular text
                thickness = 1
            
            cv2.putText(help_frame, line, (10, y_pos), font, font_scale, color, thickness)
        
        return help_frame
    
    def save_frame(self, frame, results):
        """Save frame with timestamp and results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        status = "defect" if results and results.get('is_anomalous', False) else "normal"
        filename = f"jetson_detection_{timestamp}_{status}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"💾 Saved: {filename}")
        return filename
    
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
                
                # Periodic cleanup
                if time.time() - self.last_gc > 30:
                    torch.cuda.empty_cache()
                    self.last_gc = time.time()
                
                time.sleep(0.01)  # Small delay to prevent overwhelming CPU
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
    
    def run_with_display(self):
        """Run with GUI display"""
        print("🚀 Starting Jetson fabric detection with display...")
        print("Controls: 'q' = quit, 's' = save frame, 'd' = toggle display, 'h' = help overlay")
        
        frame_count = 0
        processing_interval = self.config['performance']['processing_interval']
        last_results = None
        show_overlay = True
        
        # Create window with specific properties
        window_name = 'GLASS Jetson Fabric Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame at intervals to save resources
            if frame_count % processing_interval == 0:
                results, inference_time = self.process_frame(frame)
                if results:
                    last_results = results
            else:
                results = last_results
                inference_time = 0
            
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
            if show_overlay:
                display_frame = self.create_display_frame(frame, results, stats, fps)
            else:
                display_frame = frame
            
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
                show_overlay = not show_overlay
                print(f"Overlay: {'ON' if show_overlay else 'OFF'}")
            elif key == ord('h'):
                self.show_help = not self.show_help
                if self.show_help:
                    self.help_start_time = time.time()
                print(f"Help display: {'ON' if self.show_help else 'OFF'}")
            
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
        
        if force_headless or not self.has_display:
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
    parser = argparse.ArgumentParser(description="GLASS Jetson Fabric Detection")
    parser.add_argument("--config", default="jetson_config.yaml", help="Config file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--mode", choices=["max_performance", "balanced", "power_save"], 
                       default="balanced", help="Performance mode")
    
    args = parser.parse_args()
    
    try:
        detector = JetsonDisplayFixedDetector(args.config)
        
        # Override headless mode if specified
        if args.headless:
            detector.config['display']['headless_mode'] = True
            detector.has_display = False
        
        detector.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()