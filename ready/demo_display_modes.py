#!/usr/bin/env python3
"""
Demo script showing the improved display modes
"""
import cv2
import numpy as np
import time

def create_demo_frame():
    """Create a demo frame with fabric texture"""
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    
    # Create fabric-like texture
    for i in range(0, frame.shape[0], 20):
        cv2.line(frame, (0, i), (frame.shape[1], i), (100, 100, 120), 2)
    for j in range(0, frame.shape[1], 20):
        cv2.line(frame, (j, 0), (j, frame.shape[0]), (100, 100, 120), 2)
    
    # Add some "defects"
    cv2.circle(frame, (200, 150), 30, (0, 0, 255), -1)  # Red defect
    cv2.ellipse(frame, (350, 250), (40, 20), 45, 0, 360, (255, 0, 0), -1)  # Blue defect
    
    return frame

def create_demo_anomaly_map():
    """Create a demo anomaly map"""
    anomaly_map = np.zeros((360, 480), dtype=np.float32)
    
    # High anomaly areas where defects are
    cv2.circle(anomaly_map, (200, 150), 30, 1.0, -1)
    cv2.ellipse(anomaly_map, (350, 250), (40, 20), 45, 0, 360, 0.8, -1)
    
    # Add some noise
    noise = np.random.random((360, 480)) * 0.3
    anomaly_map = np.clip(anomaly_map + noise, 0, 1)
    
    return anomaly_map

def demo_display_modes():
    """Demo the different display modes"""
    print("Fabric Defect Detection - Display Modes Demo")
    print("=" * 50)
    print("This demo shows the 4 different display modes:")
    print("1. All Views (2x2 grid)")
    print("2. Original + Anomaly Map (side by side)")
    print("3. Original + Overlay (side by side)")
    print("4. Original Only")
    print("\nPress SPACE to cycle through modes, 'q' to quit")
    print("=" * 50)
    
    # Create demo data
    demo_frame = create_demo_frame()
    demo_anomaly = create_demo_anomaly_map()
    
    # Apply colormap to anomaly
    anomaly_colored = cv2.applyColorMap((demo_anomaly * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Create overlay
    overlay = cv2.addWeighted(demo_frame, 0.6, anomaly_colored, 0.4, 0)
    
    display_mode = 0
    mode_names = ["All Views", "Original + Anomaly", "Original + Overlay", "Original Only"]
    
    while True:
        # Create display based on mode
        if display_mode == 0:  # All views
            top_row = np.hstack([demo_frame, anomaly_colored])
            bottom_row = np.hstack([overlay, np.zeros_like(demo_frame)])
            combined = np.vstack([top_row, bottom_row])
            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Anomaly Map", (490, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Overlay", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif display_mode == 1:  # Original + Anomaly
            combined = np.hstack([demo_frame, anomaly_colored])
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Anomaly Map", (490, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif display_mode == 2:  # Original + Overlay
            combined = np.hstack([demo_frame, overlay])
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Overlay", (490, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:  # Original only
            combined = demo_frame.copy()
            cv2.putText(combined, "Live Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add mode info
        cv2.putText(combined, f"Mode {display_mode + 1}: {mode_names[display_mode]}", 
                   (10, combined.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(combined, "SPACE = Next Mode, Q = Quit", 
                   (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show demo detection results
        cv2.putText(combined, "DEFECTS DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(combined, "Score: 0.85", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Display Modes Demo', combined)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to cycle modes
            display_mode = (display_mode + 1) % 4
            print(f"Switched to Mode {display_mode + 1}: {mode_names[display_mode]}")
        
        time.sleep(0.1)  # Slow down for demo
    
    cv2.destroyAllWindows()
    print("Demo completed!")

if __name__ == "__main__":
    demo_display_modes()