#!/usr/bin/env python3
"""
Create a video from all images in the WFDD dataset
Organizes images by category and defect type for comprehensive visualization
"""

import cv2
import os
import numpy as np
from pathlib import Path
import argparse

def get_all_images(dataset_path):
    """Get all images from the WFDD dataset organized by category"""
    images = []
    
    # Define the categories and their structure
    categories = {
        # 'grey_cloth': {
        #     'train': ['good'],
        #     'test': ['good', 'flecked', 'line', 'string', 'contaminated']
        # },
        'grid_cloth': {
            'train': ['good'],
            'test': ['good', 'defect']
        },
        # 'pink_flower': {
        #     'train': ['good'],
        #     'test': ['good', 'defect']
        # },
        # 'yellow_cloth': {
        #     'train': ['good'],
        #     'test': ['good', 'defect']
        # }
    }
    
    for category, splits in categories.items():
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            continue
            
        for split, defect_types in splits.items():
            split_path = os.path.join(category_path, split)
            if not os.path.exists(split_path):
                continue
                
            for defect_type in defect_types:
                defect_path = os.path.join(split_path, defect_type)
                if not os.path.exists(defect_path):
                    continue
                    
                # Get all PNG files
                png_files = [f for f in os.listdir(defect_path) if f.endswith('.png')]
                png_files.sort()  # Sort for consistent ordering
                
                for png_file in png_files:
                    image_path = os.path.join(defect_path, png_file)
                    images.append({
                        'path': image_path,
                        'category': category,
                        'split': split,
                        'defect_type': defect_type,
                        'filename': png_file
                    })
    
    return images

def resize_image(image, target_size=(640, 480)):
    """Resize image maintaining aspect ratio with padding"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Place resized image in center
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    # Debug print
    # print(f"Resized shape: {resized.shape}, Padded shape: {padded.shape}, Target: {target_size}")
    return padded

def add_text_overlay(image, text, position=(10, 30), color=(255, 255, 255), scale=0.7):
    """Add text overlay to image"""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return image

def create_video(images, output_path, fps=2, frame_size=(1280, 720)):
    """Create video from list of images"""
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    # Group images by category and defect type
    current_category = None
    current_defect = None
    
    for i, img_info in enumerate(images):
        print(f"Processing {i+1}/{len(images)}: {img_info['category']}/{img_info['split']}/{img_info['defect_type']}/{img_info['filename']}")
        
        # Load image
        image = cv2.imread(img_info['path'])
        if image is None:
            print(f"Warning: Could not load {img_info['path']}")
            continue
        
        # Resize image to exact size needed
        resized = resize_image(image, (frame_size[0]//2, frame_size[1]))
        
        # Create frame with two images side by side
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        # Place image on left side
        frame[:, :frame_size[0]//2] = resized
        
        # Add text overlay
        category_text = f"Category: {img_info['category']}"
        split_text = f"Split: {img_info['split']}"
        defect_text = f"Type: {img_info['defect_type']}"
        filename_text = f"File: {img_info['filename']}"
        
        # Add text to left side
        add_text_overlay(frame, category_text, (10, 30), (255, 255, 255))
        add_text_overlay(frame, split_text, (10, 60), (255, 255, 255))
        add_text_overlay(frame, defect_text, (10, 90), (255, 255, 255))
        add_text_overlay(frame, filename_text, (10, 120), (255, 255, 255))
        
        # Add progress info
        progress_text = f"Progress: {i+1}/{len(images)}"
        add_text_overlay(frame, progress_text, (10, frame_size[1]-30), (0, 255, 0))
        
        # If we have a next image, show it on the right side
        if i + 1 < len(images):
            next_img_info = images[i + 1]
            next_image = cv2.imread(next_img_info['path'])
            if next_image is not None:
                next_resized = resize_image(next_image, (frame_size[0]//2, frame_size[1]))
                frame[:, frame_size[0]//2:] = next_resized
                
                # Add text for next image
                next_category_text = f"Next: {next_img_info['category']}"
                next_defect_text = f"Next Type: {next_img_info['defect_type']}"
                add_text_overlay(frame, next_category_text, (frame_size[0]//2 + 10, 30), (255, 255, 255))
                add_text_overlay(frame, next_defect_text, (frame_size[0]//2 + 10, 60), (255, 255, 255))
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create video from WFDD dataset images")
    parser.add_argument("--dataset_path", type=str, default="datasets/WFDD",
                       help="Path to WFDD dataset")
    parser.add_argument("--output", type=str, default="wfdd_dataset_video.mp4",
                       help="Output video path")
    parser.add_argument("--fps", type=int, default=2,
                       help="Frames per second")
    parser.add_argument("--width", type=int, default=1280,
                       help="Video width")
    parser.add_argument("--height", type=int, default=720,
                       help="Video height")
    
    args = parser.parse_args()
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} not found")
        return
    
    # Get all images
    print("Scanning WFDD dataset...")
    images = get_all_images(args.dataset_path)
    
    if not images:
        print("No images found in the dataset")
        return
    
    print(f"Found {len(images)} images")
    
    # Print summary
    categories = {}
    for img in images:
        cat_key = f"{img['category']}_{img['split']}_{img['defect_type']}"
        if cat_key not in categories:
            categories[cat_key] = 0
        categories[cat_key] += 1
    
    print("\nDataset Summary:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} images")
    
    # Create video
    print(f"\nCreating video with {len(images)} images...")
    create_video(images, args.output, args.fps, (args.width, args.height))
    
    print("Done!")

if __name__ == "__main__":
    main() 