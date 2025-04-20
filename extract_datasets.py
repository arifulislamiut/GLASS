import cv2
import os

# === CONFIG ===
video_path = 'datasets/raw_data/videos/keyboard_good.mp4'  # Your video file path
output_folder = 'datasets/keyboard/train/good'  # Where frames will be saved
frame_interval = 5  # Save every 5th frame (adjust if needed)

# === MAKE FOLDER ===
os.makedirs(output_folder, exist_ok=True)

# === EXTRACT FRAMES ===
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames to {output_folder}")
