import cv2
import os

video_path = "data/input_video.mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_rate = 2  # frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

interval = int(fps / frame_rate)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % interval == 0:
        filename = f"{output_folder}/frame_{saved:04d}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1

    count += 1

cap.release()
print(f"✅ Extracted {saved} frames")