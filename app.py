import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import time
import os
from collections import deque

# ---------------- SETUP ----------------

os.makedirs("output", exist_ok=True)

model = YOLO("yolo26n")  # your model

video_path = "data/input_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output_video.mp4', fourcc, fps, (width, height))

# ---------------- PARAMETERS ----------------

CONF_THRESHOLD = 0.35
MIN_AREA = 2000
MIN_FRAMES = 5
MAX_DISAPPEAR = 30
DIST_THRESHOLD = 50  # centroid match distance

LINE_LEFT = int(width * 0.33)
LINE_RIGHT = int(width * 0.66)

# ---------------- TRACKING ----------------

track_history = {}
smooth_centroids = {}
last_seen = {}
lost_ids = {}

entered_ids = set()
exited_ids = set()
counted_ids = set()
unique_person_ids = set()

person_data = {}

frame_index = 0
heatmap = None
# ---------------- HELPERS ----------------

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box_area = (x2 - x1) * (y2 - y1)
    boxg_area = (x2g - x1g) * (y2g - y1g)

    return inter_area / float(box_area + boxg_area - inter_area + 1e-6)

# ---------------- MAIN LOOP ----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if heatmap is None:
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    current_time = time.time()

    results = model.track(frame, persist=True, conf=CONF_THRESHOLD, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        # ---------------- NMS DUPLICATE FILTER ----------------

        keep = []
        for i in range(len(boxes)):
            duplicate = False
            for j in keep:
                if compute_iou(boxes[i], boxes[j]) > 0.7:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(i)

        boxes = boxes[keep]
        ids = ids[keep]
        classes = classes[keep]

        for box, id, cls in zip(boxes, ids, classes):

            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box)

            if (x2 - x1) * (y2 - y1) < MIN_AREA:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            heatmap[cy,cx] += 10
            # ---------------- CENTROID SMOOTHING ----------------

            if id not in smooth_centroids:
                smooth_centroids[id] = deque(maxlen=5)

            smooth_centroids[id].append((cx, cy))
            cx = int(np.mean([p[0] for p in smooth_centroids[id]]))
            cy = int(np.mean([p[1] for p in smooth_centroids[id]]))

            # ---------------- ID RECOVERY ----------------

            for lost_id, (lx, ly, lf) in list(lost_ids.items()):
                dist = np.hypot(cx - lx, cy - ly)
                if dist < DIST_THRESHOLD and frame_index - lf < MAX_DISAPPEAR:
                    id = lost_id
                    del lost_ids[lost_id]
                    break

            unique_person_ids.add(id)
            last_seen[id] = frame_index

            # ---------------- PERSON DATA ----------------

            if id not in person_data:
                person_data[id] = {
                    "entry_time": current_time,
                    "exit_time": None,
                    "frames": 0
                }

            person_data[id]["frames"] += 1

            # ---------------- TRACK HISTORY ----------------

            if id not in track_history:
                track_history[id] = []

            track_history[id].append((cx, cy))
            if len(track_history[id]) > 10:
                track_history[id].pop(0)

            # ---------------- HYSTERESIS ENTRY/EXIT ----------------
            if id in track_history and len(track_history[id]) >= 2:
                prev_x = track_history[id][-2][0]
                curr_x = track_history[id][-1][0]

                # ENTRY (left → right)
                if prev_x < LINE_LEFT and curr_x > LINE_RIGHT:
                    if id not in counted_ids:
                        entered_ids.add(id)
                        counted_ids.add(id)

                # EXIT (right → left)
                elif prev_x > LINE_RIGHT and curr_x < LINE_LEFT:
                    if id not in counted_ids:
                        exited_ids.add(id)
                        counted_ids.add(id)

            # ---------------- DRAW ----------------

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {id}", (cx - 20, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ---------------- HANDLE DISAPPEAR ----------------

    for id in list(last_seen.keys()):
        if frame_index - last_seen[id] > MAX_DISAPPEAR:
            if id not in exited_ids:
                exited_ids.add(id)

            lost_ids[id] = (*track_history.get(id, [(0,0)])[-1], frame_index)

            if id in person_data:
                person_data[id]["exit_time"] = current_time

            del last_seen[id]

    # ---------------- DRAW LINES ----------------

        # Vertical lines
    cv2.line(frame, (LINE_LEFT, 0), (LINE_LEFT, height), (255, 0, 0), 2)
    cv2.line(frame, (LINE_RIGHT, 0), (LINE_RIGHT, height), (0, 0, 255), 2)
    cv2.putText(frame, "ENTRY", (LINE_LEFT - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, "EXIT", (LINE_RIGHT - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # ---------------- TEXT ----------------

    cv2.putText(frame, f"Entry: {len(entered_ids)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Exit: {len(exited_ids)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Unique: {len(unique_person_ids)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # Heatmap overlay
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
    out.write(overlay)
    cv2.imshow("Final System", overlay)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- SAVE DATA ----------------

detailed_data = []
for id, data in person_data.items():
    if data["exit_time"]:
        duration = data["exit_time"] - data["entry_time"]
    else:
        duration = current_time - data["entry_time"]
    detailed_data.append([id, duration, data["frames"]])

pd.DataFrame(detailed_data, columns=["ID", "Time", "Frames"]).to_csv("output/person_details.csv", index=False)
# Save heatmap image
heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite("output/heatmap.png", heatmap_color)
# Summary report
report_data = {
    "Metric": [
        "Total Unique Customers",
        "Total Entry",
        "Total Exit"
    ],
    "Value": [
        len(unique_person_ids),
        len(entered_ids),
        len(exited_ids)
    ]
}

df_report = pd.DataFrame(report_data)

output_file = "output/reports.csv"
if os.path.exists(output_file):
    os.remove(output_file)

df_report.to_csv(output_file, index=False)
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n✅ FINAL PRODUCTION SYSTEM COMPLETE")