from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

def detect_and_track(frame):
    results = model.track(frame, persist=True)

    if results[0].boxes.id is None:
        return [], [], []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    return boxes, ids, classes