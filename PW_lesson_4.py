import os
import cv2
import time
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, 'out')
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
model = YOLO("yolov8n.pt")

CONF_THRESH = 0.5
PERSON_CLASS_ID = 0

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

out_path = os.path.join(OUT_DIR, f"rec_{int(time.time())}.mp4")
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf = CONF_THRESH, verbose=False)

    people_count = 0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == PERSON_CLASS_ID:
                people_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    if people_count > 0:
        cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)
        if not recording:
            print("start")
        recording = True
    else:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        if recording:
            recording = False
    if recording:
        out.write(frame)
    cv2.imshow('YOLO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()