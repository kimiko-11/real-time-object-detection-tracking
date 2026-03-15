from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

trajectories = defaultdict(list)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist=True)

    boxes = results[0].boxes

    if boxes.id is not None:

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for box, track_id in zip(xyxy, ids):

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            trajectories[track_id].append((cx, cy))

            points = trajectories[track_id]

            # draw trajectory
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0,255,0), 2)

            # trajectory prediction
            if len(points) >= 2:

                x_prev, y_prev = points[-2]
                x_curr, y_curr = points[-1]

                vx = x_curr - x_prev
                vy = y_curr - y_prev

                x_future = x_curr + vx
                y_future = y_curr + vy

                cv2.circle(frame, (x_future, y_future), 6, (0,0,255), -1)

    annotated_frame = results[0].plot()

    cv2.imshow("Perception Pipeline", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()