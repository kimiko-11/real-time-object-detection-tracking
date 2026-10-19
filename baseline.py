"""Baseline YOLOv8n tracking loop based on the original project."""
from collections import defaultdict
import argparse
import cv2
from ultralytics import YOLO

def run(source=0, model_path="yolov8n.pt", imgsz=640, conf=0.25, show=True):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    trajectories = defaultdict(list)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.track(frame, persist=True, imgsz=imgsz, conf=conf, verbose=False)[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            for box, track_id in zip(xyxy, ids):
                x1, y1, x2, y2 = box
                center = (int((x1+x2)/2), int((y1+y2)/2))
                trajectories[track_id].append(center)
                points = trajectories[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (0,255,0), 2)
                if len(points) >= 2:
                    x_prev, y_prev = points[-2]
                    x_curr, y_curr = points[-1]
                    cv2.circle(frame, (x_curr + x_curr-x_prev, y_curr + y_curr-y_prev), 6, (0,0,255), -1)
        annotated = result.plot()
        if show:
            cv2.imshow("Baseline YOLOv8n tracking (ESC to exit)", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()
    run(a.source, a.model, a.imgsz, a.conf, not a.no_show)
