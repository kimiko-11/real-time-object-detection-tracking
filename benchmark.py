"""Reproducible YOLOv8n speed + accuracy benchmark.

Run from the package root:
python src/benchmark.py --device 0 --runs 100 --warmup 20
python src/benchmark.py --video /path/to/clip.mp4 --device 0 --runs 100
"""
import argparse, json, os, platform, statistics, time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def sync(device):
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_inference(model, imgsz, half, device, runs, warmup):
    # Use a fixed in-memory image to isolate model pipeline latency from disk/camera I/O.
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(warmup):
        _ = model.predict(image, imgsz=imgsz, device=device, half=half, verbose=False)
    sync(device)
    samples = []
    for _ in range(runs):
        sync(device)
        start = time.perf_counter()
        _ = model.predict(image, imgsz=imgsz, device=device, half=half, verbose=False)
        sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_mean_ms": float(np.mean(samples)),
        "latency_median_ms": float(np.median(samples)),
        "latency_std_ms": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "fps_from_mean_latency": float(1000.0 / np.mean(samples)),
    }

def video_benchmark(model, video, imgsz, half, device, runs):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    frame_times, processed = [], 0
    while processed < runs:
        ok, frame = cap.read()
        if not ok:
            break
        sync(device)
        start = time.perf_counter()
        _ = model.track(frame, persist=True, imgsz=imgsz, device=device,
                        half=half, verbose=False)
        sync(device)
        frame_times.append((time.perf_counter()-start)*1000)
        processed += 1
    cap.release()
    if not frame_times:
        return {}
    return {
        "video_frames": len(frame_times),
        "video_latency_mean_ms": float(np.mean(frame_times)),
        "video_latency_std_ms": float(np.std(frame_times, ddof=1)) if len(frame_times)>1 else 0.0,
        "video_fps": float(1000/np.mean(frame_times)),
    }

def validate(model, imgsz, half, device):
    # COCO8 is a tiny smoke-test validation set; replace with a larger fixed validation set for stronger evidence.
    metrics = model.val(data="coco8.yaml", imgsz=imgsz, device=device, half=half,
                        plots=False, verbose=False)
    box = metrics.box
    return {
        "coco8_precision": float(box.mp),
        "coco8_recall": float(box.mr),
        "coco8_map50": float(box.map50),
        "coco8_map50_95": float(box.map),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--video", default="")
    p.add_argument("--skip-validation", action="store_true")
    a = p.parse_args()

    is_gpu = str(a.device) != "cpu" and torch.cuda.is_available()
    variants = [
        ("baseline_fp32_640", 640, False),
        ("optimized_fp16_640", 640, True),
        ("optimized_fp16_512", 512, True),
    ]
    rows = []
    for name, imgsz, half in variants:
        if half and not is_gpu:
            rows.append({"variant":name, "status":"skipped", "reason":"FP16 GPU benchmark requires CUDA"})
            continue
        print(f"\n=== {name}: imgsz={imgsz}, half={half} ===")
        model = YOLO(a.model)
        row = {"variant":name, "status":"ok", "imgsz":imgsz, "half":half,
               "device":a.device, "model":a.model,
               "model_size_mb": round(os.path.getsize(a.model)/(1024*1024), 3) if os.path.exists(a.model) else None}
        row.update(timed_inference(model, imgsz, half, a.device, a.runs, a.warmup))
        if a.video:
            row.update(video_benchmark(model, a.video, imgsz, half, a.device, a.runs))
        if not a.skip_validation:
            try:
                row.update(validate(model, imgsz, half, a.device))
            except Exception as e:
                row["validation_error"] = str(e)
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    csv_path = OUT / "benchmark_results.csv"
    json_path = OUT / "benchmark_results.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_argument": a.device,
        "runs": a.runs, "warmup": a.warmup, "video": a.video or None
    }
    (OUT/"environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    valid = df[df["status"] == "ok"] if "status" in df else df
    if len(valid) and "latency_mean_ms" in valid:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9,5))
        plt.bar(valid["variant"], valid["latency_mean_ms"])
        plt.ylabel("Mean inference latency (ms)")
        plt.title("YOLOv8n inference latency comparison")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(OUT/"performance_comparison.png", dpi=160)
        plt.close()
    print(f"\nSaved results to {OUT}")

if __name__ == "__main__":
    main()
