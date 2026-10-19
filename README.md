# Task 2 — Real-Time Computer Vision Optimization

Project: YOLOv8n object detection + multi-object tracking  
Original repository: https://github.com/kimiko-11/real-time-object-detection-tracking

## What this package does

- Preserves a baseline implementation resembling the original `main.py`.
- Benchmarks baseline PyTorch FP32 at 640 px against optimized GPU inference (FP16, where supported) and an optional lower input size.
- Measures model-only inference latency, end-to-end per-frame latency, FPS, model file size, and detection quality on Ultralytics' COCO8 validation sample.
- Exports CSV/JSON results and a plot.
- Includes a report template and report-generation script.

**Important:** No benchmark numbers are pre-filled because they must be measured on the actual Colab/Kaggle runtime. Do not claim an improvement until the scripts have run successfully.

## Recommended runtime

Google Colab or Kaggle Notebook with an NVIDIA GPU enabled. Run cells/commands in order.

## Quick start

```bash
pip install -r requirements.txt
python src/benchmark.py --video "" --device 0 --runs 100 --warmup 20
```

If no video is supplied, the script still benchmarks model inference on the COCO8 validation dataset. For real tracking throughput, supply a short local video:

```bash
python src/benchmark.py --video /path/to/test_video.mp4 --device 0 --runs 100 --warmup 20
```

On Colab/Kaggle, upload a short video or use a publicly usable sample video. Use the exact same video for all variants.

The first run downloads `yolov8n.pt` and COCO8 data if not already available.

## Variants

1. `baseline_fp32_640`: PyTorch FP32, 640 px.
2. `optimized_fp16_640`: PyTorch FP16, 640 px, GPU only.
3. `optimized_fp16_512`: PyTorch FP16, 512 px, GPU only.

The 512 px variant is a combined precision + input-resolution experiment. Compare it with the 640 px FP16 variant to isolate the resolution effect. FP16 is not available on CPU; the script falls back to a skipped result rather than silently claiming FP16.

## Outputs

- `results/benchmark_results.csv`
- `results/benchmark_results.json`
- `results/performance_comparison.png`
- `results/environment.json`

## Interpretation

- FPS is calculated as `1000 / mean latency in milliseconds`.
- Latency includes a warm-up period; CUDA is synchronized around timed inference.
- `model.val()` reports mAP50 and mAP50-95 on COCO8. COCO8 is a small smoke-test dataset, not a statistically strong evaluation set.
- For the best comparison, use a larger fixed validation set and a fixed video representative of your application.
- The included results measure a GPU environment. They do not directly predict webcam FPS on a laptop CPU.
- Exporting to TensorRT can be explored as future work; export/runtime versions and GPU compatibility can affect results.

## Reproducibility checklist

- [ ] Record GPU name and runtime/library versions.
- [ ] Use the same video, dataset, confidence threshold, IoU threshold, and tracker for all variants.
- [ ] Run at least 3 benchmark repeats and report mean and standard deviation.
- [ ] Compare mAP50 / mAP50-95 as well as speed.
- [ ] Keep original source untouched; commit the optimization in a separate branch.


## Measured results included

- Baseline FP32 640: 7.547 ms mean latency, 132.51 derived FPS.
- FP16 640: 8.443 ms mean latency, 118.43 derived FPS.
- FP16 512: 8.651 ms mean latency, 115.59 derived FPS.
- In this run, neither tested FP16 configuration improved latency over the FP32 baseline.
- See report/YOLOv8n_Optimization_Report.pdf for the full measured report and figures.
- Accuracy results are from COCO8, a small smoke-test dataset; validate on a larger fixed dataset before making production claims.
