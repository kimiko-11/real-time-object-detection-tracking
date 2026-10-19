# Task 2: YOLOv8n Optimization — Measured Results

- Platform: Linux-6.6.122+-x86_64-with-glibc2.39
- Python: 3.13.15
- PyTorch: 2.11.0+cu130
- Ultralytics: 8.4.182
- GPU: Tesla T4
- Runs: 100; warm-up: 20

| Variant | Size | Precision | Mean ms | FPS | mAP50 | mAP50-95 | Model MB | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| baseline_fp32_640 | 640 | FP32 | 7.547 | 132.51 | 0.8957 | 0.6299 | 6.246 | ok |
| optimized_fp16_640 | 640 | FP16 | 8.443 | 118.43 | 0.8957 | 0.6278 | 6.246 | ok |
| optimized_fp16_512 | 512 | FP16 | 8.651 | 115.59 | 0.8965 | 0.7046 | 6.246 | ok |

## Interpretation

Review the table alongside `performance_comparison.png`. Do not generalize results beyond this GPU/runtime. COCO8 is a small smoke-test dataset; validate the selected variant on a larger fixed dataset before reporting accuracy preservation.

## Speedup calculation

For each optimized variant with a valid latency: speedup = baseline mean latency / optimized mean latency. Report quality changes as optimized mAP minus baseline mAP.