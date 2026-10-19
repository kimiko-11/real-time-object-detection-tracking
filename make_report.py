"""Create a concise Markdown report from benchmark outputs."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "results"
template = ROOT / "report" / "REPORT_TEMPLATE.md"
csv = results / "benchmark_results.csv"
envfile = results / "environment.json"
if not csv.exists():
    raise SystemExit("Run src/benchmark.py first; benchmark_results.csv not found.")

df = pd.read_csv(csv)
env = json.loads(envfile.read_text()) if envfile.exists() else {}
lines = [
    "# Task 2: YOLOv8n Optimization — Measured Results",
    "",
    f"- Platform: {env.get('platform','not recorded')}",
    f"- Python: {env.get('python','not recorded')}",
    f"- PyTorch: {env.get('torch','not recorded')}",
    f"- Ultralytics: {env.get('ultralytics','not recorded')}",
    f"- GPU: {env.get('gpu') or 'CPU / no CUDA GPU recorded'}",
    f"- Runs: {env.get('runs','not recorded')}; warm-up: {env.get('warmup','not recorded')}",
    "",
    "| Variant | Size | Precision | Mean ms | FPS | mAP50 | mAP50-95 | Model MB | Status |",
    "|---|---:|---|---:|---:|---:|---:|---:|---|"
]
for _, r in df.iterrows():
    lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        r.get("variant","—"), r.get("imgsz","—"), "FP16" if str(r.get("half","")).lower()=="true" else "FP32",
        round(r["latency_mean_ms"],3) if pd.notna(r.get("latency_mean_ms")) else "—",
        round(r["fps_from_mean_latency"],2) if pd.notna(r.get("fps_from_mean_latency")) else "—",
        round(r["coco8_map50"],4) if pd.notna(r.get("coco8_map50")) else "—",
        round(r["coco8_map50_95"],4) if pd.notna(r.get("coco8_map50_95")) else "—",
        r.get("model_size_mb","—"), r.get("status","ok")
    ))
lines += [
    "", "## Interpretation", "",
    "Review the table alongside `performance_comparison.png`. Do not generalize results beyond this GPU/runtime. COCO8 is a small smoke-test dataset; validate the selected variant on a larger fixed dataset before reporting accuracy preservation.",
    "", "## Speedup calculation", "",
    "For each optimized variant with a valid latency: speedup = baseline mean latency / optimized mean latency. Report quality changes as optimized mAP minus baseline mAP.",
]
out = results / "MEASURED_REPORT.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}")
