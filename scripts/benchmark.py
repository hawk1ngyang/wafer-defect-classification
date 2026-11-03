"""Inference throughput and model size for each trained run.

Usage:
    python scripts/benchmark.py                 # all runs, CPU and (if available) GPU
    python scripts/benchmark.py --device cpu
Writes runs/benchmark.json.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gradcam import load_checkpoint  # noqa: E402


@torch.no_grad()
def throughput(model, device, batch, size=64, seconds=5.0):
    x = torch.zeros(batch, 3, size, size, device=device)
    for _ in range(5):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        model(x)
        n += batch
    if device == "cuda":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="runs/benchmark.json")
    args = ap.parse_args()

    devices = ["cpu", "cuda"] if args.device == "all" else [args.device]
    devices = [d for d in devices if d == "cpu" or torch.cuda.is_available()]
    result = {"cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "batch": args.batch, "runs": {}}

    seen = set()
    for ckpt in sorted(Path("runs").glob("*/best.pt")):
        arch = torch.load(ckpt, map_location="cpu")["model"]
        if arch in seen:  # throughput depends on the architecture, not the loss
            continue
        seen.add(arch)
        model, _ = load_checkpoint(str(ckpt))
        entry = {"params_m": sum(p.numel() for p in model.parameters()) / 1e6,
                 "checkpoint_mb": ckpt.stat().st_size / 1e6}
        for dev in devices:
            m = model.to(dev).eval()
            entry[f"wafers_per_s_{dev}"] = throughput(m, dev, args.batch)
            entry[f"wafers_per_s_{dev}_batch1"] = throughput(m, dev, 1, seconds=3.0)
        result["runs"][arch] = entry
        print(arch, json.dumps(entry, indent=1))

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
