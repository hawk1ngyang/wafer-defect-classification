"""Generate a small SYNTHETIC wafer dataset to smoke-test the pipeline without the real data.

Results on this data are meaningless - never report them. Use only to check the code runs.

Usage:
    python scripts/make_synthetic.py --n 2000
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASSES  # noqa: E402


def synth(cls, size, rng):
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    r = np.hypot(yy - c, xx - c) / c
    theta = np.arctan2(yy - c, xx - c)
    wafer = r <= 1.0
    fail = rng.random((size, size)) < 0.03
    a = rng.uniform(-np.pi, np.pi)
    if cls == "Center":
        fail |= r < rng.uniform(0.2, 0.35)
    elif cls == "Donut":
        fail |= (r > 0.35) & (r < 0.6)
    elif cls == "Edge-Ring":
        fail |= r > 0.85
    elif cls == "Edge-Loc":
        fail |= (r > 0.75) & (np.abs(np.angle(np.exp(1j * (theta - a)))) < 0.5)
    elif cls == "Loc":
        py, px = c + 0.5 * c * np.sin(a), c + 0.5 * c * np.cos(a)
        fail |= np.hypot(yy - py, xx - px) < size * 0.1
    elif cls == "Near-full":
        fail |= rng.random((size, size)) < 0.8
    elif cls == "Random":
        fail |= rng.random((size, size)) < 0.25
    elif cls == "Scratch":
        d = np.abs((yy - c) * np.cos(a) - (xx - c) * np.sin(a))
        fail |= (d < 1.0) & (r < 0.8)
    m = np.zeros((size, size), np.uint8)
    m[wafer] = 1
    m[wafer & fail] = 2
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--out", default="data/processed/synthetic_64.npz")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    # Imbalanced like the real data: mostly "none".
    probs = np.array([4, 1, 5, 9, 3, 0.5, 1, 1, 75], dtype=float)
    y = rng.choice(len(CLASSES), size=args.n, p=probs / probs.sum())
    X = np.stack([synth(CLASSES[k], args.size, rng) for k in y])
    a, b = int(0.7 * args.n), int(0.8 * args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X_train=X[:a], y_train=y[:a], X_val=X[a:b], y_val=y[a:b],
                        X_test=X[b:], y_test=y[b:], classes=np.array(CLASSES))
    print(f"Saved {args.out} ({args.n} synthetic wafers)")


if __name__ == "__main__":
    main()
