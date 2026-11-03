"""One correctly classified test wafer per defect class, with its Grad-CAM overlay.

Usage:
    python scripts/gradcam_gallery.py --run runs/cnn_ce --out docs/gradcam_gallery.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gradcam import gradcam, load_checkpoint  # noqa: E402

WAFER_CMAP = ListedColormap(["#f4f4f4", "#7fb77e", "#d1495b"])  # off-wafer / pass / fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/cnn_ce")
    p.add_argument("--data", default="data/processed/wm811k_64.npz")
    p.add_argument("--out", default="docs/gradcam_gallery.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, classes = load_checkpoint(str(Path(args.run) / "best.pt"))
    d = np.load(args.data)
    X, y = d["X_test"], d["y_test"]
    rng = np.random.default_rng(args.seed)
    defects = [c for c in classes if c != "none"]

    fig, axes = plt.subplots(2, len(defects), figsize=(2.2 * len(defects), 4.8))
    for j, cls in enumerate(defects):
        ci = classes.index(cls)
        idxs = rng.permutation(np.flatnonzero(y == ci))
        for i in idxs:  # first correctly classified sample
            probs, cam = gradcam(model, X[i])
            if int(probs.argmax()) == ci:
                break
        axes[0, j].imshow(X[i], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[0, j].set_title(f"{cls}\np={probs[ci]:.2f}", fontsize=9)
        axes[1, j].imshow(X[i], cmap="gray", vmin=0, vmax=2, interpolation="nearest")
        axes[1, j].imshow(cam, cmap="jet", alpha=0.5)
        for ax in axes[:, j]:
            ax.axis("off")
    axes[0, 0].text(-8, 32, "wafer", rotation=90, va="center", fontsize=9)
    axes[1, 0].text(-8, 32, "Grad-CAM", rotation=90, va="center", fontsize=9)
    fig.suptitle(f"Grad-CAM, {Path(args.run).name} (test set)", fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
