"""Post-hoc analysis of a trained run, without retraining:

1. Test-time augmentation (TTA): average softmax over the 8 rotations/flips of each wafer.
   Calibration: temperature scaling fitted on validation, expected calibration error on test.
2. Alarm threshold: flag a wafer as defective when 1 - p(none) > t. The threshold is chosen on the
   VALIDATION set and only then applied to the test set, so the test numbers stay unbiased.
3. Error gallery: the most common test-set confusions, with Grad-CAM of the (wrong) predicted class.

Usage:
    python scripts/analyze.py --run runs/cnn_ce
Writes runs/<run>/analysis.json and docs/<run>_threshold.png, docs/<run>_errors.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import to_tensor  # noqa: E402
from src.gradcam import gradcam, load_checkpoint  # noqa: E402

WAFER_CMAP = ListedColormap(["#f4f4f4", "#7fb77e", "#d1495b"])


def dihedral(x):
    """The 8 rotations/flips of a (N, C, H, W) batch."""
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        yield r
        yield torch.flip(r, dims=(3,))


@torch.no_grad()
def probabilities(model, X, device, tta, batch=512):
    out = []
    for i in range(0, len(X), batch):
        x = torch.stack([to_tensor(m) for m in X[i:i + batch]]).to(device)
        views = list(dihedral(x)) if tta else [x]
        out.append(torch.stack([model(v).softmax(1) for v in views]).mean(0).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def logits_of(model, X, device, batch=512):
    return torch.cat([model(torch.stack([to_tensor(m) for m in X[i:i + batch]]).to(device)).float().cpu()
                      for i in range(0, len(X), batch)])


def fit_temperature(logits, y):
    """Temperature scaling (Guo et al., 2017): one scalar T fitted on validation NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    y = torch.as_tensor(y)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def ece(probs, y, bins=15):
    """Expected calibration error: |confidence - accuracy| averaged over confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(total)


def scores(y, pred, none_idx, n):
    defects = [i for i in range(n) if i != none_idx]
    is_def, flagged = y != none_idx, pred != none_idx
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=range(n), average="macro", zero_division=0)),
        "macro_f1_defects_only": float(f1_score(y, pred, labels=defects, average="macro", zero_division=0)),
        "miss_rate": float((~flagged[is_def]).mean()),          # defective wafers called "none"
        "false_alarm_rate": float(flagged[~is_def].mean()),     # clean wafers flagged as defective
    }


def thresholded(p, t, none_idx):
    """Flag as defective when 1 - p(none) > t; the defect type is then the most likely defect class."""
    q = p.copy()
    q[:, none_idx] = -1
    pred = q.argmax(1)
    pred[1 - p[:, none_idx] <= t] = none_idx
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/cnn_ce")
    ap.add_argument("--data", default="data/processed/wm811k_64.npz")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--errors", type=int, default=4, help="confusion pairs to show")
    args = ap.parse_args()

    run = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_checkpoint(str(run / "best.pt"), device)
    n, none_idx = len(classes), classes.index("none")
    d = np.load(args.data)
    Xv, yv, Xt, yt = d["X_val"], d["y_val"], d["X_test"], d["y_test"]

    res = {"run": run.name}
    # 1. TTA
    pv, pt = probabilities(model, Xv, device, False), probabilities(model, Xt, device, False)
    pv_tta, pt_tta = probabilities(model, Xv, device, True), probabilities(model, Xt, device, True)
    res["test_plain"] = scores(yt, pt.argmax(1), none_idx, n)
    res["test_tta"] = scores(yt, pt_tta.argmax(1), none_idx, n)

    # Calibration: temperature fitted on val, ECE reported on test
    lv, lt = logits_of(model, Xv, device), logits_of(model, Xt, device)
    T = fit_temperature(lv, yv)
    nll = lambda lg, y: float(torch.nn.functional.cross_entropy(lg, torch.as_tensor(y)))
    res["calibration"] = {
        "temperature": T,
        "test_ece_before": ece(lt.softmax(1).numpy(), yt),
        "test_ece_after": ece((lt / T).softmax(1).numpy(), yt),
        "test_nll_before": nll(lt, yt),
        "test_nll_after": nll(lt / T, yt),
    }

    # 2. Threshold sweep (on plain probabilities; chosen on val, reported on test)
    ts = np.round(np.arange(0.02, 0.99, 0.02), 2)
    val_f1 = [f1_score(yv, thresholded(pv, t, none_idx), average="macro") for t in ts]
    t_best = float(ts[int(np.argmax(val_f1))])
    curve = [scores(yt, thresholded(pt, t, none_idx), none_idx, n) for t in ts]
    res["threshold_chosen_on_val"] = t_best
    res["test_at_threshold"] = scores(yt, thresholded(pt, t_best, none_idx), none_idx, n)
    res["test_threshold_curve"] = [{"t": float(t), **c} for t, c in zip(ts, curve)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([c["false_alarm_rate"] * 100 for c in curve], [c["miss_rate"] * 100 for c in curve], "-", color="#3b6ea5")
    for label, s, mk in [("argmax (default)", res["test_plain"], "o"), (f"t={t_best:.2f} (best val macro-F1)", res["test_at_threshold"], "s")]:
        ax.plot(s["false_alarm_rate"] * 100, s["miss_rate"] * 100, mk, ms=8, label=label)
    for t, c in zip(ts, curve):
        if t in (0.1, 0.3, 0.5, 0.7, 0.9):
            ax.annotate(f"t={t:.1f}", (c["false_alarm_rate"] * 100, c["miss_rate"] * 100), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("False alarm rate (% of clean wafers flagged)")
    ax.set_ylabel("Miss rate (% of defective wafers called none)")
    ax.set_title(f"{run.name}: alarm threshold on 1 - p(none), test set", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(args.docs).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(args.docs) / f"{run.name}_threshold.png", dpi=150)
    plt.close(fig)

    # 3. Error gallery: top confusion pairs, 2 examples each, Grad-CAM of the predicted class
    pred = pt.argmax(1)
    wrong = np.flatnonzero(pred != yt)
    pairs = {}
    for i in wrong:
        pairs.setdefault((int(yt[i]), int(pred[i])), []).append(int(i))
    top = sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:args.errors]
    res["top_confusions"] = [{"true": classes[a], "pred": classes[b], "count": len(v)} for (a, b), v in top]

    model_cpu, _ = load_checkpoint(str(run / "best.pt"), "cpu")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2 * len(top), figsize=(2.2 * 2 * len(top), 4.8))
    for j, ((a, b), idxs) in enumerate(top):
        for k, i in enumerate(rng.choice(idxs, size=min(2, len(idxs)), replace=False)):
            col = 2 * j + k
            probs, cam = gradcam(model_cpu, Xt[i], target=b)
            axes[0, col].imshow(Xt[i], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
            axes[0, col].set_title(f"true {classes[a]}\npred {classes[b]} ({probs[b]:.2f})", fontsize=8)
            axes[1, col].imshow(Xt[i], cmap="gray", vmin=0, vmax=2, interpolation="nearest")
            axes[1, col].imshow(cam, cmap="jet", alpha=0.5)
        for ax in axes[:, 2 * j:2 * j + 2].ravel():
            ax.axis("off")
    fig.suptitle(f"Most common test-set errors, {run.name} (Grad-CAM of the predicted class)", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(args.docs) / f"{run.name}_errors.png", dpi=150)
    plt.close(fig)

    (run / "analysis.json").write_text(json.dumps(res, indent=2))
    show = {k: v for k, v in res.items() if k != "test_threshold_curve"}
    print(json.dumps(show, indent=2))


if __name__ == "__main__":
    main()
