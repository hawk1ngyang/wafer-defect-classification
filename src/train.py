"""Train and evaluate a wafer-map defect classifier.

Usage:
    python -m src.train --data data/processed/wm811k_64.npz --model resnet18 --loss weighted --epochs 30
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data import WaferDataset
from src.models import build_model


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight, self.gamma = weight, gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def class_weights(y, n):
    """Inverse-sqrt frequency: softer than pure inverse, avoids over-boosting tiny classes."""
    counts = np.bincount(y, minlength=n).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/wm811k_64.npz")
    p.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    p.add_argument("--loss", default="weighted", choices=["ce", "weighted", "focal"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data)
    classes = [str(c) for c in d["classes"]]  # plain str so torch.load(weights_only=True) accepts the checkpoint
    n = len(classes)

    name = args.name or f"{args.model}_{args.loss}{'_noaug' if args.no_augment else ''}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    loader = lambda X, y, aug, shuffle: DataLoader(
        WaferDataset(X, y, augment=aug), batch_size=args.batch, shuffle=shuffle,
        num_workers=args.workers, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    train_dl = loader(d["X_train"], d["y_train"], not args.no_augment, True)
    val_dl = loader(d["X_val"], d["y_val"], False, False)
    test_dl = loader(d["X_test"], d["y_test"], False, False)

    model = build_model(args.model, n).to(device)
    w = class_weights(d["y_train"], n).to(device)
    criterion = {"ce": nn.CrossEntropyLoss(),
                 "weighted": nn.CrossEntropyLoss(weight=w),
                 "focal": FocalLoss(weight=w)}[args.loss]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    print(f"device={device} model={args.model} loss={args.loss} train={len(train_dl.dataset)}")
    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)

        pv, tv = predict(model, val_dl, device)
        val_f1 = f1_score(tv, pv, average="macro")
        history.append({"epoch": epoch, "loss": total / len(train_dl.dataset), "val_macro_f1": val_f1})
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": args.model, "classes": classes, "state_dict": model.state_dict()}, out / "best.pt")
            flag = " *"
        print(f"epoch {epoch:>3}  loss {history[-1]['loss']:.4f}  val macro-F1 {val_f1:.4f}  {time.time() - t0:.0f}s{flag}")

    # Final evaluation on the held-out test set with the best checkpoint.
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    pt, tt = predict(model, test_dl, device)
    report = classification_report(tt, pt, labels=range(n), target_names=classes, digits=4, zero_division=0)
    print(report)
    (out / "report.txt").write_text(report)

    metrics = {
        "args": vars(args),
        "test_accuracy": float((pt == tt).mean()),
        "test_macro_f1": float(f1_score(tt, pt, average="macro")),
        "test_macro_f1_defects_only": float(f1_score(tt, pt, labels=[i for i, c in enumerate(classes) if c != "none"], average="macro")),
        "best_val_macro_f1": best_f1,
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(tt, pt, labels=range(n), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(f"{name}  |  test macro-F1 {metrics['test_macro_f1']:.3f}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
