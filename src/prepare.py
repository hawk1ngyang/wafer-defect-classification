"""Convert LSWMD.pkl into a compact, fixed-size .npz with a stratified split.

Usage:
    python -m src.prepare --pkl data/raw/LSWMD.pkl --size 64
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data import CLASSES, load_lswmd, resize_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default="data/raw/LSWMD.pkl")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None, help="subsample for quick CPU runs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Loading {args.pkl} ...")
    maps, labels = load_lswmd(args.pkl)
    idx = np.arange(len(labels))
    if args.limit and args.limit < len(idx):
        idx, _ = train_test_split(idx, train_size=args.limit, stratify=labels, random_state=args.seed)

    X = np.stack([resize_map(maps[i], args.size) for i in tqdm(idx, desc="resize")])
    y = labels[idx]

    # 70 / 10 / 20 stratified split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=args.seed)
    tr, va = train_test_split(tr, test_size=0.125, stratify=y[tr], random_state=args.seed)

    out = Path(args.out or f"data/processed/wm811k_{args.size}{'_' + str(args.limit) if args.limit else ''}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X_train=X[tr], y_train=y[tr], X_val=X[va], y_val=y[va],
                        X_test=X[te], y_test=y[te], classes=np.array(CLASSES))

    print(f"Saved {out}  train={len(tr)} val={len(va)} test={len(te)}")
    for c, n in zip(CLASSES, np.bincount(y, minlength=len(CLASSES))):
        print(f"  {c:<10} {n:>7}  ({n / len(y):.2%})")


if __name__ == "__main__":
    main()
