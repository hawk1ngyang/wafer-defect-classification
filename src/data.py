"""WM-811K loading, preprocessing and PyTorch dataset."""
import sys
import types

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _patch_legacy_pandas():
    """LSWMD.pkl was pickled with pandas 0.x; alias the old module paths."""
    import pandas.core.indexes as indexes
    sys.modules.setdefault("pandas.indexes", indexes)
    sys.modules.setdefault("pandas.indexes.base", indexes.base)
    sys.modules.setdefault("pandas.indexes.numeric", types.ModuleType("pandas.indexes.numeric"))
    sys.modules["pandas.indexes.numeric"].Int64Index = indexes.base.Index
    sys.modules.setdefault("pandas.indexes.range", indexes.range)


def load_lswmd(pkl_path):
    """Return (wafer_maps, labels) for the labeled subset of LSWMD.pkl."""
    from pandas.compat.pickle_compat import Unpickler
    _patch_legacy_pandas()
    # Pickled under Python 2: byte strings need latin1, which pd.read_pickle can't pass through.
    with open(pkl_path, "rb") as f:
        df = Unpickler(f, encoding="latin1").load()

    def label_of(x):
        x = np.asarray(x)
        return str(x.ravel()[0]) if x.size else None

    df["label"] = df["failureType"].map(label_of)
    df = df[df["label"].isin(CLASS_TO_IDX)]
    maps = list(df["waferMap"])
    labels = np.array([CLASS_TO_IDX[l] for l in df["label"]], dtype=np.int64)
    return maps, labels


def resize_map(wafer_map, size):
    """Nearest-neighbour resize so die values stay in {0: off-wafer, 1: pass, 2: fail}."""
    img = Image.fromarray(np.asarray(wafer_map, dtype=np.uint8))
    return np.asarray(img.resize((size, size), Image.NEAREST), dtype=np.uint8)


def to_tensor(wafer_map):
    """One-hot encode a (H, W) map into a (3, H, W) float tensor."""
    m = torch.as_tensor(np.asarray(wafer_map), dtype=torch.long).clamp(0, 2)
    return torch.nn.functional.one_hot(m, 3).permute(2, 0, 1).float()


class WaferDataset(Dataset):
    def __init__(self, maps, labels, augment=False):
        self.maps = maps
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = to_tensor(self.maps[i])
        if self.augment:
            # Defect patterns are rotation/flip invariant on a round wafer.
            x = torch.rot90(x, int(torch.randint(4, (1,))), dims=(1, 2))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(2,))
        return x, int(self.labels[i])
