import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASSES, WaferDataset, resize_map, to_tensor  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import FocalLoss, class_weights  # noqa: E402


def test_resize_keeps_die_states_discrete():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 3, size=(37, 51))
    out = resize_map(m, 64)
    assert out.shape == (64, 64)
    assert set(np.unique(out)) <= {0, 1, 2}


def test_to_tensor_is_one_hot():
    m = np.array([[0, 1], [2, 1]])
    x = to_tensor(m)
    assert x.shape == (3, 2, 2)
    assert torch.equal(x.sum(0), torch.ones(2, 2))
    assert torch.equal(x.argmax(0), torch.as_tensor(m))


def test_augmentation_preserves_die_counts():
    m = np.random.default_rng(1).integers(0, 3, size=(64, 64)).astype(np.uint8)
    ds = WaferDataset([m], np.array([0]), augment=True)
    for _ in range(10):
        x, _ = ds[0]
        assert torch.equal(x.sum(dim=(1, 2)), to_tensor(m).sum(dim=(1, 2)))


def test_focal_with_gamma_zero_equals_cross_entropy():
    logits, y = torch.randn(16, 9), torch.randint(0, 9, (16,))
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), F.cross_entropy(logits, y), atol=1e-6)


def test_focal_downweights_easy_examples():
    y = torch.tensor([0])
    easy, hard = torch.tensor([[5.0, 0, 0]]), torch.tensor([[0.0, 0, 0]])
    ratio_easy = FocalLoss()(easy, y) / F.cross_entropy(easy, y)
    ratio_hard = FocalLoss()(hard, y) / F.cross_entropy(hard, y)
    assert ratio_easy < ratio_hard


def test_class_weights_favour_rare_classes():
    y = np.array([0] * 900 + [1] * 100 + [2] * 1)
    w = class_weights(y, 3)
    assert w[2] > w[1] > w[0]
    assert torch.isclose(w.mean(), torch.tensor(1.0))


def test_models_output_shape():
    x = torch.zeros(2, 3, 64, 64)
    for name in ("cnn", "resnet18"):
        model = build_model(name, len(CLASSES)).eval()
        assert model(x).shape == (2, len(CLASSES))
        assert hasattr(model, "cam_layer")
