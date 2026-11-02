"""Grad-CAM: which region of the wafer drove the prediction."""
import torch
import torch.nn.functional as F

from src.data import to_tensor
from src.models import build_model


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model"], len(ckpt["classes"])).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["classes"]


def gradcam(model, wafer_map, target=None):
    """Return (probabilities, heatmap in [0, 1] at input resolution)."""
    device = next(model.parameters()).device
    x = to_tensor(wafer_map).unsqueeze(0).to(device)
    store = {}
    h1 = model.cam_layer.register_forward_hook(lambda m, i, o: store.update(act=o))
    h2 = model.cam_layer.register_full_backward_hook(lambda m, gi, go: store.update(grad=go[0]))
    try:
        logits = model(x)
        probs = logits.softmax(1)[0]
        target = int(probs.argmax()) if target is None else target
        model.zero_grad()
        logits[0, target].backward()
    finally:
        h1.remove()
        h2.remove()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return probs.detach().cpu().numpy(), cam.detach().cpu().numpy()
