"""Interactive dashboard: browse test wafers, inspect predictions and Grad-CAM, compare runs.

Usage:
    streamlit run app.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

from src.gradcam import gradcam, load_checkpoint
from src.data import resize_map

WAFER_CMAP = ListedColormap(["#f4f4f4", "#7fb77e", "#d1495b"])  # off-wafer / pass / fail

st.set_page_config(page_title="Wafer Defect Inspector", layout="wide")
st.title("Wafer Map Defect Inspector")

runs = sorted(p.parent for p in Path("runs").glob("*/best.pt"))
datasets = sorted(Path("data/processed").glob("*.npz"))
if not runs or not datasets:
    st.warning("No trained runs or processed data yet. Run `python -m src.prepare` and `python -m src.train` first.")
    st.stop()

run = st.sidebar.selectbox("Model run", runs, format_func=lambda p: p.name)
data_path = st.sidebar.selectbox("Dataset", datasets, format_func=lambda p: p.name)


@st.cache_resource
def get_model(path):
    return load_checkpoint(path)


@st.cache_data
def get_test(path):
    d = np.load(path)
    return d["X_test"], d["y_test"], list(d["classes"])


model, classes = get_model(str(run / "best.pt"))
X, y, _ = get_test(str(data_path))
size = X.shape[-1]

tab_inspect, tab_runs = st.tabs(["Inspect wafer", "Compare runs"])

with tab_inspect:
    source = st.radio("Source", ["Test set", "Upload .npy"], horizontal=True)
    if source == "Test set":
        cls = st.selectbox("True class", classes)
        idxs = np.flatnonzero(y == classes.index(cls))
        k = st.slider("Sample", 0, len(idxs) - 1, 0) if len(idxs) > 1 else 0
        wafer, truth = X[idxs[k]], cls
    else:
        f = st.file_uploader("2-D wafer map (.npy, values 0/1/2)", type="npy")
        if f is None:
            st.stop()
        wafer, truth = resize_map(np.load(f), size), None

    probs, cam = gradcam(model, wafer)
    pred = classes[int(probs.argmax())]

    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(wafer, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"True: {truth}" if truth else "Uploaded wafer")
        ax.axis("off")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(wafer, cmap="gray", vmin=0, vmax=2, interpolation="nearest")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"Grad-CAM → {pred}")
        ax.axis("off")
        st.pyplot(fig)
    with c3:
        st.metric("Prediction", pred, f"{probs.max():.1%} confidence")
        if truth:
            st.write("✅ Correct" if pred == truth else "❌ Misclassified")
        st.bar_chart(pd.Series(probs, index=classes, name="probability"))

with tab_runs:
    rows = []
    for r in runs:
        m = r / "metrics.json"
        if m.exists():
            j = json.loads(m.read_text())
            rows.append({"run": r.name, "test acc": j["test_accuracy"], "macro-F1": j["test_macro_f1"],
                         "macro-F1 (defects)": j["test_macro_f1_defects_only"]})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("run").style.format("{:.4f}").highlight_max(axis=0))
    cm = run / "confusion_matrix.png"
    if cm.exists():
        st.image(str(cm), caption=f"Normalized confusion matrix — {run.name}")
    rep = run / "report.txt"
    if rep.exists():
        st.code(rep.read_text())
