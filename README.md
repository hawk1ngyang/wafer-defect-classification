# Wafer Map Defect Classification

Classifying wafer-level failure patterns on **WM-811K** — 811K wafer maps from a real semiconductor fab, ~173K of them labeled into 9 classes — with CNNs, under severe class imbalance, with Grad-CAM explanations and an interactive inspection dashboard.

> 📌 Results below are placeholders until training on the full dataset finishes.

## Why it matters

In a fab, the *spatial pattern* of failing dies on a wafer points to the root cause: an **Edge-Ring** suggests an etch or deposition uniformity issue, a **Scratch** points at wafer handling, a **Center** blob at a CMP or spin-coat problem. Classifying patterns automatically lets yield engineers triage thousands of wafers per day and route them to the right tool owner.

## Data

| Class | Share | | Class | Share |
|---|---|---|---|---|
| none | ~85% | | Center | ~2.5% |
| Edge-Ring | ~5.6% | | Scratch | ~0.7% |
| Edge-Loc | ~3.0% | | Random | ~0.5% |
| Loc | ~2.1% | | Donut | ~0.3% |
| | | | Near-full | ~0.1% |

Wafer maps have variable sizes, so they are resized to 64×64 with nearest-neighbour (keeping die states discrete) and one-hot encoded into 3 channels: off-wafer / pass / fail. Split is stratified 70 / 10 / 20.

## Approach

- **Models:** a 4-block baseline CNN, and ResNet-18 with the ImageNet stem replaced for 64×64 inputs
- **Imbalance:** plain CE vs. inverse-sqrt class-weighted CE vs. focal loss
- **Augmentation:** random 90° rotations + flips (defect patterns on a round wafer are rotation-invariant)
- **Selection metric:** macro-F1 on the validation set — accuracy is misleading when 85% of wafers are `none`
- **Explainability:** Grad-CAM on the last conv stage

## Results (test set)

| Run | Accuracy | Macro-F1 | Macro-F1 (defect classes) |
|---|---|---|---|
| cnn_ce | – | – | – |
| cnn_weighted | – | – | – |
| resnet18_weighted | – | – | – |
| resnet18_focal | – | – | – |

<!-- Add runs/<best>/confusion_matrix.png and a dashboard screenshot here. -->

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu
pip install -r requirements.txt

# 1. Data (needs a Kaggle API token in ~/.kaggle/access_token)
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw --unzip
python -m src.prepare --pkl data/raw/LSWMD.pkl --size 64

# 2. Train
python -m src.train --model cnn --loss ce
python -m src.train --model cnn --loss weighted
python -m src.train --model resnet18 --loss weighted --epochs 30
python -m src.train --model resnet18 --loss focal --epochs 30

# 3. Dashboard
streamlit run app.py
```

Quick CPU smoke test without the real data (results are meaningless):

```bash
python scripts/make_synthetic.py --n 2000
python -m src.train --data data/processed/synthetic_64.npz --epochs 2 --workers 0
```

## Project layout

```
src/data.py       loading LSWMD.pkl, resizing, one-hot, dataset + augmentation
src/prepare.py    pkl -> fixed-size .npz with stratified split
src/models.py     baseline CNN, ResNet-18 for 64x64
src/train.py      training loop, weighted / focal loss, AMP, evaluation + confusion matrix
src/gradcam.py    Grad-CAM
app.py            Streamlit dashboard
```
