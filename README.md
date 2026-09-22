# A3 — Noisy Microscope: CNN Classification & GRU Restoration


## Contents

- `A3_BloodMNIST.ipynb` — completed notebook (code + written answers)
- `prepare_data.py` — supplied script that builds the BloodMNIST subset (`data/blood_mnist_subset.npz`)

## What's inside the notebook

1. **Data exploration** — BloodMNIST subset (3200 train / 640 val / 800 test, 8 cell classes)
2. **CNN classifier** — 2 conv blocks (BatchNorm + ReLU + MaxPool) + FC head, manual parameter count (422,152 params), trained 30 epochs, **90.1% test accuracy** (vs 12.5% random baseline)
3. **Confusion matrix** — 8×8, main confusion cluster: Basophil / Monocyte / Immature granulocytes
4. **Noisy acquisition** — Gaussian noise (σ = 0.20), clipped to [0,1]
5. **GRU denoiser** — unidirectional, hidden size 128, row-by-row sequence (84 features/row), manual parameter count (93,012 params), trained 25 epochs
6. **Restoration evaluation** — MSE 0.0308 → 0.0045, PSNR 15.1 → 23.6 dB (**85.5%** relative MSE improvement)
7. **Optional experiment** — CNN accuracy on clean/noisy/restored images (90.1% / 15.1% / 54.6%): restoration recovers ~53% of the accuracy lost to noise

## How to run

```bash
pip install -r requirements.txt   # torch, numpy, matplotlib, scikit-learn
jupyter notebook BLANCHIER_Felix_A3_BloodMNIST.ipynb
```

Run all cells top to bottom (kernel was restarted and fully re-executed before submission). The BloodMNIST subset is downloaded/prepared automatically on first run via `prepare_data.py` if `data/blood_mnist_subset.npz` is not already present.

## Notes

Educational exercise on synthetic RGB images (MedMNIST BloodMNIST, CC BY 4.0) — results have no clinical validity.
