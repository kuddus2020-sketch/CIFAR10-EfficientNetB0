# ⚡ CIFAR-10 Image Classification with EfficientNet-B0

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/CIFAR10_EfficientNetB0_Enhanced.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **EfficientNet-B0** fine-tuned on CIFAR-10 with two-phase training, Grad-CAM visualization, and a fully interactive inference widget — all in a single Colab notebook.

---

## 📌 Table of Contents

- [Demo](#-demo)
- [Project Overview](#-project-overview)
- [Notebook Structure](#-notebook-structure)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Requirements](#-requirements)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [License](#-license)

---

## 🎬 Demo

Click **"Open in Colab"** above → set runtime to **T4 GPU** → **Run All**.  
Everything is automatic: data downloads, model trains, evaluation plots generate, and a **live interactive widget** launches at the end where you can:

- 📤 Upload your own image and get a real-time prediction
- 🎲 Pick a random CIFAR-10 test sample by class
- 🔥 Toggle **Grad-CAM** attention heatmaps on/off

---

## 🔍 Project Overview

| Item | Details |
|---|---|
| **Dataset** | CIFAR-10 (60,000 images, 10 classes, auto-downloaded) |
| **Base Model** | EfficientNet-B0 (ImageNet pretrained) |
| **Input Size** | 32×32 → upscaled to 224×224 |
| **Training** | 2-phase: head warmup (5 ep) + full fine-tune (15 ep) |
| **Explainability** | Grad-CAM on last conv block |
| **Target Accuracy** | ~93–94% on test set |
| **Hardware** | Free Colab T4 GPU is sufficient |

---

## 📒 Notebook Structure

| Step | Description |
|---|---|
| **Step 1** | Install dependencies (`timm`, `pytorch-grad-cam`) & imports |
| **Step 2** | Config, data transforms, CIFAR-10 auto-download, train/val/test split |
| **Step 3** | EDA — class distribution bar chart + sample image grid |
| **Step 4** | Model definition — EfficientNet-B0 backbone + custom BN→Dropout→Linear head |
| **Step 5** | Two-phase training with AMP, gradient clipping, OneCycleLR |
| **Step 6** | Training curves — loss & accuracy plots |
| **Step 7** | Test evaluation — confusion matrix, per-class accuracy, ROC curves |
| **Step 8** | Grad-CAM visualization on test samples |
| **Step 9** | Hard sample analysis — worst misclassified examples |
| **Step 10** | Interactive inference widget (upload image / random sample) |

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~93–94% |
| Training Epochs | 20 (5 warmup + 15 fine-tune) |
| Best Checkpoint | Auto-saved during training |

*Exact numbers will appear in your notebook output after training.*

---

## 🚀 How to Run

### ▶️ Option 1 — Google Colab (Recommended — Zero Setup)

1. Click the **"Open in Colab"** badge at the top
2. Go to **Runtime → Change runtime type → T4 GPU** → Save
3. Click **Runtime → Run all**
4. Done! The notebook handles everything:
   - ✅ Installs all dependencies
   - ✅ Downloads CIFAR-10 automatically
   - ✅ Trains for 20 epochs and saves the best model
   - ✅ Generates all evaluation plots
   - ✅ Launches the interactive widget at the end

> ⚠️ **Note on the interactive widget (Step 10):** It needs the trained model (`best_model.pth`) from Step 5. If you want to skip training and run only inference, save your `best_model.pth` to Google Drive at:
> `MyDrive/CIFAR10_EfficientNetB0/best_model.pth`
> and mount Drive before running Step 10.

---

### 💻 Option 2 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook CIFAR10_EfficientNetB0_Enhanced.ipynb
```

> A CUDA-capable GPU is recommended for local runs.

---

## 📦 Requirements

All dependencies install automatically in the first notebook cell.  
For local setup, use `requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pytorch-grad-cam>=1.4.8
numpy
matplotlib
seaborn
tqdm
ipywidgets
Pillow
scikit-learn
```

---

## 🏗 Model Architecture

```
Input (3 × 224 × 224)
        │
        ▼
EfficientNet-B0 Backbone   ← ImageNet pretrained weights
  (8 MBConv blocks)
        │
        ▼
AdaptiveAvgPool2d(1)  →  (1280,)
        │
        ▼
Custom Classification Head:
  BatchNorm1d(1280)
  Dropout(0.4)
  Linear(1280 → 256)
  SiLU activation
  BatchNorm1d(256)
  Dropout(0.2)
  Linear(256 → 10)
        │
        ▼
   Output (10 classes)
```

---

## 🔥 Training Strategy

**Phase 1 — Head Warmup (5 epochs)**
- Backbone frozen → only custom head trains
- AdamW, lr = 1e-3, OneCycleLR

**Phase 2 — Full Fine-tuning (15 epochs)**
- All layers unfrozen with **differential learning rates**
  - Backbone: `lr = 1e-4`
  - Head: `lr = 5e-4`
- Mixed precision training (AMP)
- Gradient clipping (max norm = 1.0)
- Label smoothing = 0.1
- Data augmentation: RandAugment + RandomErasing

---

## 📁 Repository Structure

```
YOUR_REPO_NAME/
├── CIFAR10_EfficientNetB0_Enhanced.ipynb   ← Main notebook (run this)
├── requirements.txt                         ← Python dependencies
├── LICENSE                                  ← MIT License
└── README.md                                ← This file
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [EfficientNet paper](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Jacob Gildenblat
- [torchvision](https://pytorch.org/vision/) — CIFAR-10 loader & pretrained weights
