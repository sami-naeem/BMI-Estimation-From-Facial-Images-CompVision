# BMI Estimation from Facial Images — EfficientNetB3

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Predicts Body Mass Index (BMI) from a single facial photograph using a two-stage pipeline: EfficientNetB3 transfer learning for feature extraction, followed by an ensemble of regression models.

---

## Overview

Estimating BMI non-invasively from facial images has applications in healthcare screening, fitness tracking, and medical research. This project replicates and extends the methodology from the reference paper (included in the repo), achieving competitive regression performance on the provided dataset.

**This repository covers the EfficientNetB3 approach** — one of five model architectures evaluated in the broader study.

**Pipeline summary:**
1. Fine-tune EfficientNetB3 (pre-trained on ImageNet) on facial images
2. Extract 1,536-dimensional feature vectors per image
3. Append a binary gender feature (1,537-dim total)
4. Train and evaluate 8 regression models; combine the top 3 into a VotingRegressor ensemble

**Assignment deliverables addressed:**
- Trained model saved as TFLite + joblib for lightweight API deployment
- Full experimental write-up in the accompanying notebook
- Live demo presented using saved model inference

---

## Key Results

Results on the held-out test set (752 images):

| Model | MSE | MAE | Pearson r |
|---|---|---|---|
| Random Forest | 47.35 | 4.73 | 0.673 |
| Ridge Regression | 48.15 | 4.77 | 0.667 |
| CatBoost | 48.50 | 4.76 | 0.665 |
| LightGBM | 48.67 | 4.78 | 0.662 |
| XGBoost | 49.50 | 4.83 | 0.657 |
| SVR (RBF) | 49.76 | 4.89 | 0.655 |
| KNN (k=5) | 50.04 | 4.89 | 0.649 |
| MLP | 51.03 | 4.90 | 0.650 |
| **Ensemble (RF + Ridge + CatBoost)** | **~47.2** | **~4.7** | **~0.675** |

Performance was evaluated separately for male and female subsets; results are consistent across genders.

> The goal of this project is to match or surpass the Pearson correlation reported in the reference paper. See `Reference Journal Paper.pdf` for the baseline metrics.

---

## Architecture

### Stage 1 — CNN Feature Extraction

| Component | Detail |
|---|---|
| Base model | EfficientNetB3 (ImageNet weights) |
| Input size | 300 × 300 |
| Fine-tuning | Last 30 layers unfrozen |
| Training epochs | 15 (early stopping, patience = 3) |
| Optimizer | Adam (lr = 0.001) |
| Head | GlobalAveragePooling2D → Dropout(0.2) → Dense(1, linear) |
| Output features | 1,536-dim per image |

**Data augmentation applied during fine-tuning:**
- Rotation: ±20°
- Width / height shift: 10%
- Zoom: 10%
- Horizontal flip: enabled

### Stage 2 — Ensemble Regression

Gender (binary-encoded) is appended to the CNN features, giving a **1,537-dimensional** input to each regressor.

```
Facial Image (300×300 px)
        │
EfficientNetB3 (fine-tuned, last 30 layers unfrozen)
        │
GlobalAveragePooling2D
        │
  [1536-dim vector]
        │  + gender (0/1)
  [1537-dim vector]
        │
VotingRegressor
 ├── Random Forest (n=100)
 ├── Ridge Regression
 └── CatBoost (iter=100, lr=0.1)
        │
   BMI prediction
```

---

## Dataset

- **Source:** Provided alongside the reference paper (included in `Data/Data/`)
- **Images:** 3,962 BMP facial photographs (`img_0.bmp` – `img_4205.bmp`)
- **Labels:** `Data/Data/data.csv`

| Column | Description |
|---|---|
| `name` | Image filename |
| `bmi` | Continuous target (range ≈ 21.7 – 51.7) |
| `gender` | Male / Female |
| `is_training` | Pre-defined split flag (1 = train, 0 = test) |

**Split:** 3,210 training images / 752 test images (pre-defined in the CSV)

---

## Project Structure

```
BMI-Estimation-From-Facial-Images/
├── Data/
│   └── Data/
│       ├── data.csv                                          # Labels + metadata
│       └── Images/                                           # 3,963 BMP face images
├── EfficientNetB3_Transfer_Learning_final_refactored.ipynb   # Main notebook
├── EfficientNetB3_Transfer_Learning_final_refactored.html    # Exported notebook (static view)
├── efficientnet_feature_extractor.tflite                     # TFLite feature extractor (11.7 MB)
├── EfficientNetB3_lrsched_bmi_model_quant.pkl                # Quantized TFLite model, pickled (11.7 MB)
├── EfficientNet_Ridge_regressor.joblib                       # Ridge regressor (13 KB)
├── Reference Journal Paper.pdf                               # Source paper
└── README.md
```

---

## Installation

Python 3.11 recommended. The project was developed in Google Colab.

```bash
pip install tensorflow numpy pandas scikit-learn xgboost lightgbm catboost joblib matplotlib scipy
```

---

## Usage

### Run the notebook

```bash
jupyter notebook EfficientNetB3_Transfer_Learning_final_refactored.ipynb
```

Or upload to [Google Colab](https://colab.research.google.com/). The notebook is self-contained and runs the full pipeline:

1. Baseline evaluation of EfficientNet B0 – B7 (5 epochs each)
2. Fine-tuning of EfficientNetB3 (15 epochs, last 30 layers)
3. Feature extraction (training + test sets)
4. Training and evaluation of all 8 regressors
5. Ensemble construction and final metrics
6. Export of TFLite and joblib model files

### Load saved models for inference

```python
import joblib
import pickle
import numpy as np

# Load Ridge regressor
regressor = joblib.load('EfficientNet_Ridge_regressor.joblib')

# Load quantized TFLite feature extractor
with open('EfficientNetB3_lrsched_bmi_model_quant.pkl', 'rb') as f:
    tflite_model = pickle.load(f)
```

### Deployment

The saved TFLite model (`efficientnet_feature_extractor.tflite`) and Ridge regressor (`EfficientNet_Ridge_regressor.joblib`) are the inference artifacts for a lightweight deployment. They can be wired into a Flask or Streamlit app:

1. Accept an image upload
2. Preprocess to 300×300 and run through the TFLite interpreter to get features
3. Pass features (+ gender) to the loaded regressor
4. Return the predicted BMI value

---

## Reference

Wan, C., et al. (2020). *BMI Estimation from Facial Images using Deep Learning.* Included as `Reference Journal Paper.pdf`.

---

## License

MIT — Copyright 2026 Sami Naeem
