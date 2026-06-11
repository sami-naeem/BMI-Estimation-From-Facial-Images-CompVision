# BMI Estimation from Facial Images — Replicating Face-to-BMI with Modern Computer Vision

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Predicts Body Mass Index (BMI) from a single facial photograph using deep-learning-based computer vision: an EfficientNetB3 backbone fine-tuned via transfer learning for facial feature extraction, followed by an ensemble of classical regression models.

---

## Project Background

BMI is a key health indicator, widely used in medical studies and treatments — but most publicly available BMI data is self-reported and often inaccurate. Traditionally, obtaining BMI data requires either accurate self-reporting or an in-person clinical measurement.

This project **replicates and extends** the reference paper:

> Kocabey, E., Camurcu, M., Ofli, F., Aytar, Y., Marin, J., Torralba, A., & Weber, I. (2017). *Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media.* Proceedings of the Eleventh International AAAI Conference on Web and Social Media (ICWSM 2017). (Included as `Reference Journal Paper.pdf`.)

The original Face-to-BMI system showed that BMI can be inferred from noisy, real-world social media face images with near-human accuracy. Its computer vision pipeline had two stages:

1. **Deep feature extraction** — fc6-layer features from pre-trained convolutional networks (VGG-Net, trained on general object classification, and VGG-Face, trained on face recognition)
2. **Regression** — an epsilon support vector regression (ε-SVR) model trained on those features

The paper's reported test-set performance (Pearson r):

| Paper model | Male | Female | Overall |
|---|---|---|---|
| Face-to-BMI – VGG-Net | 0.58 | 0.36 | 0.47 |
| Face-to-BMI – VGG-Face | 0.71 | 0.57 | **0.65** |

**Our goal:** replicate this two-stage computer vision pipeline with modern pre-trained CNN backbones, beat the paper's performance metrics, and deploy the model behind a simple web API for real-time BMI prediction from images or webcam input.

---

## Approach: Computer Vision Pipeline

We kept the paper's two-stage design (CNN feature extractor → regressor) but systematically evaluated **five modern pre-trained backbones**, each fine-tuned on the BMI dataset with transfer learning:

| Backbone | Input size | Fine-tuning strategy | Features extracted |
|---|---|---|---|
| VGGFace | 224×224 | Blocks 4–5 unfrozen, augmentation | fc6 (4,096-d) |
| VGG16 / VGG19 (VGGNet) | 224×224 | Frozen base + regression head, upper layers fine-tuned | conv features |
| ResNet50 | 224×224 | GAP + Dense head, top 25 layers unfrozen | pooled features |
| FaceNet (Inception-ResNet) | 160×160 | MTCNN face alignment, Block8 + FC fine-tuned | 512-d embeddings |
| **EfficientNet (B3)** | 300×300 | GAP + Dropout + Dense head, top 30 layers unfrozen | 1,536-d embeddings |

Standard computer-vision preprocessing was applied throughout: resizing to each backbone's native resolution, normalization ([0, 1] scaling or ImageNet-style mean subtraction), float32 4D tensor formatting, and — for FaceNet — MTCNN face detection and alignment. In every pipeline, a binary gender feature was appended to the CNN embeddings before fitting the downstream regressors.

### Backbone comparison — final results

Each fine-tuned backbone's features were fed to regression models and evaluated on the held-out test set:

| Model | MAE | Pearson r (Overall) | Pearson r (Male) | Pearson r (Female) |
|---|---|---|---|---|
| VGGFace | 5.04 | 0.641 | 0.653 | 0.631 |
| VGGNet | 4.99 | 0.649 | 0.699 | 0.583 |
| ResNet50 | 6.043 | 0.465 | 0.409 | 0.508 |
| FaceNet | 5.52 | 0.577 | 0.618 | 0.531 |
| **EfficientNet** | **4.72** | **0.67** | **0.69** | **0.65** |

**EfficientNet was the best performer**, with an overall Pearson r of **0.67 — beating the paper's best result of 0.65 (VGG-Face)**. It was also markedly more balanced across genders (0.69 male / 0.65 female) than the paper's VGG-Face model (0.71 / 0.57). **This repository covers the EfficientNet pipeline** — the winning approach.

---

## Selecting the Best EfficientNet Variant (B0–B7)

The EfficientNet family spans eight variants (B0–B7) that scale depth, width, and input resolution together. Rather than assuming a variant, **all eight were trained and compared as baselines** before committing to one (see `Notebooks - Models/1_EfficientNet_B0-B7_Model_Selection.ipynb`). Each variant was trained for **5 epochs** with its base frozen (ImageNet weights), a GlobalAveragePooling2D → Dropout(0.2) → Dense(1) regression head, Adam optimizer, MSE loss, and early stopping — at its native input resolution.

### Baseline screening results (validation set, 5 frozen epochs)

| Variant | Input size | MSE | MAE | Pearson r |
|---|---|---|---|---|
| EfficientNetB0 | 224×224 | **71.02** | **6.49** | 0.066 |
| EfficientNetB1 | 240×240 | 74.50 | 6.51 | 0.089 |
| EfficientNetB2 | 260×260 | 81.91 | 6.90 | 0.013 |
| EfficientNetB3 | 300×300 | 85.11 | 7.10 | 0.051 |
| EfficientNetB4 | 380×380 | 80.14 | 7.00 | 0.049 |
| EfficientNetB5 | 456×456 | 79.14 | 6.94 | 0.041 |
| EfficientNetB6 | 528×528 | 76.23 | 6.87 | **0.093** |
| EfficientNetB7 | 600×600 | 74.40 | 6.81 | 0.049 |

In this short frozen-baseline screening, **EfficientNetB0** achieved the lowest error (MSE 71.02, MAE 6.49) and **EfficientNetB6** the highest Pearson r (0.093). All correlations are near zero at this stage — with the backbone frozen and only 5 epochs of head training, the screening is largely inconclusive on predictive power and mainly verifies that each variant trains stably at its native resolution.

**EfficientNetB3 was selected for full fine-tuning as the best accuracy-vs-compute trade-off**: a mid-family variant with substantially more capacity than B0/B1 at a 300×300 input — a fraction of the training and inference cost of B6 (528×528) or B7 (600×600), which matters for the real-time API deployment goal. The bet paid off once the last 30 layers were unfrozen and fine-tuned with augmentation: validation MAE dropped from ~7.1 (frozen baseline) to ~4.4, and the final pipeline beat the reference paper.

---

## Key Results — EfficientNetB3 Pipeline

After fine-tuning EfficientNetB3, its 1,536-d features (+ gender) were used to train 8 regression models, evaluated on the held-out test set (752 images):

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
| **Ensemble (RF + Ridge + CatBoost)** | **47.76** | **4.73** | **0.670** |

Performance was evaluated separately for male and female subsets; results are consistent across genders — a notable improvement over the paper's large gender gap.

**Bottom line: overall Pearson r 0.67 and MAE 4.72 vs. the paper's best of 0.65 — goal achieved.**

---

## Architecture

### Stage 1 — CNN Feature Extraction (Computer Vision)

| Component | Detail |
|---|---|
| Base model | EfficientNetB3 (ImageNet weights) |
| Input size | 300 × 300 |
| Fine-tuning | Last 30 layers unfrozen |
| Training epochs | 15 (early stopping, patience = 3) |
| Optimizer | Adam (lr = 0.001) |
| Head | GlobalAveragePooling2D → Dropout(0.2) → Dense(1, linear) |
| Output features | 1,536-dim per image |

**Data augmentation applied during fine-tuning** (to compensate for the small ~4K-image dataset and improve robustness to real-world image variability):
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

- **Source:** The **VisualBMI dataset** used in the reference paper — facial images originally collected from public Reddit posts (r/progresspics), manually cropped and annotated with gender and BMI (computed from reported height and weight). Included in `Data/Data/`.
- **Images:** 3,962 BMP facial photographs, average size ~351×286 px before resizing
- **Labels:** `Data/Data/data.csv` (4,206 entries; rows with missing image files were dropped)

| Column | Description |
|---|---|
| `name` | Image filename |
| `bmi` | Continuous target (mean ≈ 32.8, range ≈ 17.7 – 86.0) |
| `gender` | Male / Female (~58% male / 42% female in training) |
| `is_training` | Pre-defined split flag (1 = train, 0 = test) |

**Split after filtering missing images:** 3,210 training images / 752 test images (pre-defined in the CSV, matching the paper's subject-disjoint protocol)

---

## Project Structure

```
BMI-Estimation-From-Facial-Images/
├── Data/
│   └── Data/
│       ├── data.csv                                  # Labels + metadata
│       └── Images/                                   # BMP face images
├── Notebooks - Models/
│   ├── 1_EfficientNet_B0-B7_Model_Selection.ipynb    # Baseline screening of all 8 EfficientNet variants
│   ├── 2_EfficientNetB3_Final_Pipeline.ipynb         # Final pipeline: fine-tuning, regressors, ensemble, export
│   └── 2_EfficientNetB3_Final_Pipeline.html          # Static export of the final pipeline notebook
├── Saved Models/
│   ├── EfficientNetB3_feature_extractor.tflite       # TFLite feature extractor (11.7 MB)
│   ├── EfficientNetB3_bmi_model_quant.pkl            # Quantized TFLite model, pickled (11.7 MB)
│   └── EfficientNetB3_Ridge_regressor.joblib         # Ridge regressor (13 KB)
├── Reference Journal Paper.pdf                       # Kocabey et al. (2017), ICWSM
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

### Run the notebooks

```bash
# 1. Model selection: trains all EfficientNet B0–B7 baselines and compares them
jupyter notebook "Notebooks - Models/1_EfficientNet_B0-B7_Model_Selection.ipynb"

# 2. Final pipeline: fine-tunes EfficientNetB3 and builds the regressor ensemble
jupyter notebook "Notebooks - Models/2_EfficientNetB3_Final_Pipeline.ipynb"
```

Or upload to [Google Colab](https://colab.research.google.com/). Together the notebooks run the full pipeline:

1. Baseline evaluation of all EfficientNet variants B0 – B7 (5 epochs each, frozen base) — notebook 1
2. Fine-tuning of the selected EfficientNetB3 (15 epochs, last 30 layers unfrozen, augmentation) — notebooks 1 & 2
3. Feature extraction (training + test sets) — notebook 2
4. Training and evaluation of all 8 regressors — notebook 2
5. Ensemble construction and final metrics — notebook 2
6. Export of TFLite and joblib model files — notebook 2

### Load saved models for inference

```python
import joblib
import pickle
import numpy as np

# Load Ridge regressor
regressor = joblib.load('Saved Models/EfficientNetB3_Ridge_regressor.joblib')

# Load quantized TFLite feature extractor
with open('Saved Models/EfficientNetB3_bmi_model_quant.pkl', 'rb') as f:
    tflite_model = pickle.load(f)
```

### Deployment

The saved TFLite model (`Saved Models/EfficientNetB3_feature_extractor.tflite`) and Ridge regressor (`Saved Models/EfficientNetB3_Ridge_regressor.joblib`) are the inference artifacts for a lightweight deployment. They can be wired into a Flask or Streamlit app:

1. Accept an image upload (or webcam frame)
2. Preprocess to 300×300 and run through the TFLite interpreter to get features
3. Pass features (+ gender) to the loaded regressor
4. Return the predicted BMI value

---

## Limitations & Future Work

As discussed in the reference paper, face-based BMI prediction is noisy at the individual level and best suited to population-level trend analysis; demographic bias is a real concern. Planned refinements:

- **Strengthen the dataset** — annotate race/ethnicity, expand age range, balance genders, add varied training data
- **Refine the modeling strategy** — further ensemble and hyperparameter tuning (learning rate, epochs, layers to unfreeze), experiment with attention-based backbones
- **Embed fairness & transparency** — evaluate performance within gender / age / ethnic subgroups, integrate interpretability tools

With these refinements, facial-based BMI estimation could serve as an early, non-invasive screening tool complementing clinical consultations and traditional health assessments.

---

## Reference

Kocabey, E., Camurcu, M., Ofli, F., Aytar, Y., Marin, J., Torralba, A., & Weber, I. (2017). *Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media.* ICWSM 2017. Included as `Reference Journal Paper.pdf`.

---

## License

MIT — Copyright 2026 Sami Naeem
