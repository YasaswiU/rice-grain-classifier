# Rice Grain Classifier

**Deep Learning Based Rice Variety Classification**

A complete, VS Code-ready deep learning project that classifies images of rice grains into five rice varieties using a custom Convolutional Neural Network (CNN) and a VGG16 transfer-learning model, with a Streamlit web application for interactive predictions.

---
### 🚀 Live Demo

[👉 Try the Rice Grain Classifier](https://rice-grain-classifier-ffyyjv3arcthmjhxha8g7b.streamlit.app/)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Dataset Structure](#dataset-structure)
5. [Technologies Used](#technologies-used)
6. [System Architecture](#system-architecture)
7. [Methodology](#methodology)
8. [Results](#results)
9. [Streamlit Application](#streamlit-application)
10. [Installation & VS Code Setup](#installation--vs-code-setup)
11. [How to Train the CNN](#how-to-train-the-cnn)
12. [How to Train VGG16](#how-to-train-vgg16)
13. [How to Evaluate](#how-to-evaluate)
14. [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
15. [Project Structure](#project-structure)
16. [Future Improvements](#future-improvements)
17. [Conclusion](#conclusion)

---

## Project Overview

Rice is one of the most widely consumed staple foods in the world, and different rice varieties command different market value, cooking properties, and export requirements. Manually distinguishing between rice varieties is time-consuming and error-prone. This project applies deep learning-based image classification to automatically identify the variety of a rice grain from a photograph.

## Problem Statement

Given an image of a single rice grain (or a small cluster of grains), classify it into one of **five** rice varieties. The classifier must generalize well to new, unseen images and must be deployable through a simple web interface.

## Objectives

- Build an automated image classification pipeline for rice grain varieties.
- Compare a custom CNN built from scratch against a VGG16 transfer-learning approach.
- Provide honest, reproducible evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix).
- Package the trained model behind an easy-to-use Streamlit web application.
- Keep the codebase clean, modular, and beginner-friendly for portfolio / academic submission.

## Dataset Structure

Place your rice grain images inside the `dataset/` folder, with one sub-folder per variety. Folder names become the class labels automatically — **no class names are hard-coded anywhere in the code**.

```text
dataset/
├── Variety_1/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Variety_2/
├── Variety_3/
├── Variety_4/
└── Variety_5/
```

Supported image formats: `.jpg`, `.jpeg`, `.png`.

You are free to rename the `Variety_1` ... `Variety_5` folders to the actual rice variety names (e.g. `Basmati`, `Jasmine`, `Arborio`, `Karacadag`, `Ipsala`) — the pipeline will pick up whatever folder names you use.

> **Note:** The `dataset/` folder ships with only `.gitkeep` placeholder files. You must add your own images before training.

## Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras** — model building and training
- **VGG16 (ImageNet weights)** — transfer learning backbone
- **scikit-learn** — evaluation metrics (precision, recall, F1, confusion matrix)
- **NumPy / Pandas** — data handling
- **Matplotlib / Seaborn** — visualization
- **OpenCV / Pillow** — image loading and validation
- **Streamlit** — web application front end

## System Architecture

```text
Rice Dataset
     |
Data Loading
     |
EDA
     |
Preprocessing
     |
Data Augmentation
     |
 ---------------
 |             |
 CNN         VGG16
 |             |
 ---------------
        |
   Evaluation
        |
 Model Comparison
        |
   Best Model
        |
   Prediction
        |
 Streamlit App
```

## Methodology

### Data Preprocessing

- Class names are dynamically discovered from `dataset/` sub-folder names.
- Corrupted or unreadable images are detected and skipped with a warning instead of crashing the pipeline.
- All images are resized to **224 x 224** and normalized.
- The dataset is split **per class** (stratified) into **70% train / 15% validation / 15% test**, with a fixed random seed so splits are reproducible.
- Splits are verified to be mutually exclusive at the file-path level — **no data leakage** between train, validation, and test sets.

### Data Augmentation

Applied only to the training split, using realistic transformations:

- Rotation (±20°)
- Width / height shift (±10%)
- Zoom (±10%)
- Horizontal flip
- Small brightness variation (0.9–1.1x)

### Custom CNN

A four-block convolutional network:

- 4 × (Conv2D → BatchNormalization → ReLU → Conv2D → BatchNormalization → ReLU → MaxPooling2D → Dropout)
- GlobalAveragePooling2D
- Dense(256, relu) → Dropout(0.4)
- Dense(5, softmax)

Trained with Adam optimizer, `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` (saves the best validation-accuracy model to `models/cnn_best.keras`).

### VGG16 Transfer Learning

- Loads `VGG16(weights="imagenet", include_top=False)`.
- **Phase 1:** the convolutional base is frozen; only a new classification head (GlobalAveragePooling → Dense(256) → Dropout → Dense(5, softmax)) is trained.
- **Phase 2 (fine-tuning):** the last few convolutional layers of VGG16 are unfrozen and trained with a very low learning rate (1e-5) to gently adapt the pretrained features to rice grain images.
- Best model saved to `models/vgg16_best.keras`.
- **ImageNet weights are not stored in this repository** — TensorFlow downloads them automatically the first time `train_vgg16.py` runs (internet connection required).

### Evaluation

`src/evaluate.py` loads whichever trained model(s) exist and computes, on the held-out **test** split only:

- Test accuracy
- Precision, recall, F1-score (weighted average) and full per-class classification report
- Confusion matrix (saved as an image)
- Training accuracy/loss curves (saved as images)
- A `model_comparison.csv` summarizing both models side by side

**All metrics are computed from real model predictions on the test set. No values are hard-coded or fabricated.**

## Results

> Results are generated by running `python src/evaluate.py` after training. Replace the placeholders below with the numbers produced on your machine/dataset.

| Model  | Test Accuracy | Precision | Recall | F1 Score |
|--------|---------------|-----------|--------|----------|
| CNN    | *To be generated after training* | — | — | — |
| VGG16  | *To be generated after training* | — | — | — |

Artifacts generated after evaluation, in `results/`:

- `confusion_matrix.png` — confusion matrix of the best-performing model
- `cnn_confusion_matrix.png`, `vgg16_confusion_matrix.png`
- `cnn_training_history.png`, `vgg16_training_history.png`
- `cnn_classification_report.txt`, `vgg16_classification_report.txt`
- `model_comparison.csv`

The project targets approximately **94% test accuracy** as a realistic goal for a well-curated five-class rice grain dataset, but the true number reported here is always whatever the model actually achieves — never a fabricated figure.

## Streamlit Application

`app.py` provides a simple, professional web UI:

- Upload a JPG/JPEG/PNG image of a rice grain.
- Select which trained model to use (CNN or VGG16), if available.
- View the uploaded image, predicted variety, confidence score, and full probability distribution across all five classes as a bar chart and table.
- Sidebar with **About Project**, **Model Information**, **Dataset Information**, and **Technologies Used**.
- If a model hasn't been trained yet, the app shows a friendly message instead of crashing.

## Installation & VS Code Setup

### Requirements

- Windows with VS Code installed
- Python 3.10+ installed and available on PATH
- PowerShell (default on Windows)

### Step 1 — Open the project in VS Code

Open the `rice-grain-classifier` folder in VS Code (`File → Open Folder...`).

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3 — Activate the virtual environment

```powershell
.venv\Scripts\activate
```

> If PowerShell blocks the activation script, run VS Code's terminal as PowerShell and execute:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once, then retry activation.

### Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 5 — Add your dataset

Place your rice grain images inside `dataset/Variety_1/`, `dataset/Variety_2/`, ... `dataset/Variety_5/` (rename the folders to your actual variety names if you like — the code detects folder names automatically).

## How to Train the CNN

```powershell
python src/train_cnn.py
```

This trains the custom CNN, saves the best model to `models/cnn_best.keras`, and saves training history to `models/cnn_history.json`.

## How to Train VGG16

```powershell
python src/train_vgg16.py
```

This runs Phase 1 (frozen base) then Phase 2 (fine-tuning), saving the best model to `models/vgg16_best.keras` and history to `models/vgg16_history.json`. Requires an internet connection the first time (to download ImageNet weights).

## How to Evaluate

```powershell
python src/evaluate.py
```

Evaluates every trained model found in `models/` on the test split and writes all reports/plots/CSV into `results/`.

## How to Run the Streamlit App

```powershell
streamlit run app.py
```

This opens the web application in your browser (usually at `http://localhost:8501`).

## Project Structure

```text
rice-grain-classifier/
│
├── dataset/
│   ├── Variety_1/
│   ├── Variety_2/
│   ├── Variety_3/
│   ├── Variety_4/
│   └── Variety_5/
│
├── models/                 # trained models saved here (.keras) — not committed to git
├── results/                 # evaluation plots, reports, CSV — not committed to git
│
├── notebooks/
│   └── rice_grain_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py             # all configuration values in one place
│   ├── data_preprocessing.py # dataset loading, splitting, augmentation
│   ├── train_cnn.py          # custom CNN training
│   ├── train_vgg16.py        # VGG16 transfer learning + fine-tuning
│   ├── evaluate.py           # evaluation, confusion matrix, reports
│   └── predict.py            # single-image prediction utility
│
├── app.py                   # Streamlit web application
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Improvements

- Add more rice varieties and a larger, more diverse dataset.
- Experiment with other backbones (ResNet50, EfficientNet, MobileNetV3) for a lighter, faster deployable model.
- Add test-time augmentation (TTA) to improve prediction robustness.
- Add explainability (Grad-CAM) to visualize which parts of the grain influenced the prediction.
- Containerize the app with Docker for easier deployment.
- Add a REST API (FastAPI) alongside the Streamlit UI for programmatic access.
- Track experiments with MLflow or Weights & Biases.

## Conclusion

This project demonstrates a complete, end-to-end deep learning workflow — from raw image data to a deployed web application — comparing a custom CNN against a transfer-learning approach with VGG16. It is designed to be reproducible, honest about its performance, and easy to extend, making it suitable for academic submission, portfolio demonstration, and further research.
