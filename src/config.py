"""
config.py
---------
Central configuration for the Rice Grain Classifier project.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_best.keras")
VGG16_MODEL_PATH = os.path.join(MODELS_DIR, "vgg16_best.keras")

CNN_HISTORY_PATH = os.path.join(MODELS_DIR, "cnn_history.json")
VGG16_HISTORY_PATH = os.path.join(MODELS_DIR, "vgg16_history.json")

# Create required folders
for directory in (DATASET_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------

# 16 is safer for VGG16 on normal laptops
BATCH_SIZE = 16

# Custom CNN
EPOCHS_CNN = 40
LEARNING_RATE_CNN = 1e-3

# VGG16
EPOCHS_VGG16_HEAD = 8
EPOCHS_VGG16_FINE_TUNE = 8

LEARNING_RATE_VGG16_HEAD = 1e-4
LEARNING_RATE_VGG16_FINE_TUNE = 1e-5

# Number of VGG16 layers to fine-tune
VGG16_FINE_TUNE_LAYERS = 4


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

EARLY_STOPPING_PATIENCE = 3

REDUCE_LR_PATIENCE = 2

REDUCE_LR_FACTOR = 0.5

MIN_LEARNING_RATE = 1e-7