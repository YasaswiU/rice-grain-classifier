"""
evaluate.py
-----------
Evaluates whichever trained model(s) are available (CNN and/or VGG16) on
the held-out test split.

Produces:
    results/confusion_matrix.png
    results/cnn_confusion_matrix.png
    results/vgg16_confusion_matrix.png
    results/cnn_training_history.png
    results/vgg16_training_history.png
    results/cnn_classification_report.txt
    results/vgg16_classification_report.txt
    results/model_comparison.csv

All numbers are computed from actual model predictions.

Run from the project root:
    python src/evaluate.py
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

try:
    from src import config
    from src.data_preprocessing import (
        collect_image_records,
        discover_classes,
        stratified_split,
        records_to_arrays,
    )
except ImportError:
    import config
    from data_preprocessing import (
        collect_image_records,
        discover_classes,
        stratified_split,
        records_to_arrays,
    )


# ============================================================
# CLASS NAMES
# ============================================================

def load_class_names() -> list[str]:
    """
    Load class names from models/class_names.json.

    If the file does not exist, discover the class names
    directly from the dataset folders.
    """

    class_map_path = os.path.join(
        config.MODELS_DIR,
        "class_names.json"
    )

    if os.path.exists(class_map_path):
        with open(class_map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return discover_classes()


# ============================================================
# TEST DATA
# ============================================================

def get_test_set(class_names: list[str]):
    """
    Create the held-out test set.

    The original image size may be 224x224 because the same
    test data is used for both CNN and VGG16 evaluation.

    Resizing is performed later inside evaluate_model()
    according to the model's expected input size.
    """

    records = collect_image_records()

    _, _, test_records = stratified_split(records)

    X_test, y_test = records_to_arrays(
        test_records,
        class_names
    )

    return X_test, y_test


# ============================================================
# TRAINING HISTORY
# ============================================================

def plot_training_history(
    history_path: str,
    output_path: str,
    model_label: str
) -> bool:
    """
    Plot training and validation accuracy/loss.

    Returns True if the history was successfully plotted.
    """

    if not os.path.exists(history_path):
        print(
            f"[WARNING] No training history found at "
            f"{history_path}, skipping plot."
        )
        return False

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if "accuracy" in history:
        axes[0].plot(
            history["accuracy"],
            label="Train Accuracy"
        )

    if "val_accuracy" in history:
        axes[0].plot(
            history["val_accuracy"],
            label="Validation Accuracy"
        )

    axes[0].set_title(
        f"{model_label} - Accuracy"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    if "loss" in history:
        axes[1].plot(
            history["loss"],
            label="Train Loss"
        )

    if "val_loss" in history:
        axes[1].plot(
            history["val_loss"],
            label="Validation Loss"
        )

    axes[1].set_title(
        f"{model_label} - Loss"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[INFO] Saved training history plot to "
        f"{output_path}"
    )

    return True


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(
    model,
    X_test: np.ndarray,
    preprocess: str
) -> np.ndarray:
    """
    Resize and preprocess test images according to the model.

    CNN:
        128 x 128 x 3

    VGG16:
        224 x 224 x 3
        followed by VGG16 preprocess_input()
    """

    # --------------------------------------------------------
    # Determine expected model input size
    # --------------------------------------------------------

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Unexpected model input shape: {input_shape}"
        )

    expected_height = input_shape[1]
    expected_width = input_shape[2]
    expected_channels = input_shape[3]

    print(
        f"[INFO] Model expects input shape: "
        f"{input_shape}"
    )

    # --------------------------------------------------------
    # Check number of channels
    # --------------------------------------------------------

    if expected_channels != 3:
        raise ValueError(
            f"Expected 3-channel RGB images, "
            f"but model expects {expected_channels} channels."
        )

    # --------------------------------------------------------
    # Resize test images
    # --------------------------------------------------------

    if (
        X_test.shape[1] != expected_height
        or X_test.shape[2] != expected_width
    ):
        print(
            f"[INFO] Resizing test images from "
            f"{X_test.shape[1]}x{X_test.shape[2]} "
            f"to "
            f"{expected_height}x{expected_width}"
        )

        X_input = tf.image.resize(
            X_test,
            (
                expected_height,
                expected_width
            ),
            method="bilinear"
        ).numpy()

    else:
        X_input = X_test.copy()

    # --------------------------------------------------------
    # VGG16 preprocessing
    # --------------------------------------------------------

    if preprocess == "vgg16":
        from keras.applications.vgg16 import preprocess_input

        print(
            "[INFO] Applying VGG16 preprocessing..."
        )

        # records_to_arrays() returns images in 0-1 range.
        # VGG16 preprocess_input() expects pixel values
        # corresponding to the 0-255 range.
        X_input = X_input * 255.0

        X_input = preprocess_input(
            X_input
        )

    # --------------------------------------------------------
    # CNN preprocessing
    # --------------------------------------------------------

    elif preprocess == "cnn":

        print(
            "[INFO] Using CNN preprocessing "
            "(0-1 normalized images)..."
        )

    else:
        raise ValueError(
            f"Unknown preprocessing type: {preprocess}"
        )

    return X_input.astype(
        np.float32
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_path: str,
    model_label: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    preprocess: str = "cnn",
) -> dict | None:
    """
    Load a saved model, run predictions on the test set,
    calculate evaluation metrics, print the classification
    report, and save a confusion matrix.

    Returns:
        Dictionary containing evaluation metrics,
        or None if the model does not exist.
    """

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(model_path):

        print(
            f"[WARNING] Model not found at "
            f"{model_path}. "
            f"Skipping {model_label} evaluation."
        )

        return None

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        f"\n[INFO] Loading {model_label} from "
        f"{model_path}..."
    )

    model = tf.keras.models.load_model(
        model_path
    )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    print(
        f"[INFO] Preparing test images for "
        f"{model_label}..."
    )

    X_input = prepare_model_input(
        model,
        X_test,
        preprocess
    )

    print(
        f"[INFO] Final evaluation input shape: "
        f"{X_input.shape}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(
        f"[INFO] Running predictions for "
        f"{model_label}..."
    )

    y_prob = model.predict(
        X_input,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    y_pred = np.argmax(
        y_prob,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    acc = accuracy_score(
        y_test,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"{model_label} Classification Report"
    )

    print(
        f"{'=' * 60}"
    )

    print(report)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_path = os.path.join(
        config.RESULTS_DIR,
        f"{model_label.lower()}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{model_label} Test Accuracy: "
            f"{acc:.4f}\n\n"
        )

        f.write(report)

    print(
        f"[INFO] Saved classification report to "
        f"{report_path}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        f"{model_label} - Confusion Matrix "
        f"(Test Accuracy: {acc * 100:.2f}%)"
    )

    fig.tight_layout()

    cm_path = os.path.join(
        config.RESULTS_DIR,
        f"{model_label.lower()}_confusion_matrix.png"
    )

    fig.savefig(
        cm_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[INFO] Saved confusion matrix to "
        f"{cm_path}"
    )

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {
        "Model": model_label,
        "Accuracy": round(
            float(acc),
            4
        ),
        "Precision": round(
            float(precision),
            4
        ),
        "Recall": round(
            float(recall),
            4
        ),
        "F1 Score": round(
            float(f1),
            4
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RICE GRAIN CLASSIFIER - MODEL EVALUATION"
    )

    print(
        "=" * 70
        + "\n"
    )

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

    class_names = load_class_names()

    print(
        f"[INFO] Classes: {class_names}"
    )

    print(
        f"[INFO] Number of classes: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Load test set
    # --------------------------------------------------------

    print(
        "\n[INFO] Loading held-out test set..."
    )

    X_test, y_test = get_test_set(
        class_names
    )

    print(
        f"[INFO] Test set size: "
        f"{len(y_test)} images"
    )

    print(
        f"[INFO] Original test image shape: "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # Results container
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Evaluate CNN
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATING CUSTOM CNN"
    )

    print(
        "=" * 70
    )

    cnn_result = evaluate_model(
        config.CNN_MODEL_PATH,
        "CNN",
        X_test,
        y_test,
        class_names,
        preprocess="cnn"
    )

    if cnn_result:
        results.append(
            cnn_result
        )

    # --------------------------------------------------------
    # Evaluate VGG16
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATING VGG16"
    )

    print(
        "=" * 70
    )

    vgg_result = evaluate_model(
        config.VGG16_MODEL_PATH,
        "VGG16",
        X_test,
        y_test,
        class_names,
        preprocess="vgg16"
    )

    if vgg_result:
        results.append(
            vgg_result
        )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not results:

        print(
            "\n[ERROR] No trained models were found."
        )

        print(
            "Train the CNN and/or VGG16 model first:"
        )

        print(
            "  python src/train_cnn.py"
        )

        print(
            "  python src/train_vgg16.py"
        )

        return

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best = max(
        results,
        key=lambda r: r["Accuracy"]
    )

    best_label = best["Model"].lower()

    src_cm = os.path.join(
        config.RESULTS_DIR,
        f"{best_label}_confusion_matrix.png"
    )

    dst_cm = os.path.join(
        config.RESULTS_DIR,
        "confusion_matrix.png"
    )

    if os.path.exists(src_cm):

        shutil.copyfile(
            src_cm,
            dst_cm
        )

        print(
            f"\n[INFO] Best model "
            f"({best['Model']}) confusion matrix "
            f"also saved to {dst_cm}"
        )

    # --------------------------------------------------------
    # Training history plots
    # --------------------------------------------------------

    print(
        "\n[INFO] Creating training history plots..."
    )

    plot_training_history(
        config.CNN_HISTORY_PATH,
        os.path.join(
            config.RESULTS_DIR,
            "cnn_training_history.png"
        ),
        "CNN"
    )

    plot_training_history(
        config.VGG16_HISTORY_PATH,
        os.path.join(
            config.RESULTS_DIR,
            "vgg16_training_history.png"
        ),
        "VGG16"
    )

    # --------------------------------------------------------
    # Model comparison CSV
    # --------------------------------------------------------

    df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    )

    comparison_path = os.path.join(
        config.RESULTS_DIR,
        "model_comparison.csv"
    )

    df.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\n[INFO] Model comparison saved to "
        f"{comparison_path}"
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATION COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()