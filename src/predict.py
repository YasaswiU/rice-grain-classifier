"""
predict.py
----------
Reusable single-image prediction utility, shared by the Streamlit app,
the notebook, and the command line.

Usage (CLI):
    python src/predict.py path/to/image.jpg --model cnn
    python src/predict.py path/to/image.jpg --model vgg16
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from src import config
except ImportError:
    import config


class PredictionError(Exception):
    """Raised when an image cannot be classified (missing model, bad file, etc.)."""


def load_class_names() -> list[str]:
    class_map_path = os.path.join(config.MODELS_DIR, "class_names.json")
    if os.path.exists(class_map_path):
        with open(class_map_path) as f:
            return json.load(f)
    try:
        from src.data_preprocessing import discover_classes
    except ImportError:
        from data_preprocessing import discover_classes
    return discover_classes()


def _load_and_prepare_image(image_path: str, model_type: str) -> np.ndarray:
    """
    Load an image file, validate it, resize to 224x224, and apply the
    preprocessing appropriate for the given model type ('cnn' or 'vgg16').
    Raises PredictionError on any problem with the file.
    """
    if not os.path.exists(image_path):
        raise PredictionError(f"Image file not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise PredictionError(
            f"Unsupported image format '{ext}'. Supported formats: {config.SUPPORTED_EXTENSIONS}"
        )

    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(config.IMAGE_SIZE)
            arr = np.asarray(img, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PredictionError(f"The image file is corrupted or unreadable: {exc}") from exc

    if model_type == "vgg16":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


def _resolve_model_path(model_type: str) -> str:
    if model_type == "cnn":
        return config.CNN_MODEL_PATH
    if model_type == "vgg16":
        return config.VGG16_MODEL_PATH
    raise PredictionError(f"Unknown model_type '{model_type}'. Use 'cnn' or 'vgg16'.")


def predict_image(image_path: str, model_type: str = "cnn") -> dict:
    """
    Predict the rice variety for a single image.

    Returns a dict:
        {
            "predicted_class": str,
            "confidence": float,        # 0-100
            "probabilities": {class_name: float, ...}  # each 0-100
        }

    Raises PredictionError for any recoverable problem (missing model,
    bad image, etc.) so callers (e.g. Streamlit) can show a friendly message.
    """
    import tensorflow as tf

    model_path = _resolve_model_path(model_type)
    if not os.path.exists(model_path):
        raise PredictionError(
            f"No trained {model_type.upper()} model found at '{model_path}'. "
            f"Train it first with `python src/train_{model_type}.py`."
        )

    class_names = load_class_names()
    if not class_names:
        raise PredictionError("No class names could be determined. Check the dataset/ folder.")

    x = _load_and_prepare_image(image_path, model_type)

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(x, verbose=0)[0]

    predicted_idx = int(np.argmax(probs))
    predicted_class = class_names[predicted_idx]
    confidence = float(probs[predicted_idx]) * 100.0

    probabilities = {
        class_names[i]: round(float(probs[i]) * 100.0, 2) for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Predict rice grain variety for a single image.")
    parser.add_argument("image_path", type=str, help="Path to the rice grain image.")
    parser.add_argument(
        "--model", type=str, default="cnn", choices=["cnn", "vgg16"], help="Which trained model to use."
    )
    args = parser.parse_args()

    try:
        result = predict_image(args.image_path, model_type=args.model)
    except PredictionError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"Predicted Variety: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print("Probability distribution:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {prob:.2f}%")


if __name__ == "__main__":
    _main()
