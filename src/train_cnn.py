"""
train_cnn.py
------------
Builds, trains, and saves a custom CNN for rice grain variety classification.

The CNN uses balanced subsets of the dataset to keep training
memory-efficient and practical on CPU.

Run from the project root:
    python src/train_cnn.py
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

try:
    from src import config
    from src.data_preprocessing import (
        ImageRecord,
        collect_image_records,
        discover_classes,
        stratified_split,
        records_to_arrays,
        build_train_augmentor,
        build_eval_augmentor,
    )
except ImportError:
    import config
    from data_preprocessing import (
        ImageRecord,
        collect_image_records,
        discover_classes,
        stratified_split,
        records_to_arrays,
        build_train_augmentor,
        build_eval_augmentor,
    )


# ============================================================
# SETTINGS SPECIFIC TO CUSTOM CNN
# ============================================================

CNN_IMAGE_SIZE = (128, 128)

# Number of images used per class
TRAIN_IMAGES_PER_CLASS = 1000
VAL_IMAGES_PER_CLASS = 200

# Keep CNN training short and practical
CNN_EPOCHS = 10

# Batch size
CNN_BATCH_SIZE = 32


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# BALANCED SUBSET CREATION
# ============================================================

def create_balanced_subset(
    records: list[ImageRecord],
    class_names: list[str],
    images_per_class: int,
    seed: int = config.RANDOM_SEED,
) -> list[ImageRecord]:
    """
    Select an equal number of images from every class.

    If a class contains fewer images than requested, all available
    images from that class are used.
    """

    rng = random.Random(seed)

    by_class: dict[str, list[ImageRecord]] = {
        class_name: [] for class_name in class_names
    }

    for record in records:
        if record.label in by_class:
            by_class[record.label].append(record)

    selected: list[ImageRecord] = []

    for class_name in class_names:
        items = by_class[class_name][:]

        rng.shuffle(items)

        count = min(images_per_class, len(items))

        selected.extend(items[:count])

        print(
            f"    {class_name}: {count} images"
        )

    rng.shuffle(selected)

    return selected


# ============================================================
# CUSTOM CNN MODEL
# ============================================================

def build_custom_cnn(
    num_classes: int,
    input_shape: tuple[int, int, int],
) -> tf.keras.Model:
    """
    Efficient custom CNN for rice grain classification.

    Four convolutional blocks are used, followed by
    GlobalAveragePooling and a classification head.
    """

    inputs = layers.Input(shape=input_shape)

    x = inputs

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------
    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # --------------------------------------------------------
    # Block 4
    # --------------------------------------------------------
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu",
    )(x)

    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="rice_custom_cnn",
    )

    return model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(checkpoint_path: str) -> list:
    """Create training callbacks."""

    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),

        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main() -> None:

    set_seeds()

    print()
    print("=" * 70)
    print("RICE GRAIN CLASSIFIER - CUSTOM CNN")
    print("=" * 70)

    # --------------------------------------------------------
    # Discover classes
    # --------------------------------------------------------

    print()
    print("[INFO] Discovering classes...")

    class_names = discover_classes()

    print(
        f"[INFO] Classes ({len(class_names)}): {class_names}"
    )

    # --------------------------------------------------------
    # Collect records
    # --------------------------------------------------------

    print()
    print("[INFO] Collecting image records...")

    records = collect_image_records()

    print(
        f"[INFO] Total valid images: {len(records)}"
    )

    # --------------------------------------------------------
    # Original stratified split
    # --------------------------------------------------------

    train_records, val_records, test_records = stratified_split(
        records
    )

    print()
    print("[INFO] Original dataset split:")
    print(
        f"    Train: {len(train_records)}"
    )
    print(
        f"    Validation: {len(val_records)}"
    )
    print(
        f"    Test: {len(test_records)}"
    )

    # --------------------------------------------------------
    # Balanced CNN training subset
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING BALANCED CNN SUBSETS")
    print("=" * 70)

    print()
    print(
        f"[INFO] Selecting approximately "
        f"{TRAIN_IMAGES_PER_CLASS} images per class for training..."
    )

    cnn_train_records = create_balanced_subset(
        train_records,
        class_names,
        TRAIN_IMAGES_PER_CLASS,
        seed=config.RANDOM_SEED,
    )

    print()
    print(
        f"[INFO] Selecting approximately "
        f"{VAL_IMAGES_PER_CLASS} images per class for validation..."
    )

    cnn_val_records = create_balanced_subset(
        val_records,
        class_names,
        VAL_IMAGES_PER_CLASS,
        seed=config.RANDOM_SEED + 1,
    )

    print()
    print(
        f"[INFO] CNN training images: "
        f"{len(cnn_train_records)}"
    )

    print(
        f"[INFO] CNN validation images: "
        f"{len(cnn_val_records)}"
    )

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print()
    print(
        "[INFO] Loading CNN training images..."
    )

    X_train, y_train = records_to_arrays(
        cnn_train_records,
        class_names,
        CNN_IMAGE_SIZE,
    )

    print(
        f"[INFO] X_train shape: {X_train.shape}"
    )

    print()
    print(
        "[INFO] Loading CNN validation images..."
    )

    X_val, y_val = records_to_arrays(
        cnn_val_records,
        class_names,
        CNN_IMAGE_SIZE,
    )

    print(
        f"[INFO] X_val shape: {X_val.shape}"
    )

    # --------------------------------------------------------
    # Data generators
    # --------------------------------------------------------

    print()
    print("[INFO] Creating data generators...")

    train_gen = build_train_augmentor().flow(
        X_train,
        y_train,
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    val_gen = build_eval_augmentor().flow(
        X_val,
        y_val,
        batch_size=CNN_BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"[INFO] Training batches per epoch: "
        f"{len(train_gen)}"
    )

    print(
        f"[INFO] Validation batches per epoch: "
        f"{len(val_gen)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    input_shape = (
        CNN_IMAGE_SIZE[0],
        CNN_IMAGE_SIZE[1],
        config.IMAGE_CHANNELS,
    )

    print()
    print("[INFO] Building custom CNN...")

    model = build_custom_cnn(
        num_classes=len(class_names),
        input_shape=input_shape,
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=config.LEARNING_RATE_CNN
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print()
    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = get_callbacks(
        config.CNN_MODEL_PATH
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING CNN TRAINING")
    print("=" * 70)

    print()
    print(
        f"[INFO] Image size: {CNN_IMAGE_SIZE}"
    )

    print(
        f"[INFO] Training images: {len(X_train)}"
    )

    print(
        f"[INFO] Validation images: {len(X_val)}"
    )

    print(
        f"[INFO] Epochs: {CNN_EPOCHS}"
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CNN_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------------
    # Ensure model exists
    # --------------------------------------------------------

    if not os.path.exists(
        config.CNN_MODEL_PATH
    ):
        model.save(
            config.CNN_MODEL_PATH
        )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    class_map_path = os.path.join(
        config.MODELS_DIR,
        "class_names.json",
    )

    with open(
        class_map_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            class_names,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(
        config.CNN_HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history_dict,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_epoch = (
        history.history["val_accuracy"].index(
            best_val_accuracy
        ) + 1
    )

    print()
    print("=" * 70)
    print("CNN TRAINING COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print()
    print(
        f"Model:"
    )
    print(
        config.CNN_MODEL_PATH
    )

    print()
    print(
        f"History:"
    )
    print(
        config.CNN_HISTORY_PATH
    )

    print()
    print(
        f"Class names:"
    )
    print(
        class_map_path
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run: python src/evaluate.py"
    )


if __name__ == "__main__":
    main()