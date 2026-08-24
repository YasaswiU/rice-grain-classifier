"""
train_vgg16.py
--------------
Memory-efficient VGG16 transfer learning for Rice Grain Classification.

Training strategy:
    - 5,000 balanced training images
    - 1,000 balanced validation images
    - VGG16 ImageNet transfer learning
    - Phase 1: frozen VGG16 base
    - Phase 2: fine-tune last few layers

The full dataset is NOT loaded into RAM.
"""

from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from src import config
    from src.data_preprocessing import (
        collect_image_records,
        discover_classes,
        stratified_split,
    )
except ImportError:
    import config
    from data_preprocessing import (
        collect_image_records,
        discover_classes,
        stratified_split,
    )


# ============================================================
# SETTINGS FOR FAST CPU TRAINING
# ============================================================

VGG16_TRAIN_SAMPLES = 5000
VGG16_VAL_SAMPLES = 1000

BATCH_SIZE = 16

HEAD_EPOCHS = 3
FINE_TUNE_EPOCHS = 3

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seeds():

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


# ============================================================
# BALANCED SUBSET
# ============================================================

def create_balanced_subset(
    records,
    class_names,
    total_samples,
    seed=SEED,
):
    """
    Select approximately equal numbers of images
    from every class.
    """

    rng = random.Random(seed)

    records_by_class = {
        class_name: []
        for class_name in class_names
    }

    for record in records:

        if record.label in records_by_class:
            records_by_class[record.label].append(record)

    samples_per_class = total_samples // len(class_names)

    selected = []

    print(
        f"\n[INFO] Selecting approximately "
        f"{samples_per_class} images per class..."
    )

    for class_name in class_names:

        class_records = records_by_class[class_name][:]

        rng.shuffle(class_records)

        selected_records = class_records[
            :samples_per_class
        ]

        selected.extend(selected_records)

        print(
            f"    {class_name}: "
            f"{len(selected_records)} images"
        )

    rng.shuffle(selected)

    return selected


# ============================================================
# MEMORY-EFFICIENT GENERATOR
# ============================================================

class RiceDataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        records,
        class_names,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=False,
    ):

        self.records = records
        self.class_names = class_names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        self.class_to_idx = {
            name: index
            for index, name in enumerate(class_names)
        }

        self.indices = np.arange(
            len(self.records)
        )

        if augment:

            self.datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.08,
                height_shift_range=0.08,
                zoom_range=0.08,
                horizontal_flip=True,
                fill_mode="nearest",
            )

        else:

            self.datagen = None

        self.on_epoch_end()

    def __len__(self):

        return math.ceil(
            len(self.records) /
            self.batch_size
        )

    def __getitem__(self, index):

        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_records = [
            self.records[i]
            for i in batch_indices
        ]

        X = np.zeros(
            (
                len(batch_records),
                config.IMAGE_SIZE[0],
                config.IMAGE_SIZE[1],
                config.IMAGE_CHANNELS,
            ),
            dtype=np.float32,
        )

        y = np.zeros(
            len(batch_records),
            dtype=np.int32,
        )

        for i, record in enumerate(batch_records):

            try:

                image = tf.keras.utils.load_img(
                    record.path,
                    target_size=config.IMAGE_SIZE,
                )

                image = tf.keras.utils.img_to_array(
                    image
                )

                # Apply augmentation before VGG preprocessing
                if self.augment:

                    image = self.datagen.random_transform(
                        image
                    )

                image = preprocess_input(
                    image
                )

                X[i] = image

                y[i] = self.class_to_idx[
                    record.label
                ]

            except Exception as error:

                print(
                    f"[WARNING] Could not load:"
                    f"\n{record.path}"
                )

                print(error)

        return X, y

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# BUILD VGG16
# ============================================================

def build_vgg16_model(
    num_classes,
    input_shape,
):

    print(
        "\n[INFO] Loading VGG16 ImageNet weights..."
    )

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Phase 1
    base_model.trainable = False

    inputs = layers.Input(
        shape=input_shape
    )

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.4
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="rice_vgg16_transfer"
    )

    return model, base_model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks():

    return [

        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),

        ModelCheckpoint(
            filepath=config.VGG16_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    set_seeds()

    print("\n" + "=" * 70)
    print("RICE GRAIN CLASSIFIER - VGG16")
    print("=" * 70)

    # --------------------------------------------------------
    # Discover classes
    # --------------------------------------------------------

    print("\n[INFO] Discovering classes...")

    class_names = discover_classes()

    print(
        f"[INFO] Classes ({len(class_names)}):"
        f" {class_names}"
    )

    # --------------------------------------------------------
    # Collect records
    # --------------------------------------------------------

    print(
        "\n[INFO] Collecting image records..."
    )

    records = collect_image_records()

    print(
        f"[INFO] Total valid images: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Original stratified split
    # --------------------------------------------------------

    (
        train_records,
        val_records,
        test_records,
    ) = stratified_split(records)

    print(
        f"\n[INFO] Original split:"
    )

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
    # Balanced subsets
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING BALANCED VGG16 SUBSETS"
    )

    print(
        "=" * 70
    )

    train_subset = create_balanced_subset(
        train_records,
        class_names,
        VGG16_TRAIN_SAMPLES,
    )

    val_subset = create_balanced_subset(
        val_records,
        class_names,
        VGG16_VAL_SAMPLES,
        seed=SEED + 1,
    )

    print(
        f"\n[INFO] VGG16 training images:"
        f" {len(train_subset)}"
    )

    print(
        f"[INFO] VGG16 validation images:"
        f" {len(val_subset)}"
    )

    # --------------------------------------------------------
    # Generators
    # --------------------------------------------------------

    print(
        "\n[INFO] Creating image generators..."
    )

    train_gen = RiceDataGenerator(
        train_subset,
        class_names,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
    )

    val_gen = RiceDataGenerator(
        val_subset,
        class_names,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,
    )

    print(
        f"[INFO] Training batches per epoch:"
        f" {len(train_gen)}"
    )

    print(
        f"[INFO] Validation batches per epoch:"
        f" {len(val_gen)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    input_shape = (
        config.IMAGE_SIZE[0],
        config.IMAGE_SIZE[1],
        config.IMAGE_CHANNELS,
    )

    model, base_model = build_vgg16_model(
        num_classes=len(class_names),
        input_shape=input_shape,
    )

    # ========================================================
    # PHASE 1
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 1 - FROZEN VGG16"
    )

    print(
        "=" * 70
    )

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=1e-4
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = get_callbacks()

    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ========================================================
    # PHASE 2
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 2 - FINE-TUNING"
    )

    print(
        "=" * 70
    )

    base_model.trainable = True

    # Freeze all but last 4 layers
    for layer in base_model.layers[
        :-4
    ]:

        layer.trainable = False

    # Keep BatchNormalization frozen
    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model.save(
        config.VGG16_MODEL_PATH
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
    ) as file:

        json.dump(
            class_names,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Merge history
    # --------------------------------------------------------

    history = {}

    keys = set(
        history_head.history.keys()
    ).union(
        history_fine.history.keys()
    )

    for key in keys:

        history[key] = (
            [
                float(value)
                for value in history_head.history.get(
                    key,
                    []
                )
            ]
            +
            [
                float(value)
                for value in history_fine.history.get(
                    key,
                    []
                )
            ]
        )

    with open(
        config.VGG16_HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "VGG16 TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nModel:"
    )

    print(
        config.VGG16_MODEL_PATH
    )

    print(
        "\nHistory:"
    )

    print(
        config.VGG16_HISTORY_PATH
    )

    print(
        "\nClass names:"
    )

    print(
        class_map_path
    )

    print(
        "\nNext step:"
    )

    print(
        "Run the evaluation script."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()