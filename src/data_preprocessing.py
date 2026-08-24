"""
data_preprocessing.py
---------------------
Memory-efficient data preprocessing for rice grain classification.

Important:
- Images are NOT loaded into RAM all at once.
- Only file paths and labels are stored in memory.
- Images are loaded batch-by-batch during training.
- Classes are discovered dynamically from dataset folders.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from src import config
except ImportError:
    import config


# ============================================================
# IMAGE RECORD
# ============================================================

@dataclass
class ImageRecord:
    path: str
    label: str


# ============================================================
# CLASS DISCOVERY
# ============================================================

def discover_classes(dataset_dir: str = config.DATASET_DIR) -> list[str]:
    """
    Discover class names from sub-folders inside dataset/.
    """

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    classes = []

    for entry in sorted(os.listdir(dataset_dir)):

        full_path = os.path.join(dataset_dir, entry)

        if not os.path.isdir(full_path):
            continue

        if _folder_has_images(full_path):
            classes.append(entry)

    return classes


def _folder_has_images(folder: str) -> bool:

    for fname in os.listdir(folder):

        if fname.lower().endswith(config.SUPPORTED_EXTENSIONS):
            return True

    return False


# ============================================================
# IMAGE VALIDATION
# ============================================================

def _is_valid_image(path: str) -> bool:

    try:

        with Image.open(path) as img:
            img.verify()

        return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        return False


# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

def collect_image_records(
    dataset_dir: str = config.DATASET_DIR
) -> list[ImageRecord]:
    """
    Collect image paths without loading images into RAM.
    """

    classes = discover_classes(dataset_dir)

    if len(classes) == 0:

        raise ValueError(
            f"No class folders found inside {dataset_dir}"
        )

    records = []

    skipped = 0

    for class_name in classes:

        class_dir = os.path.join(
            dataset_dir,
            class_name
        )

        for fname in sorted(os.listdir(class_dir)):

            if not fname.lower().endswith(
                config.SUPPORTED_EXTENSIONS
            ):
                continue

            full_path = os.path.join(
                class_dir,
                fname
            )

            if _is_valid_image(full_path):

                records.append(
                    ImageRecord(
                        path=full_path,
                        label=class_name
                    )
                )

            else:

                skipped += 1

                print(
                    f"[WARNING] Skipping invalid image: {full_path}"
                )

    if skipped:

        print(
            f"[INFO] Skipped {skipped} invalid image(s)."
        )

    if len(records) == 0:

        raise ValueError(
            "No valid images found."
        )

    return records


# ============================================================
# STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def stratified_split(
    records: list[ImageRecord],
    train_split: float = config.TRAIN_SPLIT,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
):

    total = (
        train_split
        + val_split
        + test_split
    )

    if not np.isclose(total, 1.0):

        raise ValueError(
            f"Split ratios must sum to 1.0. Got {total}"
        )

    rng = random.Random(seed)

    by_class = {}

    for record in records:

        by_class.setdefault(
            record.label,
            []
        ).append(record)

    train = []
    val = []
    test = []

    for label, items in by_class.items():

        items = items.copy()

        rng.shuffle(items)

        n = len(items)

        n_train = int(
            round(n * train_split)
        )

        n_val = int(
            round(n * val_split)
        )

        train.extend(
            items[:n_train]
        )

        val.extend(
            items[
                n_train:
                n_train + n_val
            ]
        )

        test.extend(
            items[
                n_train + n_val:
            ]
        )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ============================================================
# LOAD ONE IMAGE
# ============================================================

def load_image_as_array(
    path: str,
    image_size: tuple[int, int] = config.IMAGE_SIZE
):

    with Image.open(path) as img:

        img = img.convert("RGB")

        img = img.resize(image_size)

        arr = np.asarray(
            img,
            dtype=np.float32
        )

    return arr


# ============================================================
# OPTIONAL ARRAY FUNCTION
# ============================================================

def records_to_arrays(
    records: list[ImageRecord],
    class_names: list[str],
    image_size: tuple[int, int] = config.IMAGE_SIZE,
):

    """
    Converts records to arrays.

    WARNING:
    Do NOT use this function for the full VGG16 dataset.
    It is kept for compatibility with the Custom CNN.
    """

    class_to_idx = {
        name: index
        for index, name in enumerate(class_names)
    }

    X = np.zeros(
        (
            len(records),
            image_size[0],
            image_size[1],
            3
        ),
        dtype=np.float32
    )

    y = np.zeros(
        len(records),
        dtype=np.int32
    )

    for i, record in enumerate(records):

        X[i] = (
            load_image_as_array(
                record.path,
                image_size
            ) / 255.0
        )

        y[i] = class_to_idx[
            record.label
        ]

    return X, y


# ============================================================
# TRAIN AUGMENTATION
# ============================================================

def build_train_augmentor():
    """
    Returns a Keras ImageDataGenerator configured for training.
    """

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode="nearest",
    )


# ============================================================
# EVALUATION AUGMENTATION
# ============================================================

def build_eval_augmentor():
    """
    Returns an ImageDataGenerator without augmentation.
    """

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator()


# ============================================================
# DATASET STATISTICS
# ============================================================

def dataset_statistics(
    records: list[ImageRecord]
):

    stats = {}

    for record in records:

        stats[record.label] = (
            stats.get(record.label, 0) + 1
        )

    return dict(
        sorted(stats.items())
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary():

    records = collect_image_records()

    classes = discover_classes()

    train, val, test = stratified_split(
        records
    )

    print(
        f"Discovered {len(classes)} classes:"
    )

    print(classes)

    print(
        f"Total valid images: {len(records)}"
    )

    print(
        "Class distribution:"
    )

    for cls, count in dataset_statistics(
        records
    ).items():

        print(
            f"  {cls}: {count}"
        )

    print(
        f"Train: {len(train)}"
    )

    print(
        f"Validation: {len(val)}"
    )

    print(
        f"Test: {len(test)}"
    )


# ============================================================
# SAMPLE VISUALIZATION
# ============================================================

def plot_sample_images(
    records: list[ImageRecord],
    class_names: list[str],
    samples_per_class: int = 3
):

    import matplotlib.pyplot as plt

    by_class = {}

    for record in records:

        by_class.setdefault(
            record.label,
            []
        ).append(record)

    n_classes = len(class_names)

    fig, axes = plt.subplots(
        n_classes,
        samples_per_class,
        figsize=(
            3 * samples_per_class,
            3 * n_classes
        )
    )

    if n_classes == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, class_name in enumerate(
        class_names
    ):

        items = by_class.get(
            class_name,
            []
        )[:samples_per_class]

        for col in range(
            samples_per_class
        ):

            ax = axes[row][col]

            if col < len(items):

                img = Image.open(
                    items[col].path
                ).convert("RGB")

                ax.imshow(img)

            ax.set_title(
                class_name
                if col == 0
                else ""
            )

            ax.axis("off")

    fig.tight_layout()

    return fig


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_dataset_summary()