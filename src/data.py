import zipfile
from pathlib import Path

import tensorflow as tf

from src import config


# --------------------------------------------------
# Dataset extraction
# --------------------------------------------------

def extract_dataset():
    """Extract the dataset archive if it has not already been extracted."""

    archive_path = config.DATA_ARCHIVE
    destination = config.DATA_DIR

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {archive_path}"
        )

    if destination.exists() and any(destination.iterdir()):
        print("Dataset already extracted.")
        return destination

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(destination)

    print(f"Dataset extracted to: {destination}")

    return destination


# --------------------------------------------------
# Dataset directory validation
# --------------------------------------------------

def get_dataset_root():
    """Locate the directory containing the train, validation and test folders."""

    expected_splits = ["train", "validation", "test"]

    for candidate in [config.DATA_DIR, *config.DATA_DIR.iterdir()]:
        if not candidate.is_dir():
            continue

        if all((candidate / split).is_dir() for split in expected_splits):
            return candidate

    raise FileNotFoundError(
        "Could not locate train, validation and test directories "
        f"inside {config.DATA_DIR}"
    )


def validate_dataset_structure():
    """Check that all expected class directories exist."""

    dataset_root = get_dataset_root()

    for split in ["train", "validation", "test"]:
        split_path = dataset_root / split

        for class_name in config.CLASSES:
            class_path = split_path / class_name

            if not class_path.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_path}"
                )

    print("Dataset structure validated successfully.")

    return dataset_root


# --------------------------------------------------
# Image loading
# --------------------------------------------------

def load_split(split_name, shuffle=False, seed=None):
    """Load one dataset split using a consistent image configuration."""

    dataset_root = get_dataset_root()
    split_path = dataset_root / split_name

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_path,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASSES,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=seed,
    )

    return dataset


# --------------------------------------------------
# Colour-space transformations
# --------------------------------------------------

def transform_images(images, color_space):
    """Transform RGB images into the requested colour-space representation."""

    images = tf.cast(images, tf.float32) / 255.0

    if color_space == "rgb":
        return images

    elif color_space == "hsv":
        return tf.image.rgb_to_hsv(images)

    elif color_space == "grayscale":
        grayscale = tf.image.rgb_to_grayscale(images)
        return tf.repeat(grayscale, repeats=3, axis=-1)

    else:
        raise ValueError(
            f"Unsupported colour space: {color_space}"
        )


def prepare_dataset(dataset, color_space):
    """Apply the selected colour-space transformation to a dataset."""

    return dataset.map(
        lambda images, labels: (
            transform_images(images, color_space),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# Complete dataset preparation
# --------------------------------------------------

def load_datasets(color_space, seed=None):
    """Load and prepare training, validation and test datasets."""

    if color_space not in config.COLOR_SPACES:
        raise ValueError(
            f"Unsupported colour space: {color_space}"
        )

    if seed is None:
        seed = config.DEFAULT_SEED

    train_raw = load_split(
        "train",
        shuffle=True,
        seed=seed
    )

    validation_raw = load_split(
        "validation",
        shuffle=False
    )

    test_raw = load_split(
        "test",
        shuffle=False
    )

    train = prepare_dataset(train_raw, color_space)
    validation = prepare_dataset(validation_raw, color_space)
    test = prepare_dataset(test_raw, color_space)

    return train, validation, test


# --------------------------------------------------
# Dataset integrity auditing
# --------------------------------------------------

import hashlib
import pandas as pd


def calculate_sha256(file_path):
    """Calculate the SHA-256 hash of an image file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def audit_dataset():
    """
    Audit dataset structure, image counts and exact duplicates.

    Returns
    -------
    dict
        Dictionary containing image records, class counts,
        duplicate records and summary statistics.
    """

    dataset_root = get_dataset_root()

    expected_splits = ["train", "validation", "test"]

    image_extensions = {".jpg", ".jpeg", ".png"}

    records = []
    count_records = []

    # Inspect dataset structure and image files
    for split in expected_splits:

        split_path = dataset_root / split

        for class_name in config.CLASSES:

            class_path = split_path / class_name

            if not class_path.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_path}"
                )

            image_files = sorted(
                file for file in class_path.iterdir()
                if file.is_file()
                and file.suffix.lower() in image_extensions
            )

            count_records.append({
                "split": split,
                "class": class_name,
                "image_count": len(image_files)
            })

            for image_path in image_files:

                records.append({
                    "split": split,
                    "class": class_name,
                    "filename": image_path.name,
                    "path": str(image_path),
                    "sha256": calculate_sha256(image_path)
                })

    # Convert results into DataFrames
    image_records = pd.DataFrame(records)
    class_counts = pd.DataFrame(count_records)

    # Find all exact duplicate files
    duplicates = image_records[
        image_records.duplicated(
            subset="sha256",
            keep=False
        )
    ].sort_values("sha256").reset_index(drop=True)

    # Summary statistics
    total_images = len(image_records)

    unique_images = (
        image_records["sha256"].nunique()
        if total_images > 0 else 0
    )

    duplicate_groups = (
        image_records["sha256"].duplicated().sum()
        if total_images > 0 else 0
    )

    summary = {
        "total_images": total_images,
        "unique_images": unique_images,
        "duplicate_groups": int(duplicate_groups),
        "duplicate_records": len(duplicates),
        "integrity_passed": duplicates.empty
    }

    return {
        "summary": summary,
        "class_counts": class_counts,
        "image_records": image_records,
        "duplicates": duplicates
    }


def print_dataset_audit(results):
    """Display a readable summary of dataset audit results."""

    print("DATASET INTEGRITY AUDIT")
    print("=" * 40)

    print("\nImage counts:")

    counts = results["class_counts"].pivot(
        index="split",
        columns="class",
        values="image_count"
    )

    counts["Total"] = counts.sum(axis=1)

    print(counts.to_string())

    print("\nOverall summary:")

    for key, value in results["summary"].items():
        print(f"{key}: {value}")

    if results["summary"]["integrity_passed"]:
        print("\nNo exact duplicate images found.")
    else:
        print("\nWARNING: Exact duplicate images detected.")
        print(results["duplicates"][
            ["split", "class", "filename", "sha256"]
        ].to_string(index=False))
