import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src import config


# --------------------------------------------------
# Model predictions
# --------------------------------------------------

def predict_dataset(model, dataset):
    """Generate predictions and collect true labels."""

    true_labels = []
    predicted_probabilities = []

    for images, labels in dataset:

        probabilities = model.predict(
            images,
            verbose=0
        )

        true_labels.extend(
            np.argmax(labels.numpy(), axis=1)
        )

        predicted_probabilities.extend(probabilities)

    true_labels = np.asarray(true_labels)
    predicted_probabilities = np.asarray(
        predicted_probabilities
    )

    predicted_labels = np.argmax(
        predicted_probabilities,
        axis=1
    )

    return (
        true_labels,
        predicted_labels,
        predicted_probabilities
    )


# --------------------------------------------------
# Classification metrics
# --------------------------------------------------

def calculate_metrics(
    true_labels,
    predicted_labels,
    color_space,
    seed
):
    """Calculate overall and per-class classification metrics."""

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=range(config.NUM_CLASSES),
        target_names=config.CLASSES,
        output_dict=True,
        zero_division=0
    )

    class_report = pd.DataFrame(report).transpose()

    summary = {
        "color_space": color_space,
        "seed": seed,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }

    return summary, class_report


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

def calculate_confusion_matrix(
    true_labels,
    predicted_labels
):
    """Generate confusion matrix using the configured class order."""

    return confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(config.NUM_CLASSES)
    )


# --------------------------------------------------
# Complete model evaluation
# --------------------------------------------------

def evaluate_model(
    model,
    test_dataset,
    color_space,
    seed=None
):
    """Evaluate a trained model on the test dataset."""

    if color_space not in config.COLOR_SPACES:
        raise ValueError(
            f"Unsupported colour space: {color_space}"
        )

    if seed is None:
        seed = config.DEFAULT_SEED

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=0
    )

    true_labels, predicted_labels, probabilities = (
        predict_dataset(model, test_dataset)
    )

    summary, class_report = calculate_metrics(
        true_labels,
        predicted_labels,
        color_space,
        seed
    )

    summary["test_loss"] = float(test_loss)
    summary["test_accuracy"] = float(test_accuracy)

    matrix = calculate_confusion_matrix(
        true_labels,
        predicted_labels
    )

    return {
        "summary": summary,
        "class_report": class_report,
        "confusion_matrix": matrix,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "predicted_probabilities": probabilities
    }


# --------------------------------------------------
# Saving evaluation results
# --------------------------------------------------

def save_evaluation_results(results, color_space, seed):
    """Save evaluation metrics and predictions."""

    summary_path = (
        config.METRICS_DIR /
        f"{color_space}_seed{seed}_evaluation.json"
    )

    report_path = (
        config.METRICS_DIR /
        f"{color_space}_seed{seed}_classification_report.csv"
    )

    confusion_path = (
        config.METRICS_DIR /
        f"{color_space}_seed{seed}_confusion_matrix.csv"
    )

    predictions_path = (
        config.METRICS_DIR /
        f"{color_space}_seed{seed}_predictions.csv"
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results["summary"], file, indent=4)

    results["class_report"].to_csv(report_path)

    confusion_df = pd.DataFrame(
        results["confusion_matrix"],
        index=config.CLASSES,
        columns=config.CLASSES
    )

    confusion_df.to_csv(confusion_path)

    predictions_df = pd.DataFrame({
        "true_label": [
            config.CLASSES[i]
            for i in results["true_labels"]
        ],
        "predicted_label": [
            config.CLASSES[i]
            for i in results["predicted_labels"]
        ]
    })

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    return {
        "summary": summary_path,
        "classification_report": report_path,
        "confusion_matrix": confusion_path,
        "predictions": predictions_path
    }
