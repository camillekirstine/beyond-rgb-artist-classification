
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src import config


def set_publication_style():
    """Apply consistent styling to all project figures."""

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2
    )

    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True
    })


def save_figure(fig, filename, save=True):
    """Save a figure as high-resolution PNG and PDF."""

    if not save:
        return

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    png_path = config.FIGURES_DIR / f"{filename}.png"
    pdf_path = config.FIGURES_DIR / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    report_png = config.REPORT_FIGURES_DIR / f"{filename}.png"
    report_pdf = config.REPORT_FIGURES_DIR / f"{filename}.pdf"

    fig.savefig(report_png, dpi=300, bbox_inches="tight")
    fig.savefig(report_pdf, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_training_history(history, color_space, seed, save=True):
    """Plot training and validation accuracy and loss.

    Accepts either a Keras History object or a pandas DataFrame
    loaded from a previously saved training history CSV.
    """

    set_publication_style()

    # Support both Keras History objects and DataFrames.
    if hasattr(history, "history"):
        history_data = history.history
    else:
        history_data = history.to_dict(orient="list")

    epochs = range(1, len(history_data["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy
    axes[0].plot(
        epochs,
        history_data["accuracy"],
        label="Training accuracy"
    )
    axes[0].plot(
        epochs,
        history_data["val_accuracy"],
        label="Validation accuracy"
    )

    axes[0].set_title("Model accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(
        epochs,
        history_data["loss"],
        label="Training loss"
    )
    axes[1].plot(
        epochs,
        history_data["val_loss"],
        label="Validation loss"
    )

    axes[1].set_title("Model loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(
        f"Training history: {color_space.upper()} (seed {seed})",
        fontsize=15
    )

    save_figure(
        fig,
        f"training_curves_{color_space}_seed{seed}",
        save
    )

    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    confusion_matrix,
    color_space,
    seed,
    normalize=False,
    save=True
):
    """Plot a confusion matrix with optional row-wise normalization."""

    set_publication_style()

    matrix = np.array(confusion_matrix)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)

        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0
        )

        annotations = np.array([
            [f"{value:.1%}" for value in row]
            for row in matrix
        ])

        value_format = ""
        title_suffix = " (Normalized)"

    else:
        annotations = True
        value_format = "d"
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=annotations,
        fmt=value_format,
        cmap="Blues",
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        cbar=True,
        ax=ax
    )

    ax.set_title(
        f"{color_space.upper()} Confusion Matrix{title_suffix}"
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    save_figure(
        fig,
        f"confusion_matrix_{color_space}_seed{seed}"
        + ("_normalized" if normalize else ""),
        save
    )

    plt.show()
    plt.close(fig)


def plot_class_metrics(class_report, color_space, seed, save=True):
    """Compare precision, recall and F1-score across classes."""

    set_publication_style()

    metrics = class_report.copy()

    # Exclude aggregate rows from the class-level comparison.
    metrics = metrics.loc[
        metrics.index.isin(config.CLASSES),
        ["precision", "recall", "f1-score"]
    ]

    metrics = metrics.rename(columns={
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1-score"
    })

    fig, ax = plt.subplots(figsize=(9, 5))

    metrics.plot(
        kind="bar",
        ax=ax,
        width=0.75
    )

    ax.set_title(
        f"Per-class performance: {color_space.upper()}"
    )
    ax.set_xlabel("Artistic style")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric")
    ax.tick_params(axis="x", rotation=0)

    save_figure(
        fig,
        f"class_metrics_{color_space}_seed{seed}",
        save
    )

    plt.show()
    plt.close(fig)
