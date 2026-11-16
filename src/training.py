import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config
from src.models import build_cnn, compile_cnn


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_random_seed(seed=None):
    """Set random seeds across Python, NumPy and TensorFlow."""

    if seed is None:
        seed = config.DEFAULT_SEED

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    return seed


# --------------------------------------------------
# Training callbacks
# --------------------------------------------------

def create_training_callbacks(color_space, seed):
    """Create callbacks used during model training."""

    checkpoint_path = (
        config.MODELS_DIR /
        f"{color_space}_seed{seed}.keras"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks, checkpoint_path


# --------------------------------------------------
# Model training
# --------------------------------------------------

def train_model(
    train_dataset,
    validation_dataset,
    color_space,
    seed=None,
    epochs=None
):
    """
    Train the CNN using the project's standard configuration.

    Parameters
    ----------
    train_dataset : tf.data.Dataset
        Training dataset.
    validation_dataset : tf.data.Dataset
        Validation dataset.
    color_space : str
        Image representation used in the experiment.
    seed : int, optional
        Random seed.
    epochs : int, optional
        Maximum number of training epochs.

    Returns
    -------
    tuple
        Trained model, training history, and metadata.
    """

    if color_space not in config.COLOR_SPACES:
        raise ValueError(
            f"Unsupported colour space: {color_space}"
        )

    if seed is None:
        seed = config.DEFAULT_SEED

    if epochs is None:
        epochs = config.MAX_EPOCHS

    set_random_seed(seed)

    model = compile_cnn(build_cnn())

    callbacks, checkpoint_path = create_training_callbacks(
        color_space,
        seed
    )

    start_time = time.perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.perf_counter() - start_time

    # Ensure the best checkpoint weights are used
    if checkpoint_path.exists():
        model = tf.keras.models.load_model(checkpoint_path)

    history_df = pd.DataFrame(history.history)

    history_path = (
        config.METRICS_DIR /
        f"{color_space}_seed{seed}_history.csv"
    )

    history_df.to_csv(history_path, index=False)

    metadata = {
        "color_space": color_space,
        "seed": seed,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": int(
            np.argmin(history.history["val_loss"]) + 1
        ),
        "training_time_seconds": training_time,
        "model_path": str(checkpoint_path),
        "history_path": str(history_path),
        "parameter_count": model.count_params()
    }

    return model, history, metadata
