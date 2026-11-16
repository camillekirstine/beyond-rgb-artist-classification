
from src import config
from src.data import load_datasets
from src.training import train_model
from src.evaluation import evaluate_model, save_evaluation_results


def run_experiment(color_space, seed=None, epochs=None):
    """
    Execute a complete CNN experiment for a selected colour space.

    Parameters
    ----------
    color_space : str
        Input colour representation: rgb, hsv or grayscale.

    seed : int, optional
        Random seed used for reproducibility.

    epochs : int, optional
        Maximum number of training epochs.

    Returns
    -------
    dict
        Model, training history, training metadata and evaluation results.
    """

    if color_space not in config.COLOR_SPACES:
        raise ValueError(
            f"Unsupported colour space: {color_space}. "
            f"Choose from {config.COLOR_SPACES}."
        )

    if seed is None:
        seed = config.DEFAULT_SEED

    print("=" * 60)
    print(f"Starting experiment: {color_space.upper()}")
    print(f"Random seed: {seed}")
    print("=" * 60)

    # Load datasets in the selected colour space.
    print("\nLoading datasets...")

    train_dataset, validation_dataset, test_dataset = load_datasets(
        color_space=color_space,
        seed=seed
    )

    # Train the CNN.
    print("\nTraining model...")

    model, history, training_metadata = train_model(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        color_space=color_space,
        seed=seed,
        epochs=epochs
    )

    # Evaluate the trained model.
    print("\nEvaluating model...")

    evaluation_results = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        color_space=color_space,
        seed=seed
    )

    # Save evaluation results.
    save_evaluation_results(
        results=evaluation_results,
        color_space=color_space,
        seed=seed
    )

    print("\nExperiment completed.")
    print(f"Colour space: {color_space.upper()}")
    print(f"Seed: {seed}")
    print(f"Best epoch: {training_metadata['best_epoch']}")
    print(
        f"Test accuracy: "
        f"{evaluation_results['summary']['accuracy']:.4f}"
    )

    return {
        "color_space": color_space,
        "seed": seed,
        "model": model,
        "history": history,
        "training_metadata": training_metadata,
        "evaluation_results": evaluation_results
    }
