# Beyond RGB: Artistic Style Classification

## Project overview

This project investigates how colour-space representation influences
the classification performance and generalisation of convolutional
neural networks when distinguishing between artistic styles.

## Research question

How does colour-space representation influence the classification
performance and generalisation of convolutional neural networks
when distinguishing between artistic styles?

## Dataset

The dataset contains paintings from three artistic styles:

- Monet
- Picasso
- Van Gogh

The experiments compare three image representations:

- RGB
- HSV
- Grayscale

## Experimental setup

A consistent CNN architecture and training configuration are used
to compare the colour-space representations.

## Project structure

- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for experiments and analysis
- `src/`: Reusable project source code
- `outputs/`: Models, metrics, figures, tables and logs
- `reports/`: Report-ready figures, tables and report drafts

## Reproducibility

Experiments use explicit random seeds and record relevant
environment and training configuration details.

## Technologies

Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib and scikit-learn.
