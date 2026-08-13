# Swin Transformer for Pathology Image Classification

## Overview

This project implements pathology image classification using the Swin Transformer architecture.

Two independent implementations are provided:

- TensorFlow / Keras
- PyTorch

The purpose of this project is to develop and evaluate a Swin Transformer-based deep learning model for pathology image classification and compare the results of implementations using different deep learning frameworks.

## Implementations

### TensorFlow

The TensorFlow implementation is provided in:

`train_tensorflow.py`

### PyTorch

The PyTorch implementation is provided in:

`train_pytorch.py`

## Project Structure

```text
train_tf_swin_pathology/
│
├── train_tensorflow.py
├── train_pytorch.py
├── README.md
├── .gitignore
│
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_test.png
│   ├── confusion_matrix_test_counts.png
│   └── confusion_matrix_test_normalized.png
│
└── history/
    └── history_all.json

Results

The repository contains training and evaluation visualizations, including:

Training and validation accuracy curves
Training and validation loss curves
Confusion matrices
Normalized confusion matrix
Test-set evaluation results
Requirements

The project requires Python and the corresponding deep learning libraries.

TensorFlow
pip install tensorflow
PyTorch

Install PyTorch according to your operating system and CUDA configuration.

Additional dependencies may include:

pip install numpy pandas matplotlib scikit-learn pillow
Usage
TensorFlow
python train_tensorflow.py
PyTorch
python train_pytorch.py
Evaluation

The project generates visualization files for evaluating model performance.

These include accuracy, loss, and confusion matrix plots.

Reproducibility

For reproducible experiments, it is recommended to use the same dataset split, preprocessing pipeline, image resolution, batch size, number of epochs, and random seed.

Author

Michael Askari