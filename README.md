````markdown
# Gastric Cancer Pathology Image Classification with Swin Transformer

## Overview

This project implements gastric cancer pathology image classification using the Swin Transformer Tiny architecture.

Two independent deep learning implementations are provided:

- TensorFlow / Keras
- PyTorch

The goal of this project is to develop and evaluate Swin Transformer-based models for multi-class pathology image classification and provide implementations using two major deep learning frameworks.

---

## Dataset

The dataset consists of pathology images organized into eight classes:

- ADI
- DEB
- LYM
- MUC
- MUS
- NOR
- STR
- TUM

The dataset is split into:

- 70% Training
- 15% Validation
- 15% Test

Stratified splitting is used to preserve class distributions across the datasets.

> The dataset itself is not included in this repository.

---

## Model

The main architecture used in this project is:

**Swin Transformer Tiny**

Input image resolution:

```text
224 × 224
````

Transfer learning is used for initialization, followed by fine-tuning of the model backbone.

---

## Implementations

### TensorFlow / Keras

The TensorFlow implementation is available in:

```text
train_tensorflow.py
```

The implementation includes:

* Swin Transformer Tiny
* Transfer learning
* Fine-tuning
* Data augmentation
* Class weighting
* Mixed precision training
* Early stopping
* Learning-rate scheduling
* Test-Time Augmentation (TTA)
* Confusion matrix analysis

Run:

```bash
python train_tensorflow.py
```

---

### PyTorch

The PyTorch implementation is available in:

```text
train_pytorch.py
```

The implementation includes:

* Swin Transformer Tiny
* Transfer learning
* Fine-tuning
* Data augmentation
* Weighted sampling
* Label smoothing
* Mixed precision training
* Early stopping
* Balanced accuracy evaluation

Run:

```bash
python train_pytorch.py
```

---

## Evaluation Metrics

Model performance is evaluated using:

* Accuracy
* Balanced Accuracy
* Macro F1-score
* Weighted F1-score
* Classification Report
* Confusion Matrix

---

## Results

The repository includes training and evaluation visualizations:

### Training Curves

* `accuracy_curve.png`
* `loss_curve.png`

### Confusion Matrices

* `confusion_matrix_test.png`
* `confusion_matrix_test_counts.png`
* `confusion_matrix_test_normalized.png`

### Training History

* `history_all.json`

---

## Project Structure

```text
gastric-cancer-pathology-classification/
│
├── README.md
├── .gitignore
│
├── train_tensorflow.py
├── train_pytorch.py
│
├── history_all.json
│
├── accuracy_curve.png
├── loss_curve.png
│
├── confusion_matrix_test.png
├── confusion_matrix_test_counts.png
└── confusion_matrix_test_normalized.png
```

---

## Requirements

Python 3.x is required.

### TensorFlow

```bash
pip install tensorflow
```

### PyTorch

Install PyTorch according to your operating system and CUDA configuration.

### Additional Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn pillow
```

For the TensorFlow implementation, the Swin Transformer package used by the project must also be installed.

---

## Reproducibility

The experiments use a fixed random seed:

```text
SEED = 42
```

For reproducible results, use the same:

* Dataset
* Dataset split
* Random seed
* Image resolution
* Batch size
* Data augmentation
* Training configuration
* Model architecture

---

## Author

Michael Askari
[2]: https://docs.github.com/en/enterprise-cloud%40latest/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes?utm_source=chatgpt.com "About the repository README file - GitHub Enterprise Cloud Docs"
