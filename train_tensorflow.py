import os
import json
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from tfswin import SwinTransformerTiny224
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)


# =========================
# GPU Setup
# =========================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu,
            True
        )

    print("GPU memory growth enabled")


# =========================
# Mixed Precision
# =========================

mixed_precision.set_global_policy(
    "mixed_float16"
)

print(
    "Policy:",
    mixed_precision.global_policy()
)


# =========================
# Configuration
# =========================

DATA_DIR = "all_image"

IMG_SIZE = (224, 224)

BATCH = 16

SEED = 42

EPOCHS_HEAD = 5

EPOCHS_FT = 20

CLASS_NAMES = [
    "ADI",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NOR",
    "STR",
    "TUM",
]

NUM_CLASSES = len(
    CLASS_NAMES
)

print(
    "NUM_CLASSES:",
    NUM_CLASSES
)

print(
    "CLASS_NAMES:",
    CLASS_NAMES
)


# =========================
# Dataset Validation
# =========================

assert os.path.isdir(
    DATA_DIR
), f"Folder not found: {DATA_DIR}"


folders = sorted(
    [
        directory
        for directory in os.listdir(DATA_DIR)
        if os.path.isdir(
            os.path.join(
                DATA_DIR,
                directory
            )
        )
    ]
)


print(
    "Found class folders:",
    folders
)


missing = (
    set(CLASS_NAMES)
    - set(folders)
)

extra = (
    set(folders)
    - set(CLASS_NAMES)
)


print(
    "Missing:",
    missing
)

print(
    "Extra:",
    extra
)


assert len(missing) == 0, (
    f"Missing class folders: {missing}"
)

assert len(extra) == 0, (
    f"Unexpected extra folders: {extra}"
)


# =========================
# Collect Image Paths
# =========================

all_paths = []

all_labels = []


for i, class_name in enumerate(
    CLASS_NAMES
):

    pattern = os.path.join(
        DATA_DIR,
        class_name,
        "*"
    )

    files = glob.glob(
        pattern
    )

    all_paths.extend(
        files
    )

    all_labels.extend(
        [i] * len(files)
    )


all_paths = np.array(
    all_paths
)

all_labels = np.array(
    all_labels
)


print(
    "Total files:",
    len(all_paths)
)


full_counts = np.bincount(
    all_labels,
    minlength=NUM_CLASSES
)


print(
    "Full dataset counts:",
    dict(
        zip(
            CLASS_NAMES,
            full_counts
        )
    )
)


# =========================
# Train / Validation / Test Split
# =========================

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=0.30,
    random_state=SEED,
    stratify=all_labels
)


val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)


print(
    "Train counts:",
    dict(
        zip(
            CLASS_NAMES,
            np.bincount(
                train_labels,
                minlength=NUM_CLASSES
            )
        )
    )
)


print(
    "Val counts:",
    dict(
        zip(
            CLASS_NAMES,
            np.bincount(
                val_labels,
                minlength=NUM_CLASSES
            )
        )
    )
)


print(
    "Test counts:",
    dict(
        zip(
            CLASS_NAMES,
            np.bincount(
                test_labels,
                minlength=NUM_CLASSES
            )
        )
    )
)


# =========================
# Image Loading
# =========================

def load_image(path, label):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.clip_by_value(
        image,
        0.0,
        255.0
    )

    image = tf.cast(
        tf.round(image),
        tf.uint8
    )

    return image, label


# =========================
# TensorFlow Datasets
# =========================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)


val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)


test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_paths,
        test_labels
    )
)


train_ds = train_ds.shuffle(
    len(train_paths),
    seed=SEED
)


train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_ds = train_ds.batch(
    BATCH
).prefetch(
    tf.data.AUTOTUNE
)


val_ds = val_ds.batch(
    BATCH
).prefetch(
    tf.data.AUTOTUNE
)


test_ds = test_ds.batch(
    BATCH
).prefetch(
    tf.data.AUTOTUNE
)


# =========================
# Class Weights
# =========================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)


class_weights = {
    i: float(weight)
    for i, weight
    in enumerate(class_weights_array)
}


print(
    "Class weights:",
    class_weights
)


# =========================
# Data Augmentation
# =========================

data_aug = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal_and_vertical"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),
    ],
    name="augmentation"
)


def preprocess_train(x, y):

    x = tf.cast(
        x,
        tf.float32
    )

    x = data_aug(
        x,
        training=True
    )

    x = tf.clip_by_value(
        x,
        0.0,
        255.0
    )

    x = tf.round(
        x
    )

    x = tf.cast(
        x,
        tf.uint8
    )

    return x, y


def preprocess_eval(x, y):

    x = tf.cast(
        x,
        tf.float32
    )

    x = tf.clip_by_value(
        x,
        0.0,
        255.0
    )

    x = tf.round(
        x
    )

    x = tf.cast(
        x,
        tf.uint8
    )

    return x, y


# =========================
# Preprocessed Datasets
# =========================

AUTOTUNE = tf.data.AUTOTUNE


train_pp = train_ds.map(
    preprocess_train,
    num_parallel_calls=AUTOTUNE
).prefetch(
    AUTOTUNE
)


val_pp = val_ds.map(
    preprocess_eval,
    num_parallel_calls=AUTOTUNE
).prefetch(
    AUTOTUNE
)


test_pp = test_ds.map(
    preprocess_eval,
    num_parallel_calls=AUTOTUNE
).prefetch(
    AUTOTUNE
)


# =========================
# Sanity Check
# =========================

xb, yb = next(
    iter(train_pp)
)


print(
    "Sanity batch:",
    xb.dtype,
    xb.shape,
    "labels:",
    yb.shape,
    "min/max:",
    xb.numpy().min(),
    xb.numpy().max()
)


# =========================
# Model
# =========================

inputs = keras.Input(
    shape=(*IMG_SIZE, 3),
    dtype="uint8"
)


backbone = SwinTransformerTiny224(
    include_top=False
)


features = backbone(
    inputs,
    training=False
)


x = layers.GlobalAveragePooling2D()(
    features
)


x = layers.Dense(
    512,
    activation="gelu"
)(x)


x = layers.Dropout(
    0.4
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    dtype="float32"
)(x)


model = keras.Model(
    inputs,
    outputs
)


model.summary()


# =========================
# Callbacks
# =========================

ckpt_path = (
    "best_swin.weights.h5"
)


callbacks = [

    keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    ),

]


# =========================
# Stage 1
# Train Classification Head
# =========================

backbone.trainable = False


print(
    "Backbone trainable:",
    backbone.trainable
)


print(
    "Trainable weights:",
    len(
        model.trainable_weights
    )
)


model.compile(

    optimizer=keras.optimizers.AdamW(
        learning_rate=3e-4,
        weight_decay=5e-2
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="acc"
        )
    ]

)


history_head = model.fit(

    train_pp,

    validation_data=val_pp,

    epochs=EPOCHS_HEAD,

    class_weight=class_weights,

    callbacks=callbacks,

)


# =========================
# Stage 2
# Fine-Tuning
# =========================

backbone.trainable = True


for layer in backbone.layers[:-20]:

    layer.trainable = False


print(
    "Backbone trainable:",
    backbone.trainable
)


print(
    "Trainable weights:",
    len(
        model.trainable_weights
    )
)


model.compile(

    optimizer=keras.optimizers.AdamW(
        learning_rate=2e-5,
        weight_decay=5e-2
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="acc"
        )
    ]

)


history_ft = model.fit(

    train_pp,

    validation_data=val_pp,

    epochs=EPOCHS_FT,

    class_weight=class_weights,

    callbacks=callbacks,

)


print(
    "Saved best model:",
    ckpt_path
)


# =========================
# Save Training History
# =========================

hist = {}


keys = set(
    list(
        history_head.history.keys()
    )
    +
    list(
        history_ft.history.keys()
    )
)


for key in keys:

    hist[key] = (
        history_head.history.get(
            key,
            []
        )
        +
        history_ft.history.get(
            key,
            []
        )
    )


with open(
    "history_all.json",
    "w"
) as file:

    json.dump(
        hist,
        file,
        indent=2
    )


print(
    "Saved: history_all.json"
)


# =========================
# Find Best Epoch
# =========================

val_loss = np.array(
    hist.get(
        "val_loss",
        []
    ),
    dtype=np.float64
)


if val_loss.size == 0:

    raise RuntimeError(
        "val_loss not found in history."
    )


best_idx = int(
    np.argmin(
        val_loss
    )
)


best_epoch = (
    best_idx + 1
)


print(
    f"Best epoch: {best_epoch} | "
    f"val_loss={val_loss[best_idx]:.6f}"
)


# =========================
# Accuracy Curve
# =========================

epochs = np.arange(
    1,
    len(
        hist["loss"]
    ) + 1
)


acc_key = (
    "acc"
    if "acc" in hist
    else "sparse_categorical_accuracy"
)


val_acc_key = (
    "val_acc"
    if "val_acc" in hist
    else "val_sparse_categorical_accuracy"
)


plt.figure()


plt.plot(
    epochs,
    hist.get(
        acc_key,
        []
    ),
    label="train_acc"
)


plt.plot(
    epochs,
    hist.get(
        val_acc_key,
        []
    ),
    label="val_acc"
)


plt.axvline(
    best_epoch,
    linestyle="--",
    label="best_epoch"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Accuracy vs Epoch"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    "accuracy_curve.png",
    dpi=200
)


plt.show()


print(
    "Saved: accuracy_curve.png"
)


# =========================
# Loss Curve
# =========================

plt.figure()


plt.plot(
    epochs,
    hist["loss"],
    label="train_loss"
)


plt.plot(
    epochs,
    hist["val_loss"],
    label="val_loss"
)


plt.axvline(
    best_epoch,
    linestyle="--",
    label="best_epoch"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Loss vs Epoch"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    "loss_curve.png",
    dpi=200
)


plt.show()


print(
    "Saved: loss_curve.png"
)


# =========================
# Load Best Model
# =========================

model.load_weights(
    "best_swin.weights.h5"
)


# =========================
# Test-Time Augmentation
# =========================

def tta_predict_batch(
    model,
    x
):

    x0 = x

    x1 = tf.image.flip_left_right(
        x
    )

    x2 = tf.image.flip_up_down(
        x
    )


    predictions = []


    for xi in [
        x0,
        x1,
        x2
    ]:

        prediction = model.predict(
            xi,
            verbose=0
        )

        predictions.append(
            tf.convert_to_tensor(
                prediction,
                dtype=tf.float32
            )
        )


    return tf.reduce_mean(
        tf.stack(
            predictions,
            axis=0
        ),
        axis=0
    )


# =========================
# Test Evaluation
# =========================

print(
    "\n========================="
)

print(
    "TEST EVALUATION"
)

print(
    "========================="
)


y_true_test = []

y_pred_test = []


for x, y in test_pp:

    x = tf.cast(
        x,
        tf.float32
    )


    predictions = tta_predict_batch(
        model,
        x
    )


    y_true_test.extend(
        y.numpy().tolist()
    )


    y_pred_test.extend(
        tf.argmax(
            predictions,
            axis=1
        ).numpy().tolist()
    )


y_true_test = np.array(
    y_true_test
)


y_pred_test = np.array(
    y_pred_test
)


print(
    "Unique y_true_test labels:",
    np.unique(
        y_true_test,
        return_counts=True
    )
)


labels = list(
    range(NUM_CLASSES)
)


print(
    "\nTEST Classification Report:"
)


print(
    classification_report(
        y_true_test,
        y_pred_test,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# =========================
# Test Metrics
# =========================

macro_f1_test = f1_score(
    y_true_test,
    y_pred_test,
    average="macro"
)


weighted_f1_test = f1_score(
    y_true_test,
    y_pred_test,
    average="weighted"
)


balanced_accuracy_test = (
    balanced_accuracy_score(
        y_true_test,
        y_pred_test
    )
)


accuracy_test = (
    y_true_test
    == y_pred_test
).mean()


print(
    "\nTEST Summary Metrics:"
)


print(
    f"Accuracy          : "
    f"{accuracy_test:.4f}"
)


print(
    f"Balanced Accuracy : "
    f"{balanced_accuracy_test:.4f}"
)


print(
    f"Macro F1          : "
    f"{macro_f1_test:.4f}"
)


print(
    f"Weighted F1       : "
    f"{weighted_f1_test:.4f}"
)


# =========================
# Test Confusion Matrix
# =========================

cm_test = confusion_matrix(
    y_true_test,
    y_pred_test,
    labels=labels
)


print(
    "\nTEST Confusion Matrix:"
)


print(
    cm_test
)


# =========================
# Confusion Matrix - Counts
# =========================

plt.figure(
    figsize=(10, 8)
)


plt.imshow(
    cm_test,
    interpolation="nearest"
)


plt.title(
    "Confusion Matrix (TEST) - Counts"
)


plt.colorbar()


ticks = np.arange(
    len(CLASS_NAMES)
)


plt.xticks(
    ticks,
    CLASS_NAMES,
    rotation=45,
    ha="right"
)


plt.yticks(
    ticks,
    CLASS_NAMES
)


threshold = (
    cm_test.max() * 0.6
)


for i in range(
    cm_test.shape[0]
):

    for j in range(
        cm_test.shape[1]
    ):

        plt.text(
            j,
            i,
            str(
                cm_test[i, j]
            ),
            ha="center",
            va="center",
            color=(
                "white"
                if cm_test[i, j] > threshold
                else "black"
            ),
            fontsize=8
        )


plt.ylabel(
    "True label"
)


plt.xlabel(
    "Predicted label"
)


plt.tight_layout()


plt.savefig(
    "confusion_matrix_test_counts.png",
    dpi=200
)


plt.show()


print(
    "Saved: confusion_matrix_test_counts.png"
)


# =========================
# Confusion Matrix - Normalized
# =========================

cm_norm = (
    cm_test.astype(
        np.float64
    )
    /
    np.maximum(
        cm_test.sum(
            axis=1,
            keepdims=True
        ),
        1
    )
)


plt.figure(
    figsize=(10, 8)
)


plt.imshow(
    cm_norm,
    interpolation="nearest"
)


plt.title(
    "Confusion Matrix (TEST) - "
    "Normalized (Recall %)"
)


plt.colorbar()


plt.xticks(
    ticks,
    CLASS_NAMES,
    rotation=45,
    ha="right"
)


plt.yticks(
    ticks,
    CLASS_NAMES
)


for i in range(
    cm_norm.shape[0]
):

    for j in range(
        cm_norm.shape[1]
    ):

        plt.text(
            j,
            i,
            f"{cm_norm[i, j] * 100:.1f}%",
            ha="center",
            va="center",
            color=(
                "white"
                if cm_norm[i, j] > 0.6
                else "black"
            ),
            fontsize=8
        )


plt.ylabel(
    "True label"
)


plt.xlabel(
    "Predicted label"
)


plt.tight_layout()


plt.savefig(
    "confusion_matrix_test_normalized.png",
    dpi=200
)


plt.show()


print(
    "Saved: confusion_matrix_test_normalized.png"
)


# =========================
# Confusion Matrix - General
# =========================

plt.figure(
    figsize=(10, 8)
)


plt.imshow(
    cm_test,
    interpolation="nearest"
)


plt.title(
    "Confusion Matrix (TEST)"
)


plt.colorbar()


plt.xticks(
    ticks,
    CLASS_NAMES,
    rotation=45,
    ha="right"
)


plt.yticks(
    ticks,
    CLASS_NAMES
)


threshold = (
    cm_test.max() * 0.6
)


for i in range(
    cm_test.shape[0]
):

    for j in range(
        cm_test.shape[1]
    ):

        plt.text(
            j,
            i,
            str(
                cm_test[i, j]
            ),
            ha="center",
            va="center",
            color=(
                "white"
                if cm_test[i, j] > threshold
                else "black"
            ),
            fontsize=8
        )


plt.ylabel(
    "True label"
)


plt.xlabel(
    "Predicted label"
)


plt.tight_layout()


plt.savefig(
    "confusion_matrix_test.png",
    dpi=200
)


plt.show()


print(
    "Saved: confusion_matrix_test.png"
)

