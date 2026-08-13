import copy
import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
    random_split,
)
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification


# ------------------
# Setup
# ------------------

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

dataset_dir = "all_image"

full = datasets.ImageFolder(
    dataset_dir
)

class_names = full.classes
num_classes = len(class_names)

print(
    "Classes:",
    class_names
)


# ------------------
# Dataset
# ------------------

samples = full.samples

targets = [
    label
    for _, label in samples
]

train_items, test_items, _, _ = train_test_split(
    samples,
    targets,
    test_size=0.25,
    random_state=SEED,
    stratify=targets
)


class ImageLoader(Dataset):

    def __init__(
        self,
        items,
        transform=None
    ):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(
        self,
        idx
    ):
        path, label = self.items[idx]

        img = Image.open(
            path
        ).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ------------------
# Transforms
# ------------------

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]

train_tf = transforms.Compose([

    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        0.15,
        0.15,
        0.15,
        0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),

])


eval_tf = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),

])


train_ds = ImageLoader(
    train_items,
    transform=train_tf
)

test_full = ImageLoader(
    test_items,
    transform=eval_tf
)


# ------------------
# Validation / Test Split
# ------------------

len_full = len(test_full)

len_test = int(
    0.2 * len_full
)

len_val = (
    len_full - len_test
)

val_ds, test_ds = random_split(

    test_full,

    [
        len_val,
        len_test
    ],

    generator=torch.Generator().manual_seed(
        SEED
    )

)


# ------------------
# Weighted Sampler
# ------------------

train_labels = [
    lbl
    for _, lbl in train_items
]

counts = Counter(
    train_labels
)

print(
    "Train class counts:",
    counts
)


class_weights = {
    c: 1.0 / counts[c]
    for c in counts
}


sample_weights = [
    class_weights[lbl]
    for lbl in train_labels
]


sampler = WeightedRandomSampler(

    sample_weights,

    num_samples=len(
        sample_weights
    ),

    replacement=True

)


# ------------------
# DataLoaders
# ------------------

BATCH = 24

train_loader = DataLoader(

    train_ds,

    batch_size=BATCH,

    sampler=sampler,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    )

)


val_loader = DataLoader(

    val_ds,

    batch_size=BATCH,

    shuffle=False,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    )

)


test_loader = DataLoader(

    test_ds,

    batch_size=BATCH,

    shuffle=False,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    )

)


dataloaders = {

    "train": train_loader,

    "val": val_loader,

    "test": test_loader

}


# ------------------
# Model
# ------------------

model_name = (
    "microsoft/"
    "swin-tiny-patch4-window7-224"
)


id2label = {
    i: n
    for i, n in enumerate(class_names)
}


label2id = {
    n: i
    for i, n in enumerate(class_names)
}


model = AutoModelForImageClassification.from_pretrained(

    model_name,

    num_labels=num_classes,

    id2label=id2label,

    label2id=label2id,

    ignore_mismatched_sizes=True

)


# ------------------
# Classification Head
# ------------------

if (
    hasattr(model, "classifier")
    and isinstance(
        model.classifier,
        nn.Linear
    )
):

    n_in = (
        model.classifier.in_features
    )

    model.classifier = nn.Sequential(

        nn.Linear(
            n_in,
            512
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            0.35
        ),

        nn.Linear(
            512,
            256
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            0.35
        ),

        nn.Linear(
            256,
            num_classes
        ),

    )


elif (
    hasattr(model, "head")
    and isinstance(
        getattr(model, "head"),
        nn.Linear
    )
):

    n_in = (
        model.head.in_features
    )

    model.head = nn.Sequential(

        nn.Linear(
            n_in,
            512
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            0.35
        ),

        nn.Linear(
            512,
            256
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Dropout(
            0.35
        ),

        nn.Linear(
            256,
            num_classes
        ),

    )

else:

    raise RuntimeError(
        "Cannot find classifier/head to replace."
    )


model = model.to(device)


# ------------------
# Fine-Tuning
# ------------------

LAST2_A = "swin.encoder.layers.2"
LAST2_B = "swin.encoder.layers.3"


def set_trainable(stage):

    for parameter in model.parameters():
        parameter.requires_grad = False


    for name, parameter in model.named_parameters():

        if (
            "classifier" in name
            or "head" in name
        ):
            parameter.requires_grad = True


    if stage >= 2:

        for name, parameter in model.named_parameters():

            if (
                name.startswith(LAST2_A)
                or name.startswith(LAST2_B)
            ):
                parameter.requires_grad = True


    trainable = [

        name

        for name, parameter
        in model.named_parameters()

        if parameter.requires_grad

    ]


    print(
        f"Stage {stage} trainable params:",
        len(trainable)
    )

    print(
        "Sample trainable:",
        trainable[:20]
    )


# ------------------
# Training / Evaluation
# ------------------

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
).to(device)


scaler = GradScaler(
    "cuda"
    if device.type == "cuda"
    else "cpu"
)


def eval_val():

    model.eval()

    all_y = []
    all_p = []

    loss_sum = 0.0
    n = 0

    with torch.no_grad():

        for x, y in dataloaders["val"]:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            with autocast(
                "cuda"
                if device.type == "cuda"
                else "cpu"
            ):

                logits = model(x).logits

                loss = criterion(
                    logits,
                    y
                )

            p = torch.argmax(
                logits,
                1
            )

            all_y.extend(
                y.cpu().tolist()
            )

            all_p.extend(
                p.cpu().tolist()
            )

            loss_sum += (
                loss.item()
                * x.size(0)
            )

            n += x.size(0)


    val_loss = (
        loss_sum / n
    )


    val_acc = (
        np.array(all_y)
        == np.array(all_p)
    ).mean()


    val_bal = balanced_accuracy_score(
        all_y,
        all_p
    )


    return (
        val_loss,
        val_acc,
        val_bal
    )


# ------------------
# Training
# ------------------

def train_stage(
    stage,
    epochs,
    max_lr
):

    set_trainable(
        stage
    )


    if stage == 2:

        l3 = [

            name

            for name, parameter
            in model.named_parameters()

            if (
                parameter.requires_grad
                and name.startswith(
                    "swin.encoder.layers.3"
                )
            )

        ]

        print(
            "Trainable in layers.3:",
            len(l3)
        )

        if len(l3) > 0:

            print(
                "layers.3 sample:",
                l3[:10]
            )


    if stage == 1:

        optimizer = optim.AdamW(

            filter(
                lambda p: p.requires_grad,
                model.parameters()
            ),

            lr=max_lr / 10,

            weight_decay=0.05

        )


        scheduler = optim.lr_scheduler.OneCycleLR(

            optimizer,

            max_lr=max_lr,

            epochs=epochs,

            steps_per_epoch=len(
                dataloaders["train"]
            ),

            pct_start=0.2,

            anneal_strategy="cos",

            div_factor=10,

            final_div_factor=1e4

        )


    else:

        optimizer = optim.AdamW(

            filter(
                lambda p: p.requires_grad,
                model.parameters()
            ),

            lr=max_lr,

            weight_decay=0.05

        )

        scheduler = None


    history = {

        "train_loss": [],

        "train_acc": [],

        "val_loss": [],

        "val_acc": [],

        "val_bal_acc": [],

        "lr": []

    }


    best_wts = copy.deepcopy(
        model.state_dict()
    )

    best_bal = -1.0

    patience = 3

    no_improve = 0


    for ep in range(epochs):

        model.train()

        tr_loss_sum = 0.0
        tr_correct = 0
        tr_n = 0


        for x, y in tqdm.tqdm(

            dataloaders["train"],

            desc=(
                f"stage{stage}-train"
            ),

            leave=False

        ):

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            with autocast(

                "cuda"
                if device.type == "cuda"
                else "cpu"

            ):

                logits = model(x).logits

                loss = criterion(
                    logits,
                    y
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


            if scheduler is not None:
                scheduler.step()


            tr_loss_sum += (
                loss.item()
                * x.size(0)
            )


            tr_correct += (

                torch.argmax(
                    logits,
                    1
                ) == y

            ).sum().item()


            tr_n += x.size(0)


        tr_loss = (
            tr_loss_sum / tr_n
        )


        tr_acc = (
            tr_correct / tr_n
        )


        (
            val_loss,
            val_acc,
            val_bal
        ) = eval_val()


        lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        history["train_loss"].append(
            tr_loss
        )

        history["train_acc"].append(
            tr_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        history["val_bal_acc"].append(
            val_bal
        )

        history["lr"].append(
            lr
        )


        print(

            f"[Stage {stage}] "
            f"Ep {ep + 1}/{epochs} | "
            f"train loss {tr_loss:.4f} "
            f"acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} "
            f"acc {val_acc:.4f} "
            f"bal_acc {val_bal:.4f} | "
            f"lr {lr:.2e}"

        )


        if val_bal > best_bal:

            best_bal = val_bal

            best_wts = copy.deepcopy(
                model.state_dict()
            )

            no_improve = 0

            print(
                f"Best balanced_acc: "
                f"{best_bal:.4f}"
            )


        else:

            no_improve += 1

            if no_improve >= patience:

                print(
                    f"Early stopping "
                    f"(no improve for "
                    f"{patience} epochs)."
                )

                break


    model.load_state_dict(
        best_wts
    )

    return (
        best_bal,
        history
    )


# ------------------
# Run Training
# ------------------

best1, hist1 = train_stage(
    stage=1,
    epochs=4,
    max_lr=3e-4
)


best2, hist2 = train_stage(
    stage=2,
    epochs=15,
    max_lr=1e-4
)


# ------------------
# Plot Training History
# ------------------

def plot_history(
    history,
    title="Training"
):

    epochs = range(
        1,
        len(
            history["train_acc"]
        ) + 1
    )


    plt.figure()

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Acc"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Val Acc"
    )

    plt.plot(
        epochs,
        history["val_bal_acc"],
        label="Val Balanced Acc"
    )

    plt.title(
        f"{title} - Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


    plt.figure()

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Val Loss"
    )

    plt.title(
        f"{title} - Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


plot_history(
    hist1,
    "Stage 1"
)

plot_history(
    hist2,
    "Stage 2"
)


# ------------------
# Test Evaluation
# ------------------

def test_eval():

    model.eval()

    loss_sum = 0.0
    correct = 0
    n = 0

    with torch.no_grad():

        for x, y in dataloaders["test"]:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            with autocast(

                "cuda"
                if device.type == "cuda"
                else "cpu"

            ):

                logits = model(x).logits

                loss = criterion(
                    logits,
                    y
                )


            loss_sum += (
                loss.item()
                * x.size(0)
            )


            correct += (

                torch.argmax(
                    logits,
                    1
                ) == y

            ).sum().item()


            n += x.size(0)


    return (
        loss_sum / n,
        correct / n
    )


te_loss, te_acc = test_eval()


print(
    f"TEST loss={te_loss:.4f} "
    f"acc={te_acc:.4f}"
)


# ------------------
# Save Model
# ------------------

torch.save(
    model.state_dict(),
    "swin_tiny_best_state_dict.pth"
)

print(
    "Saved swin_tiny_best_state_dict.pth")

