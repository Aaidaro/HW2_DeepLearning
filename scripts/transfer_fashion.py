import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.model import PersianMNISTCNN
from utils.metrics import calculate_accuracy


FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def create_fashion_mnist_dataloaders(
    data_dir,
    batch_size=64,
    num_workers=2,
    use_augmentation=True,
    pin_memory=True
):
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


def freeze_feature_extractor(model):
    for param in model.features.parameters():
        param.requires_grad = False


def reset_classifier(model, dropout_rate=0.2, num_classes=10):
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, num_classes)
    )


def train_one_epoch_transfer(model, train_loader, criterion, optimizer, device):
    """
    Feature extractor is frozen.
    Only classifier is trained.
    """

    model.features.eval()
    model.classifier.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            features = model.features(images)

        outputs = model.classifier(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        correct, total = calculate_accuracy(outputs, labels)
        correct_predictions += correct
        total_samples += total

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc


def evaluate_transfer(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            correct, total = calculate_accuracy(outputs, labels)
            correct_predictions += correct
            total_samples += total

            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_loss_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transfer Learning Loss on FashionMNIST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "fashion_transfer_loss_curve.png")
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Transfer Learning Accuracy on FashionMNIST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "fashion_transfer_accuracy_curve.png")
    plt.savefig(save_path)
    plt.close()


def compute_confusion_matrix(labels, preds, num_classes=10):
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for true_label, predicted_label in zip(labels, preds):
        confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix.numpy()


def plot_confusion_matrix(confusion_matrix, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix on FashionMNIST Test Set")
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = confusion_matrix.max() / 2.0

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            value = confusion_matrix[i, j]
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black"
            )

    plt.tight_layout()

    save_path = os.path.join(save_dir, "fashion_transfer_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()


def train_transfer_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_dir,
    log_dir
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_test_acc = 0.0

    log_path = os.path.join(log_dir, "fashion_transfer_training_log_block_C.txt")

    with open(log_path, "w") as log_file:
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch_transfer(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )

            test_loss, test_acc, _, _ = evaluate_transfer(
                model=model,
                data_loader=test_loader,
                criterion=criterion,
                device=device
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            log_message = (
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

            print(log_message)
            log_file.write(log_message + "\n")

            if test_acc > best_test_acc:
                best_test_acc = test_acc

                checkpoint_path = os.path.join(
                    save_dir,
                    "best_fashion_transfer_model_block_C.pth"
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "block_type": "C",
                        "dataset": "FashionMNIST"
                    },
                    checkpoint_path
                )

    return history


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="models/saved_models/best_model_block_C.pth"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--use-augmentation",
        action="store_true"
    )

    parser.add_argument(
        "--groups",
        type=int,
        default=8
    )

    parser.add_argument(
        "--bottleneck-ratio",
        type=float,
        default=0.5
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("Transfer learning from Block C model")
    print("Target dataset: FashionMNIST")

    fashion_data_dir = PROJECT_ROOT / "data" / "fashion_mnist"

    output_dir = PROJECT_ROOT / "outputs"
    plot_dir = output_dir / "plots" / "fashion_transfer_block_C"
    log_dir = output_dir / "logs"
    save_dir = PROJECT_ROOT / "saved_models"

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    train_loader, test_loader = create_fashion_mnist_dataloaders(
        data_dir=str(fashion_data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=args.use_augmentation,
        pin_memory=torch.cuda.is_available()
    )

    model = PersianMNISTCNN(
        num_classes=10,
        block_type="C",
        dropout_rate=args.dropout,
        groups=args.groups,
        bottleneck_ratio=args.bottleneck_ratio
    ).to(device)

    pretrained_path = PROJECT_ROOT / args.pretrained_path

    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded pretrained Block C model from:", pretrained_path)

    freeze_feature_extractor(model)
    reset_classifier(model, dropout_rate=args.dropout, num_classes=10)
    model = model.to(device)

    print("Feature extractor frozen.")
    print("Classifier reset for FashionMNIST.")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    history = train_transfer_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        save_dir=str(save_dir),
        log_dir=str(log_dir)
    )

    plot_loss_curves(history, save_dir=str(plot_dir))
    plot_accuracy_curves(history, save_dir=str(plot_dir))

    best_checkpoint_path = save_dir / "best_fashion_transfer_model_block_C.pth"
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_acc, preds, labels = evaluate_transfer(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Final Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc:.4f}")

    result_path = log_dir / "fashion_transfer_test_result_block_C.txt"

    with open(result_path, "w") as f:
        f.write(f"Final Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc:.4f}\n")

    confusion_matrix = compute_confusion_matrix(
        labels=labels,
        preds=preds,
        num_classes=10
    )

    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=FASHION_CLASSES,
        save_dir=str(plot_dir)
    )

    print("Saved plots to:", plot_dir)


if __name__ == "__main__":
    main()