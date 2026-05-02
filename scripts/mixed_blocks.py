import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.data_loader import create_dataloaders
from models.model import MixedPersianMNISTCNN
from scripts.train import train_model
from scripts.evaluate import evaluate
from utils.visualization import (
    plot_training_history,
    compute_confusion_matrix,
    plot_confusion_matrix
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/dataset/mnist.pkl.gz"
    )

    parser.add_argument(
        "--block-sequence",
        type=str,
        default="CBC",
        help="Three-block sequence, for example CBC, CAC, BCC"
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--bottleneck-ratio", type=float, default=0.5)

    return parser.parse_args()


def main():
    args = parse_args()

    sequence = args.block_sequence.upper()

    if len(sequence) != 3:
        raise ValueError("block_sequence must have exactly 3 letters, e.g. CBC")

    for block in sequence:
        if block not in ["A", "B", "C"]:
            raise ValueError("Each block must be A, B, or C.")

    data_path = PROJECT_ROOT / args.data_path

    output_dir = PROJECT_ROOT / "outputs"
    plot_dir = output_dir / "plots" / f"mixed_{sequence}"
    log_dir = output_dir / "logs"
    save_dir = PROJECT_ROOT / "saved_models"

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("Using mixed block sequence:", sequence)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=args.use_augmentation,
        pin_memory=torch.cuda.is_available()
    )

    model = MixedPersianMNISTCNN(
        num_classes=10,
        block_sequence=sequence,
        dropout_rate=args.dropout,
        groups=args.groups,
        bottleneck_ratio=args.bottleneck_ratio
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        save_dir=str(save_dir),
        log_dir=str(log_dir),
        block_type=f"mixed_{sequence}"
    )

    plot_training_history(
        history=history,
        save_dir=str(plot_dir)
    )

    checkpoint_path = save_dir / f"best_model_block_mixed_{sequence}.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Best Mixed Model Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    test_result_path = log_dir / f"test_result_mixed_{sequence}.txt"

    with open(test_result_path, "w") as f:
        f.write(f"Block Sequence: {sequence}\n")
        f.write(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")

    confusion_matrix = compute_confusion_matrix(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=10
    )

    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        save_dir=str(plot_dir),
        class_names=[str(i) for i in range(10)],
        filename=f"confusion_matrix_mixed_{sequence}.png"
    )


if __name__ == "__main__":
    main()