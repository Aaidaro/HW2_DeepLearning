import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.data_loader import create_dataloaders
from models.model import PersianMNISTCNN
from scripts.train import train_model
from scripts.evaluate import evaluate
from utils.visualization import plot_training_history, plot_first_conv_feature_maps


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/dataset/mnist.pkl.gz"
    )

    parser.add_argument(
        "--block",
        type=str,
        default="A",
        choices=["A", "B", "C"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
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
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adam", "sgd"]
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9
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
        "--output-dir",
        type=str,
        default="outputs"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/saved_models"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir

    plot_dir = output_dir / "plots" / f"block_{args.block}"
    log_dir = output_dir / "logs"

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Using block:", args.block)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=args.use_augmentation,
        pin_memory=torch.cuda.is_available()
    )

    model = PersianMNISTCNN(
        num_classes=10,
        block_type=args.block,
        dropout_rate=args.dropout,
        groups=args.groups,
        bottleneck_ratio=args.bottleneck_ratio
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
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
        block_type=args.block
    )

    plot_training_history(
        history=history,
        save_dir=str(plot_dir)
    )

    test_loss, test_acc = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    test_result_path = log_dir / f"test_result_block_{args.block}.txt"

    with open(test_result_path, "w") as f:
        f.write(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n")

    checkpoint_path = save_dir / f"best_model_block_{args.block}.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    plot_first_conv_feature_maps(
    model=model,
    data_loader=test_loader,
    device=device,
    save_dir=str(plot_dir),
    image_index=53,
    max_filters=32,
    block_type=args.block)

if __name__ == "__main__":
    main()