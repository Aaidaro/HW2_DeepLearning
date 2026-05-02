import os
import torch

from utils.metrics import calculate_accuracy
from scripts.evaluate import evaluate


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
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


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_dir,
    log_dir,
    block_type
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    log_path = os.path.join(log_dir, f"training_log_block_{block_type}.txt")

    with open(log_path, "w") as log_file:
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )

            val_loss, val_acc = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            log_message = (
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            print(log_message)
            log_file.write(log_message + "\n")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

                checkpoint_path = os.path.join(
                    save_dir,
                    f"best_model_block_{block_type}.pth"
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "block_type": block_type
                    },
                    checkpoint_path
                )

    return history