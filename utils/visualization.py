import os
import torch
import math
import matplotlib.pyplot as plt


def plot_loss_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(save_path)
    plt.close()


def plot_training_history(history, save_dir):
    plot_loss_curves(history, save_dir)
    plot_accuracy_curves(history, save_dir)


def plot_first_conv_feature_maps(
    model,
    data_loader,
    device,
    save_dir,
    image_index=0,
    max_filters=32,
    block_type="A"
):
    """
    Plot the output feature maps of the first convolutional layer
    for one image from the test set.
    """

    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    # Get one batch from test loader
    images, labels = next(iter(data_loader))

    image = images[image_index].unsqueeze(0).to(device)
    label = labels[image_index].item()

    activations = {}

    def hook_fn(module, input, output):
        activations["first_conv"] = output.detach().cpu()

    # First Conv2d layer in the model
    first_conv_layer = model.features[0].block[0]

    hook = first_conv_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(image)

    hook.remove()

    feature_maps = activations["first_conv"][0]

    num_filters = min(max_filters, feature_maps.shape[0])
    cols = 8
    rows = math.ceil(num_filters / cols)

    # Plot original input image
    plt.figure(figsize=(4, 4))
    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f"Input Image - Label: {label}")
    plt.axis("off")
    plt.tight_layout()

    input_save_path = os.path.join(
        save_dir,
        f"input_image_block_{block_type}.png"
    )
    plt.savefig(input_save_path)
    plt.close()

    # Plot feature maps
    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(num_filters):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_maps[i], cmap="gray")
        plt.title(f"Filter {i + 1}")
        plt.axis("off")

    plt.suptitle(
        f"First Convolution Layer Feature Maps - Block {block_type}",
        fontsize=14
    )

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"first_conv_feature_maps_block_{block_type}.png"
    )

    plt.savefig(save_path)
    plt.close()

    print(f"Saved first conv feature maps to: {save_path}")


def compute_confusion_matrix(model, data_loader, device, num_classes=10):
    model.eval()

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)

            for true_label, predicted_label in zip(labels, predictions):
                confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix.cpu().numpy()


def plot_confusion_matrix(
    confusion_matrix,
    save_dir,
    class_names=None,
    filename="confusion_matrix.png"
):
    os.makedirs(save_dir, exist_ok=True)

    if class_names is None:
        class_names = [str(i) for i in range(confusion_matrix.shape[0])]

    plt.figure(figsize=(9, 7))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix on Test Set")
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names)
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

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")