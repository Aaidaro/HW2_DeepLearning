import gzip
import math
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class PersianMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Original shape is [784]. Convert to [1, 28, 28].
        image = torch.tensor(image, dtype=torch.float32).view(1, 28, 28)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class RandomAffineTensor:
    """Small torchvision-free affine augmentation for [C, H, W] tensors.

    This keeps the project easier to run on Kaggle/Colab environments where
    torch and torchvision versions can occasionally be mismatched.
    """

    def __init__(self, degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, image):
        if image.ndim != 3:
            raise ValueError("Expected image tensor with shape [C, H, W].")

        angle = math.radians(random.uniform(-self.degrees, self.degrees))
        scale = random.uniform(self.scale[0], self.scale[1])
        tx = random.uniform(-self.translate[0], self.translate[0])
        ty = random.uniform(-self.translate[1], self.translate[1])

        cos_a = math.cos(angle) * scale
        sin_a = math.sin(angle) * scale

        theta = torch.tensor(
            [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]],
            dtype=image.dtype,
            device=image.device,
        ).unsqueeze(0)

        batched = image.unsqueeze(0)
        grid = F.affine_grid(theta, batched.size(), align_corners=False)
        augmented = F.grid_sample(
            batched,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return augmented.squeeze(0)


def load_persian_mnist(data_path):
    with gzip.open(data_path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    return train_set, valid_set, test_set


def get_transforms(use_augmentation=True):
    train_transform = RandomAffineTensor() if use_augmentation else None
    test_transform = None
    return train_transform, test_transform


def create_dataloaders(
    data_path,
    batch_size=64,
    num_workers=2,
    use_augmentation=True,
    pin_memory=True
):
    train_set, valid_set, test_set = load_persian_mnist(data_path)

    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    x_test, y_test = test_set

    train_transform, test_transform = get_transforms(use_augmentation)

    train_dataset = PersianMNISTDataset(x_train, y_train, transform=train_transform)
    valid_dataset = PersianMNISTDataset(x_valid, y_valid, transform=test_transform)
    test_dataset = PersianMNISTDataset(x_test, y_test, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader
