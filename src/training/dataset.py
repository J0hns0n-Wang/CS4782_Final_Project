"""CIFAR-10 dataloader.

The Cold Diffusion paper works in [0, 1] pixel space (since the inpainting
mask multiplies pixel values, and a [0,1] image stays in [0,1] under that
mask). We follow the same convention here so D(x0, t) doesn't push values
out of range.
"""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def cifar10_loaders(
    root: str | None = None,
    batch_size: int = 64,
    num_workers: int = 2,
    augment: bool = True,
):
    if root is None:
        root = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        root = os.path.abspath(root)

    train_tf_ops = []
    if augment:
        train_tf_ops += [transforms.RandomHorizontalFlip()]
    train_tf_ops += [transforms.ToTensor()]  # -> [0, 1]
    train_tf = transforms.Compose(train_tf_ops)
    test_tf = transforms.Compose([transforms.ToTensor()])

    train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
