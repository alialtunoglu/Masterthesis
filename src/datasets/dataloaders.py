"""DataLoader creation from split JSON files."""

from __future__ import annotations

from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from .torch_datasets import SplitJsonImageDataset
except ImportError:
    from torch_datasets import SplitJsonImageDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_train_transform(image_size: int, use_augmentation: bool = True):
    transform_steps = []
    if use_augmentation:
        transform_steps.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
            ]
        )
    else:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(transform_steps)


def build_eval_transform(image_size: int):
    resize_size = int(round(image_size * 1.14))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_dataloaders(
    split_file: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    use_augmentation: bool = True,
    recovery_seed: int | None = None,
    project_root: Path | None = None,
):
    """Create train, validation, and test DataLoaders from a split file."""
    train_dataset = SplitJsonImageDataset(
        split_file=split_file,
        split="train",
        transform=build_train_transform(image_size, use_augmentation=use_augmentation),
        **({"project_root": project_root} if project_root is not None else {}),
    )
    val_dataset = SplitJsonImageDataset(
        split_file=split_file,
        split="val",
        transform=build_eval_transform(image_size),
        **({"project_root": project_root} if project_root is not None else {}),
    )
    test_dataset = SplitJsonImageDataset(
        split_file=split_file,
        split="test",
        transform=build_eval_transform(image_size),
        **({"project_root": project_root} if project_root is not None else {}),
    )

    pin_memory = torch.cuda.is_available()
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0 and recovery_seed is None,
    }

    if recovery_seed is not None:
        common_kwargs["worker_init_fn"] = seed_worker
    train_loader = DataLoader(train_dataset, shuffle=True,
                              generator=torch.Generator().manual_seed(recovery_seed) if recovery_seed is not None else None,
                              **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False,
                            generator=torch.Generator().manual_seed(recovery_seed + 1) if recovery_seed is not None else None,
                            **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             generator=torch.Generator().manual_seed(recovery_seed + 2) if recovery_seed is not None else None,
                             **common_kwargs)

    return train_loader, val_loader, test_loader, train_dataset.classes
