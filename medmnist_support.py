"""Dataset and metric helpers for the main-text MedMNIST Mixer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


MEDMNIST_DATASETS = (
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
)


def ensure_medmnist() -> None:
    try:
        import medmnist  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MedMNIST experiments require the optional 'medmnist' dependency. "
            "Install the repository requirements before running this dataset."
        ) from exc


def is_medmnist_dataset(name: str) -> bool:
    return name.lower() in MEDMNIST_DATASETS


def load_medmnist_class_and_info(flag: str):
    ensure_medmnist()
    import medmnist
    from medmnist import INFO

    dataset_flag = flag.lower()
    info = INFO[dataset_flag]
    dataset_class = getattr(medmnist, info["python_class"])
    return dataset_class, info


def medmnist_transforms(flag: str, num_channels: int):
    mean = (0.5,) * num_channels
    std = (0.5,) * num_channels
    train_ops: list[Any] = []

    dataset_flag = flag.lower()
    if dataset_flag == "pathmnist":
        train_ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    elif dataset_flag in {"dermamnist", "retinamnist"}:
        train_ops.append(transforms.RandomHorizontalFlip())

    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_ops = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_medmnist_splits(flag: str, data_root: Path):
    """Build the official MedMNIST train, validation, and test splits."""

    dataset_class, info = load_medmnist_class_and_info(flag)
    import medmnist

    num_channels = int(info["n_channels"])
    train_transform, eval_transform = medmnist_transforms(flag, num_channels)
    train_set = dataset_class(
        root=str(data_root), split="train", download=True, transform=train_transform
    )
    validation_set = dataset_class(
        root=str(data_root), split="val", download=True, transform=eval_transform
    )
    test_set = dataset_class(
        root=str(data_root), split="test", download=True, transform=eval_transform
    )
    meta = {
        "num_classes": len(info["label"]),
        "in_channels": num_channels,
        "image_size": 28,
        "patch_default": 7,
        "task": info["task"],
        "metric_primary_name": "acc",
        "metric_secondary_name": "auc",
        "label_map": info["label"],
        "is_medmnist": True,
        "validation_is_official": True,
        "dataset_source": f"MedMNIST:{flag.lower()}",
        "dataset_version": medmnist.__version__,
        "training_augmentation": (
            "random_horizontal_flip+random_vertical_flip"
            if flag.lower() == "pathmnist"
            else (
                "random_horizontal_flip"
                if flag.lower() in {"dermamnist", "retinamnist"}
                else "none"
            )
        ),
        "final_evaluation_split": "official_test",
    }
    return train_set, validation_set, test_set, meta


def is_multilabel_task(task: str) -> bool:
    return task == "multi-label, binary-class"


def task_loss(task: str) -> nn.Module:
    if is_multilabel_task(task):
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def prepare_targets(y, task: str, device: torch.device):
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    if is_multilabel_task(task):
        if y.ndim == 1:
            y = y.unsqueeze(1)
        return y.float().to(device, non_blocking=True)
    if y.ndim > 1:
        y = y.squeeze(-1)
    return y.long().to(device, non_blocking=True)


def logits_to_scores(logits: torch.Tensor, task: str) -> torch.Tensor:
    if is_multilabel_task(task):
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)


def append_medmnist_batches(
    y_true_batches: list[np.ndarray],
    y_score_batches: list[np.ndarray],
    logits: torch.Tensor,
    targets: torch.Tensor,
    task: str,
):
    y_true_batches.append(targets.detach().cpu().numpy())
    y_score_batches.append(logits_to_scores(logits.detach(), task).cpu().numpy())


def compute_medmnist_metrics(
    y_true_batches: list[np.ndarray],
    y_score_batches: list[np.ndarray],
    task: str,
) -> tuple[float, float]:
    ensure_medmnist()
    from medmnist.evaluator import getACC, getAUC

    y_true = np.concatenate(y_true_batches, axis=0)
    y_score = np.concatenate(y_score_batches, axis=0)
    return float(getACC(y_true, y_score, task)), float(getAUC(y_true, y_score, task))
