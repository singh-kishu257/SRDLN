"""SRDLN Clinical Data Engine for Diabetic Retinopathy."""
#Data Pipeline

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLINICAL_DISTRIBUTION: Dict[int, float] = {
    0: 49.29,
    1: 10.15,
    2: 27.28,
    3: 5.27,
    4: 8.06,
}


def discover_samples_from_csv(csv_path: str, image_dir: str) -> List[Tuple[Path, int]]:
    """Read (image_path, label) pairs from a Kaggle APTOS-style CSV.

    Supports column names:
      image : id_code | image | image_id | filename | file
      label : diagnosis | label | stage | class
    """
    csv_file   = Path(csv_path)
    image_root = Path(image_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    image_key_candidates = ["id_code", "image", "image_id", "filename", "file"]
    label_key_candidates = ["diagnosis", "label", "stage", "class"]
    ext_candidates       = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    samples: List[Tuple[Path, int]] = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header row: {csv_file}")

        lowered   = {name.strip().lower(): name for name in reader.fieldnames}
        image_col = next((lowered[k] for k in image_key_candidates if k in lowered), None)
        label_col = next((lowered[k] for k in label_key_candidates if k in lowered), None)

        if image_col is None or label_col is None:
            raise RuntimeError(
                f"Could not find image/label columns in CSV. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            img_id = str(row[image_col]).strip()
            label  = int(row[label_col])

            if label < 0 or label > 4:
                continue

            # Try exact path first, then try appending extensions
            candidate = image_root / img_id
            if candidate.exists():
                samples.append((candidate, label))
                continue

            found = False
            for ext in ext_candidates:
                probe = image_root / f"{img_id}{ext}"
                if probe.exists():
                    samples.append((probe, label))
                    found = True
                    break

            if not found:
                print(f"[WARN] Image not found for id: {img_id} — skipping")

    if not samples:
        raise RuntimeError(f"No valid samples found from CSV: {csv_file}")

    return samples


class RetinopathyImageDataset(Dataset):
    """PyTorch Dataset for retinal fundus images."""

    def __init__(self, samples: List[Tuple[Path, int]], transform=None) -> None:
        if not samples:
            raise RuntimeError("No samples provided to RetinopathyImageDataset.")
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(min(512, image_size * 2)),
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(min(512, image_size * 2)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _clinical_sampler_weights(labels: Sequence[int]) -> List[float]:
    prevalence    = {k: max(v / 100.0, 1e-8) for k, v in CLINICAL_DISTRIBUTION.items()}
    class_weights = {s: 1.0 / prevalence[s] for s in prevalence}
    mean_weight   = sum(class_weights.values()) / len(class_weights)
    class_weights = {k: v / mean_weight for k, v in class_weights.items()}
    return [class_weights[label] for label in labels]


def make_dataloaders(
    train_csv: str,
    train_image_dir: str,
    val_csv: str,
    val_image_dir: str,
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    train_samples = discover_samples_from_csv(train_csv, train_image_dir)
    val_samples   = discover_samples_from_csv(val_csv,   val_image_dir)

    train_dataset = RetinopathyImageDataset(train_samples, transform=_build_train_transform(image_size))
    val_dataset   = RetinopathyImageDataset(val_samples,   transform=_build_eval_transform(image_size))

    train_labels   = [label for _, label in train_samples]
    sample_weights = _clinical_sampler_weights(train_labels)
    sampler        = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,    num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def describe_dataset(samples: List[Tuple[Path, int]]) -> str:
    counts = Counter(label for _, label in samples)
    total  = sum(counts.values())
    return " | ".join(
        [f"stage {i}: {counts.get(i, 0)} ({100.0 * counts.get(i, 0) / max(total, 1):.2f}%)" for i in range(5)]
    )
