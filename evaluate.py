"""SRDLN evaluation script.

This script validates post-training accuracy on the first N samples from `train_images`
using labels from `train.csv`, then generates:
1) overall accuracy + right/wrong counts,
2) confusion matrix image,
3) per-stage sensitivity/specificity chart image.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from dr_model import SRDLNDRNet
from retinopathy_data import IMAGENET_MEAN, IMAGENET_STD

STAGE_LABELS = [
    "Stage 0: No DR",
    "Stage 1: Mild NPDR",
    "Stage 2: Moderate NPDR",
    "Stage 3: Severe NPDR",
    "Stage 4: PDR",
]

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def _find_columns(fieldnames: Sequence[str]) -> Tuple[str, str]:
    lowered = {name.lower(): name for name in fieldnames}
    image_col = next((lowered[k] for k in ["image", "image_id", "id", "filename", "file"] if k in lowered), None)
    label_col = next((lowered[k] for k in ["label", "stage", "diagnosis", "class"] if k in lowered), None)
    if image_col is None or label_col is None:
        raise RuntimeError(
            f"CSV must contain image+label columns. Found: {fieldnames}. "
            "Accepted image columns: image/image_id/id/filename/file and label columns: label/stage/diagnosis/class."
        )
    return image_col, label_col


def load_first_n_samples(csv_path: Path, image_dir: Path, limit: int) -> List[Tuple[Path, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    samples: List[Tuple[Path, int]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no headers.")
        image_col, label_col = _find_columns(reader.fieldnames)

        for row in reader:
            image_name = str(row[image_col]).strip()
            label = int(row[label_col])
            if label < 0 or label > 4:
                continue

            candidate = image_dir / image_name
            image_path = candidate if candidate.exists() else None
            if image_path is None:
                for ext in VALID_EXTENSIONS:
                    probe = image_dir / f"{image_name}{ext}"
                    if probe.exists():
                        image_path = probe
                        break

            if image_path is None:
                continue

            samples.append((image_path, label))
            if len(samples) >= limit:
                break

    if not samples:
        raise RuntimeError("No valid samples found for evaluation.")

    return samples


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.CenterCrop(448),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model(weights_path: Path, device: torch.device) -> SRDLNDRNet:
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")
    model = SRDLNDRNet(num_classes=5, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate(model: SRDLNDRNet, samples: Sequence[Tuple[Path, int]], device: torch.device) -> Tuple[List[int], List[int]]:
    transform = build_transform()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for image_path, label in samples:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = int(torch.argmax(logits, dim=1).item())
            y_true.append(label)
            y_pred.append(pred)

    return y_true, y_pred


def sensitivity_specificity(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sens = np.zeros(5, dtype=np.float32)
    spec = np.zeros(5, dtype=np.float32)

    total = cm.sum()
    for i in range(5):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sens, spec


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("SRDLN Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(5)
    plt.xticks(ticks, [f"{i}" for i in range(5)])
    plt.yticks(ticks, [f"{i}" for i in range(5)])
    plt.xlabel("Predicted Stage")
    plt.ylabel("True Stage")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_sensitivity_specificity(sens: np.ndarray, spec: np.ndarray, output_path: Path) -> None:
    x = np.arange(5)
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, sens, width=width, label="Sensitivity", color="#2C7BB6")
    plt.bar(x + width / 2, spec, width=width, label="Specificity", color="#D7191C")

    plt.ylim(0, 1.05)
    plt.xticks(x, [f"Stage {i}" for i in range(5)])
    plt.ylabel("Score")
    plt.title("Sensitivity & Specificity by DR Stage")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SRDLN model on first N training images")
    parser.add_argument("--csv", type=str, default="train.csv", help="CSV with image ids and labels")
    parser.add_argument("--images", type=str, default="train_images", help="Directory containing training images")
    parser.add_argument("--weights", type=str, default="SRDLN_clinical_weights.pth", help="Trained model weights")
    parser.add_argument("--limit", type=int, default=1929, help="Number of first samples to evaluate")
    parser.add_argument("--cm-out", type=str, default="confusion_matrix.png", help="Output path for confusion matrix image")
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="sensitivity_specificity.png",
        help="Output path for sensitivity/specificity chart",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    images_dir = Path(args.images)
    weights = Path(args.weights)

    samples = load_first_n_samples(csv_path, images_dir, args.limit)
    print(f"[INFO] Loaded {len(samples)} samples for evaluation (requested: {args.limit}).")

    device = get_device()
    print(f"[INFO] Using device: {device}")

    model = load_model(weights, device)
    y_true, y_pred = evaluate(model, samples, device)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    right = int((np.array(y_true) == np.array(y_pred)).sum())
    wrong = len(y_true) - right

    print(f"[RESULT] Accuracy: {acc:.4f}")
    print(f"[RESULT] Correct: {right}")
    print(f"[RESULT] Wrong: {wrong}")
    print("[RESULT] Confusion Matrix:")
    print(cm)

    sens, spec = sensitivity_specificity(cm)
    print("[RESULT] Sensitivity/Specificity by stage:")
    for i in range(5):
        print(f"  Stage {i} ({STAGE_LABELS[i]}): sensitivity={sens[i]:.4f}, specificity={spec[i]:.4f}")

    plot_confusion_matrix(cm, Path(args.cm_out))
    plot_sensitivity_specificity(sens, spec, Path(args.metrics_out))
    print(f"[INFO] Saved confusion matrix plot -> {args.cm_out}")
    print(f"[INFO] Saved sensitivity/specificity plot -> {args.metrics_out}")


if __name__ == "__main__":
    main()