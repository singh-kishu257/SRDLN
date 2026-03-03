"""SRDLN training engine for local + GCP execution — with loss/accuracy graphs."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from dr_model import SRDLNDRNet
from retinopathy_data import (
    RetinopathyImageDataset,
    _build_eval_transform,
    _build_train_transform,
    _clinical_sampler_weights,
    discover_samples_from_csv,
)


def pick_device() -> torch.device:
    try:
        import torch_directml  # type: ignore
        return torch_directml.device()
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, epochs):
    epoch_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SRDLN Training Curves", fontsize=14, fontweight="bold")

    ax1.plot(epoch_range, train_losses, "b-o", label="Train Loss")
    ax1.plot(epoch_range, val_losses,   "r-o", label="Val Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epoch_range, train_accs, "b-o", label="Train Accuracy")
    ax2.plot(epoch_range, val_accs,   "r-o", label="Val Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("SRDLN_training_curves.png")
    plt.show()
    print("[INFO] Graph saved as SRDLN_training_curves.png")


def train(args: argparse.Namespace) -> None:
    train_samples = discover_samples_from_csv(args.train_csv, args.train_image_dir)
    val_samples   = discover_samples_from_csv(args.val_csv,   args.val_image_dir)
    print(f"[INFO] Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    train_dataset = RetinopathyImageDataset(train_samples, transform=_build_train_transform(224))
    val_dataset   = RetinopathyImageDataset(val_samples,   transform=_build_eval_transform(224))

    train_labels   = [label for _, label in train_samples]
    sample_weights = _clinical_sampler_weights(train_labels)
    sampler        = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,   num_workers=args.num_workers, pin_memory=True)

    device = pick_device()
    print(f"[INFO] Device: {device}")

    model = SRDLNDRNet(num_classes=5, pretrained=True).to(device)

    # Load checkpoint if --resume is provided
    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict)
        print("[INFO] Checkpoint loaded successfully.")

    # Weighted cross entropy for extra class imbalance correction
    class_weights = torch.tensor([0.49, 1.20, 0.80, 2.10, 1.80]).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    best_path     = "SRDLN_clinical_weights.pth"

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  --> Best model saved! (val_loss={val_loss:.4f})")

    print("[INFO] Training complete.")
    plot_curves(train_losses, val_losses, train_accs, val_accs, args.epochs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRDLN-DR-Net on DR dataset")
    parser.add_argument("--train-csv",       type=str, default="train.csv")
    parser.add_argument("--val-csv",         type=str, default="test.csv")
    parser.add_argument("--train-image-dir", type=str, default="train_images")
    parser.add_argument("--val-image-dir",   type=str, default="test_images")
    parser.add_argument("--batch-size",      type=int, default=16)
    parser.add_argument("--epochs",          type=int, default=18)
    parser.add_argument("--num-workers",     type=int, default=2)
    parser.add_argument("--cloud-optimized", action="store_true")
    parser.add_argument("--resume",          type=str, default=None,
                        help="Path to checkpoint .pth file to resume training from")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
