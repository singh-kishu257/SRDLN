"""Explainability module using Grad-CAM for SRDLN-DR-Net.

Purpose:
- Provide visual evidence (saliency/attention) for model predictions.
- Highlight clinically relevant regions (hemorrhages/exudates) to support review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dr_model import SRDLNDRNet
from retinopathy_data import IMAGENET_MEAN, IMAGENET_STD


def _preprocess(image: Image.Image) -> torch.Tensor:
    """Preprocess input image to 224x224 tensor with ImageNet normalization."""

    image = image.convert("RGB")
    image = image.resize((224, 224))

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def generate_saliency_map(
    image_path: str,
    weights_path: str = "SRDLN_clinical_weights.pth",
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Generate a blended Grad-CAM saliency map for a retinal fundus image.

    Args:
        image_path: Path to input retinal image.
        weights_path: Path to trained SRDLN model weights.
        device: Optional explicit torch device.

    Returns:
        RGB uint8 numpy array containing the blended Grad-CAM overlay.
    """

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRDLNDRNet(num_classes=5, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    pil_image = Image.open(image_file).convert("RGB")
    resized_rgb = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0

    input_tensor = _preprocess(pil_image).to(device)

    # Requirement-specific target layer: final block of ResNet backbone.
    target_layers = [model.resnet.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    cam_overlay = show_cam_on_image(resized_rgb, grayscale_cam, use_rgb=True)

    # Convert to BGR then back to RGB for consistent output in mixed pipelines.
    cam_overlay = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)
    cam_overlay = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)
    return cam_overlay