"""SRDLN-DR-Net model definition.

First-principles:
- Reuse a strong visual backbone (ResNet-50) trained on large-scale natural images.
- Replace the final classifier with a deeper ANN "clinical funnel" that can learn
  nuanced boundaries between DR stages.
- Use He/Kaiming initialization for new linear layers to stabilize ReLU training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class SRDLNDRNet(nn.Module):
    """SRDLN-DR-Net: ResNet-50 backbone + custom MLP classification head."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.resnet = resnet50(weights=weights)

        in_features = self.resnet.fc.in_features  # 2048 for ResNet-50

        # Custom clinical funnel from the mission requirements.
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        self._init_head_weights()

    def _init_head_weights(self) -> None:
        """Apply He initialization to all linear layers in custom head."""

        for module in self.resnet.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)