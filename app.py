"""SRDLN Medical Gradio Interface.

First-principles UX:
- Keep workflow linear: upload retinal image -> receive stage diagnosis + confidence + evidence map.
- Show confidence distribution so clinicians can assess uncertainty.
- Provide Grad-CAM overlay for transparent review.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dr_model import SRDLNDRNet
from explain_ai import generate_saliency_map
from retinopathy_data import IMAGENET_MEAN, IMAGENET_STD


STAGE_LABELS = {
    0: "Stage 0: No DR",
    1: "Stage 1: Mild Non-Proliferative DR",
    2: "Stage 2: Moderate Non-Proliferative DR",
    3: "Stage 3: Severe Non-Proliferative DR",
    4: "Stage 4: Proliferative DR",
}


class SRDLNInferenceEngine:
    """Encapsulates model loading and forward inference."""

    def __init__(self, weights_path: str = "SRDLN_clinical_weights.pth") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SRDLNDRNet(num_classes=5, pretrained=False).to(self.device)

        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise FileNotFoundError(
                f"Missing model weights at '{weights_path}'. Train first with train_gcp.py."
            )

        state_dict = torch.load(weights_file, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(448),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def predict(self, image: Image.Image) -> Tuple[int, Dict[str, float]]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        conf_map = {STAGE_LABELS[i]: float(probs[i]) for i in range(5)}
        return pred_idx, conf_map


engine = None


def _get_engine() -> SRDLNInferenceEngine:
    global engine
    if engine is None:
        engine = SRDLNInferenceEngine()
    return engine


def diagnose_retinopathy(input_image: Image.Image):
    """Predict DR stage and return diagnosis label, confidence bar data, Grad-CAM image."""

    model_engine = _get_engine()
    pred_idx, conf_map = model_engine.predict(input_image)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
    input_image.save(temp_path)

    heatmap = generate_saliency_map(temp_path, device=model_engine.device)

    return STAGE_LABELS[pred_idx], conf_map, heatmap


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"), title="SRDLN Medical") as demo:
    gr.Markdown("""
    # 🏥 SRDLN Medical — Diabetic Retinopathy Diagnostic Suite
    Upload a retinal fundus image to get an AI-assisted Diabetic Retinopathy stage estimate,
    calibrated confidence profile, and Grad-CAM saliency map for visual verification.
    """)

    with gr.Row():
        input_image = gr.Image(type="pil", label="Retinal Fundus Input")
        output_heatmap = gr.Image(type="numpy", label="Grad-CAM Verification Overlay")

    diagnosis_text = gr.Textbox(label="Diagnosis Label", lines=1)
    confidence_bar = gr.Label(label="Confidence Score Bar")

    diagnose_btn = gr.Button("Run Clinical Assessment")
    diagnose_btn.click(
        fn=diagnose_retinopathy,
        inputs=[input_image],
        outputs=[diagnosis_text, confidence_bar, output_heatmap],
    )

print("SRDLN Medical Gradio interface is ready. Access it at http://localhost:7860.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)