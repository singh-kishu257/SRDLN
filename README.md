# SRDLN — Salient Retinopathy Deep Learning Network

> An end-to-end deep learning system for automated Diabetic Retinopathy (DR) detection and grading from retinal fundus photographs.

---

## Overview

SRDLN is a clinical-grade convolutional neural network that classifies retinal fundus images into five DR severity grades (0–4) using a fine-tuned ResNet-50 backbone with a custom multi-layer perceptron head. The system includes a Gradio-powered web interface for real-time inference and Grad-CAM saliency visualization to explain model predictions.

Diabetic Retinopathy is the leading cause of preventable blindness worldwide. Early automated detection can significantly improve patient outcomes by enabling timely clinical intervention.

---

## DR Grading Scale

| Grade | Severity | Description |
|-------|----------|-------------|
| 0 | No DR | No signs of diabetic retinopathy |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | More than microaneurysms but less than severe |
| 3 | Severe NPDR | Extensive hemorrhages, venous beading |
| 4 | Proliferative DR | Neovascularization, vitreous hemorrhage |

---

## Model Architecture

- **Backbone:** ResNet-50 (ImageNet pretrained)
- **Head:** 4-layer MLP — `2048 → 1024 → 512 → 256 → 5`
- **Loss:** Weighted CrossEntropyLoss (class imbalance correction)
- **Optimizer:** Adam (`lr=1e-4`)
- **Input:** 224×224 RGB retinal fundus images
- **Output:** 5-class softmax probability distribution

---

## Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | ~94% |
| Training Epochs | 28 (18 initial + 10 resumed) |
| Best Val Loss | 0.18 |
| Training Hardware | NVIDIA L4 GPU (GCP) |

---

## Project Structure

```
SRDLN/
├── app.py                        # Gradio web interface + inference + Grad-CAM
├── dr_model.py                   # ResNet-50 backbone + custom MLP head
├── retinopathy_data.py           # Data engine, CSV parsing, WeightedRandomSampler
├── explain_ai.py                 # Grad-CAM saliency map generator
├── train_gcp.py                  # Training script with checkpoint resume support
├── evaluate.py                   # Evaluation script — confusion matrix + metrics
├── SRDLN_clinical_weights.pth    # Trained model weights (see below)
└── README.md
```

---

## Dataset

This project uses the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset from Kaggle.

- **3,662** retinal fundus images
- **5 classes** (DR grades 0–4)
- Significant class imbalance handled via `WeightedRandomSampler` and weighted loss

> Download `train.csv` and `train_images/` from Kaggle and place them in the project root before training.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/SRDLN.git
cd SRDLN
pip install torch torchvision gradio grad-cam opencv-python pillow scikit-learn seaborn matplotlib
```

---

## Usage

### Run the Web App

```bash
python app.py
```

Then open `http://localhost:7860` in your browser. Upload a retinal fundus image and the model will return:
- Predicted DR grade (0–4)
- Confidence score
- Grad-CAM saliency overlay highlighting the regions driving the prediction

### Train from Scratch

```bash
python train_gcp.py \
  --train-csv train.csv \
  --train-image-dir train_images \
  --val-csv train.csv \
  --val-image-dir train_images \
  --epochs 30
```

### Resume Training from Checkpoint

```bash
python train_gcp.py \
  --train-csv train.csv \
  --train-image-dir train_images \
  --val-csv train.csv \
  --val-image-dir train_images \
  --resume SRDLN_clinical_weights.pth \
  --epochs 20
```

### Evaluate Model

```bash
python evaluate.py \
  --csv train.csv \
  --images train_images \
  --weights SRDLN_clinical_weights.pth
```

Outputs:
- Overall accuracy, correct/wrong counts
- Per-class sensitivity and specificity
- `confusion_matrix.png`
- `sensitivity_specificity.png`

---

## Model Weights

The trained weights (`SRDLN_clinical_weights.pth`, ~100MB) are not included in this repository due to file size.

Download from: [Google Cloud Storage — srdln-training bucket] *(or add your own link here)*

Place the file in the project root before running inference or evaluation.

---

## Training Infrastructure

- **Platform:** Google Cloud Platform (GCP)
- **VM:** `srdlntraining`
- **GPU:** NVIDIA L4 (24GB VRAM)
- **CUDA:** 12.4
- **Framework:** PyTorch 2.x

---

## Explainability

SRDLN uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate visual explanations for each prediction. The saliency overlay highlights which regions of the retina most influenced the model's grading decision, supporting clinical interpretability.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

- [APTOS 2019 Blindness Detection Challenge](https://www.kaggle.com/competitions/aptos2019-blindness-detection) — Aravind Eye Hospital
- [PyTorch](https://pytorch.org/)
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) by Jacob Gildenblat
- [Gradio](https://gradio.app/)
