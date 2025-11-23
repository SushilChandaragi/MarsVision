# MarsVision-A Machine Learning project on Martian Surface

### Mars Terrain Segmentation using SegFormer 🪐
This project has been visualised using a web app [CURIOSITY](https://curiosity-frontend.onrender.com/) and has its own repository,do check it out [Repo](https://github.com/SushilChandaragi/Curiosity).

This project implements a semantic segmentation pipeline to classify Martian terrain features using deep learning. The goal is to assist autonomous rover navigation by accurately identifying navigable surfaces (soil, bedrock) versus hazards (rocks, deep sand) from raw camera footage.

## 📝 Project Overview

We fine-tuned a **SegFormer** (Transformer-based) model on high-resolution Martian imagery. The project covers the entire ML lifecycle: data preprocessing, distributed multi-GPU training, quantitative evaluation, and video inference.

### The Challenge

Martian terrain is visually homogeneous (mostly red/orange), making it difficult to distinguish between **bedrock** (safe) and **sand** (hazard). Standard computer vision models often struggle with the lack of distinct color contrast.

## 🧠 Model Architecture

We utilized **SegFormer (B0 variant)**, a lightweight vision transformer designed for efficiency.

  * **Encoder:** Mix Transformer (MiT-B0) pre-trained on ImageNet. It extracts hierarchical features at multiple scales.
  * **Decoder:** A lightweight All-MLP decoder that aggregates features to produce the final segmentation mask.
  * **Why this model?** It offers an excellent balance between accuracy and inference speed, making it suitable for potential edge deployment on rover hardware.

## 📊 Dataset: S5Mars

[cite\_start]We utilized the **S5Mars** dataset[cite: 11], which consists of 6,000 high-resolution images captured by the Curiosity rover.

  * **Classes:** 10 total (Sky, Ridge, Soil, Sand, Bedrock, Rock, Rover, Trace, Hole, Background).
  * **Data Split:** We combined "easy" and "hard" difficulty subsets and applied a **70/30 train-test split**.
  * **Augmentation:** Albumentations (Horizontal Flip) was used to improve generalization.

## 🚀 Methodology

1.  **Preprocessing:** Images resized to `512x512`. Masks processed to ensure correct class mapping.
2.  **Training:** Trained for 30 epochs using **Distributed Data Parallel (DDP)** on 2x NVIDIA T4 GPUs.
3.  **Optimization:** AdamW optimizer with a `OneCycleLR` learning rate scheduler.
4.  **Validation:** Model performance was validated after every epoch, saving only the checkpoint with the highest Mean IoU.

## 🏆 Results

The model was evaluated on a hold-out test set of 1,800 images.

| Dataset | Mean IoU (mIoU) | Pixel Accuracy |
| :--- | :--- | :--- |
| **S5Mars (Combined)** | **68.06%** | **88.26%** |
| **AI4Mars (Transfer Learning)** | **85.11%** | **97.17%** |

*Key Finding:* The model generalizes well between "easy" and "hard" terrain images, showing less than a 2% drop in performance on difficult samples.

## 📹 Inference & Visualization

The project includes scripts to:

  * Generate color-coded segmentation masks for static images.
  * Process video footage (e.g., Perseverance landing) to create side-by-side comparisons of raw footage vs. model segmentation.

## 🛠️ Installation & Usage

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/mars-segmentation.git

# 2. Install dependencies
pip install torch transformers opencv-python pandas tqdm albumentations

# 3. Run inference on an image
python predict.py --image_path "path/to/mars_image.jpg" --model_path "checkpoints/best_model.pth"
```

## 📚 Reference

  * **Dataset:** [S5Mars: Semi-Supervised Learning for Mars Semantic Segmentation](https://arxiv.org/abs/2207.01200)
  * **Model:** [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
