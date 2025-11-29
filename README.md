# PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy

[![ACL 2025](https://img.shields.io/badge/ACL-2025-blue)](https://aclanthology.org/2025.acl-long.749/)

<img width="1687" alt="image" src="https://github.com/user-attachments/assets/f1b099cf-2db8-4720-9146-770932794e58" />

**Conference:** ACL 2025, Vienna, Austria

## Overview

This repository contains the implementation of PreP-OCR, a two-stage pipeline for enhancing OCR accuracy on historical documents:

1. **Stage 1: Document Image Restoration** - Applies deblurring models trained on synthetic data to restore degraded document images
2. **Stage 2: Post-OCR Linguistic Correction** - Applies ByT5-based error correction on OCR outputs

**Current Release**: This repository currently provides the synthetic data generation component used for training the image restoration models. For post-OCR correction techniques, please refer to our previous work: [https://aclanthology.org/2024.emnlp-main.862/](https://aclanthology.org/2024.emnlp-main.862/)

## Key Features

### 🎯 Synthetic Data Generation
- **Text-to-Image Conversion**: Generate clean document images with various fonts and layouts
- **Multi-level Degradation**: Create 4 different noise levels simulating real scanning artifacts
- **Historical Document Simulation**: Add paper textures, stains, and aging effects
- **Ground Truth Generation**: Automatically generate pixel-perfect annotations

## Quick Start

### Prerequisites
```bash
# Create conda environment
conda create -n prep python=3.9 -y
conda activate prep

# Install dependencies
pip install numpy pillow opencv-python matplotlib tqdm
```

### Generate Synthetic Training Data

```bash
# Preprocess background images (first time only)
python prepare_backgrounds.py

# Generate 5 base images + 20 noisy variants
python generate_ocr_data.py --base 5

# Generate larger datasets
python generate_ocr_data.py --base 50  # 50 base + 200 variants
```

## Repository Structure

```
PreP-OCR/
├── data/
│   ├── Novel_data_UTF8_processed/    # Source text files
│   └── output/                       # Generated synthetic data
│       ├── clean/                    # Clean document images
│       ├── noisy/                    # Degraded variants
│       └── ground_truth/             # Text annotations
├── font/latin/                       # Font collection
├── noise_img/                        # Background textures and stains
├── funtion/generate_base_add_noise.py # Core generation algorithms
├── generate_ocr_data.py              # Main data generation script
└── prepare_backgrounds.py            # Background preprocessing
```

## Applications

This synthetic data generation tool is designed for:
- **Historical Document Digitization**
- **OCR Model Training** on degraded documents
- **Document Image Restoration** research
- **Multi-lingual OCR** system development

## Model Weights

### Document Image Deblurring Model

**Download Links:**
- Model Checkpoint: [resshift_deblur_prep_ocr.pth](https://huggingface.co/ShuhaoGuan/prep-ocr-resshift-deblur/resolve/main/resshift_deblur_prep_ocr.pth) (457MB)
- VQ-VAE Autoencoder: [autoencoder_vq_f4.pth](https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth)
- Model Configuration: [deblur_gopro256.yaml](configs/deblur_gopro256.yaml)