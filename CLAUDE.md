# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLASS is a unified anomaly synthesis strategy with gradient ascent for industrial anomaly detection and localization. It's implemented in PyTorch and designed to enhance unsupervised anomaly detection by addressing limitations in coverage and controllability of existing anomaly synthesis strategies.

## Core Architecture

The repository implements the GLASS framework with the following key components:

- **GLASS Model** (`glass.py`): Main model class implementing the anomaly detection algorithm with discriminator training, gradient ascent optimization, and feature embedding
- **Backbone Networks** (`backbones.py`): Feature extraction networks (ResNet, WideResNet) for encoding input images
- **Model Components** (`model.py`): Discriminator and Projection network implementations
- **Data Processing** (`datasets/`): Dataset loaders for MVTec AD, VisA, MPDD, WFDD, and custom keyboard datasets
- **Common Utilities** (`common.py`): Feature aggregation, preprocessing, and segmentation utilities
- **Loss Functions** (`loss.py`): FocalLoss implementation for training
- **Metrics** (`metrics.py`): AUROC, AP, and PRO evaluation metrics

## Development Environment

Create conda environment and install dependencies:
```bash
conda create -n glass_env python=3.9.15
conda activate glass_env
pip install -r requirements.txt
```

Requires NVIDIA Tesla A800 (80GB) or similar GPU with CUDA support.

## Common Commands

### Training
Train GLASS model on a dataset (uses shell scripts):
```bash
# MVTec AD training
bash shell/run-mvtec.sh

# WFDD training  
bash shell/run-wfdd.sh

# Keyboard dataset training
bash shell/run-keyboard.sh

# Custom dataset training
bash shell/run-custom.sh
```

### Testing
Test trained model (change `--test` parameter in shell scripts):
```bash
# Edit shell script to set --test test instead of --test ckpt
bash shell/run-mvtec.sh
```

### Inference
Single image prediction:
```bash
python predict_single_image.py --image_path <path> --model_path <path>
```

Camera inference:
```bash
python camera_inference.py
```

Video inference:
```bash
python video_inference.py
```

### ONNX Conversion
Convert PyTorch model to ONNX:
```bash
python keyboard_to_onnx.py
bash shell/convert-keyboard-to-onnx.sh
```

## Key Configuration Parameters

Models are configured via command-line arguments in shell scripts:
- `--meta_epochs`: Training epochs (default: 640)
- `--eval_epochs`: Evaluation frequency (default: 1)
- `--batch_size`: Training batch size (default: 8)
- `--imagesize`: Input image size (default: 288)
- `--pretrain_embed_dimension`: Feature embedding dimension (default: 1536)
- `--backbone`: Feature extraction network (wideresnet50, resnet50, etc.)
- `--layers_to_extract_from`: Which layers to extract features from
- `--distribution`: Distribution type for anomaly synthesis (0-4)

## Dataset Structure

Datasets follow standard structure:
```
datasets/
  dataset_name/
    train/
      good/  # Normal samples
    test/
      good/  # Normal test samples
      defect_type/  # Anomaly samples
    ground_truth/
      defect_type/  # Pixel-level masks
```

## Model Training Flow

1. Feature extraction using backbone networks
2. Patch-based feature embedding and aggregation
3. Discriminator training with gradient ascent anomaly synthesis
4. Center computation for normal feature distribution
5. Evaluation on validation set with AUROC/AP metrics

## Results and Outputs

Training results are saved to:
- `results/models/`: Trained model checkpoints
- `results/training/`: Training visualizations
- `results/eval/`: Final evaluation results
- `results/results.csv`: Quantitative metrics

## Dataset Distribution Analysis

The system automatically determines optimal distribution type for each dataset class and saves results to `datasets/excel/[dataset]_distribution.xlsx`.

## Key Files for Modification

- `glass.py`: Core algorithm implementation
- `main.py`: CLI interface and training orchestration  
- `datasets/[dataset].py`: Dataset-specific data loaders
- `shell/run-[dataset].sh`: Training configuration scripts
- `configs/[dataset].yaml`: Model configuration files (if present)