# Keyboard Dataset Training Guide

This guide explains how to train GLASS on your custom keyboard dataset for anomaly detection.

## Dataset Overview

Your keyboard dataset has been successfully configured with:
- **Train images**: 178 good keyboard images
- **Test images**: 8 good + 19 defective keyboard images
- **Augmentation**: 5,640 DTD texture images for anomaly synthesis

## Dataset Structure

```
datasets/keyboard/
├── train/
│   └── good/
│       ├── frame_0001.jpg
│       ├── frame_0002.jpg
│       └── ... (178 images)
└── test/
    ├── good/
    │   ├── WIN_20250420_12_00_23_Pro.jpg
    │   └── ... (8 images)
    └── defective/
        ├── WIN_20250420_12_00_18_Pro.jpg
        └── ... (19 images)
```

## Training Configuration

The training script (`shell/run-keyboard.sh`) is configured with:

### Model Architecture
- **Backbone**: WideResNet50
- **Feature Layers**: layer2, layer3
- **Embedding Dimensions**: 1536
- **Patch Size**: 3x3

### Training Parameters
- **Meta Epochs**: 640
- **Evaluation Epochs**: 1
- **Batch Size**: 8
- **Learning Rate**: 0.0001
- **Image Size**: 288x288

### Data Augmentation
- **Random Augmentation**: Enabled
- **Foreground Masks**: Disabled (no pixel-level annotations)
- **Distribution**: 0 (auto-detect)

## Start Training

### Option 1: Using the provided script
```bash
conda activate glass_env
bash shell/run-keyboard.sh
```

### Option 2: Manual command
```bash
conda activate glass_env
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 640 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --rand_aug 1 \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 \
    -d keyboard \
    keyboard \
    datasets/keyboard \
    datasets/dtd/images
```

## Training Process

1. **Model Initialization**: Loads WideResNet50 backbone
2. **Feature Extraction**: Extracts features from layer2 and layer3
3. **Anomaly Synthesis**: Uses gradient ascent to create synthetic anomalies
4. **Discriminator Training**: Trains discriminator to distinguish normal from anomalous features
5. **Evaluation**: Evaluates performance every epoch
6. **Checkpoint Saving**: Saves best model based on validation performance

## Expected Training Time

- **Hardware**: NVIDIA GPU (recommended)
- **Estimated Time**: 2-4 hours depending on GPU
- **Memory Usage**: ~8GB GPU memory

## Monitoring Training

### TensorBoard Logs
Training progress is logged to:
```
results/models/backbone_0/keyboard/tb/
```

To monitor training:
```bash
tensorboard --logdir results/models/backbone_0/keyboard/tb/
```

### Console Output
The training will show:
- Epoch progress
- Loss values
- Performance metrics (AUROC, AP)
- Best model saving

## Training Output

### Model Files
- `ckpt.pth`: Latest checkpoint
- `ckpt_best_X.pth`: Best model at epoch X
- `glass.onnx`: ONNX export for deployment

### Results
- Training visualizations in `results/training/keyboard/`
- Evaluation results in `results/eval/keyboard/`
- Performance metrics in `results/results.csv`

## Using Trained Model

After training, you can use the model for inference:

```bash
# Single image prediction
python predict_single_image.py \
    --image your_keyboard_image.jpg \
    --model_path results/models/backbone_0/keyboard \
    --model_type onnx

# Camera inference
python camera_inference.py \
    --model_path results/models/backbone_0/keyboard

# Video inference
python video_inference.py \
    --video your_keyboard_video.mp4 \
    --model_path results/models/backbone_0/keyboard
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 4`
   - Use CPU: `--device cpu`

2. **Dataset Loading Errors**
   - Check dataset structure matches expected format
   - Ensure all images are valid JPG/PNG files

3. **Training Slow**
   - Use GPU acceleration
   - Reduce image size: `--imagesize 224`

4. **Poor Performance**
   - Increase training epochs: `--meta_epochs 1000`
   - Adjust learning rate: `--lr 0.0005`
   - Add more training data

### Performance Optimization

- **GPU Memory**: Monitor with `nvidia-smi`
- **Data Loading**: Use SSD storage for faster I/O
- **Augmentation**: Adjust `--rand_aug` for more/less augmentation

## Expected Results

Based on similar datasets, you can expect:
- **Image AUROC**: 90-95%
- **Pixel AUROC**: 85-90% (if you add pixel-level annotations later)
- **Training Time**: 2-4 hours
- **Model Size**: ~15MB

## Next Steps

1. **Start Training**: Run the training script
2. **Monitor Progress**: Use TensorBoard
3. **Evaluate Results**: Check performance metrics
4. **Deploy Model**: Use for real-time inference
5. **Fine-tune**: Adjust parameters if needed

## Customization

### Hyperparameters to Tune
- `--meta_epochs`: Training duration
- `--batch_size`: Memory vs speed trade-off
- `--lr`: Learning rate
- `--noise`: Anomaly synthesis noise
- `--radius`: Feature distribution radius

### Dataset Improvements
- Add more training images
- Include pixel-level annotations (ground_truth masks)
- Balance good/defective samples
- Improve image quality and consistency 