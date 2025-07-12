# GLASS Model Inference Guide

This guide explains how to run the GLASS anomaly detection model on your own images.

## Quick Start

### 1. Single Image Prediction

```bash
# Using ONNX model (recommended for speed)
python predict_single_image.py --image your_image.jpg --model_type onnx

# Using PyTorch model (more flexible)
python predict_single_image.py --image your_image.jpg --model_type pytorch
```

### 2. Batch Processing

```bash
# Process all images in a directory
python batch_predict.py --input_dir ./your_images/ --output_dir ./results/
```

## Available Models

The system has been trained on multiple datasets. Choose the model that best matches your use case:

- **WFDD Grid Cloth**: `results/models/backbone_0/wfdd_grid_cloth/` (Latest trained model)
- **MVTec Carpet**: `results/models/backbone_0/mvtec_carpet/`
- **MVTec Grid**: `results/models/backbone_0/mvtec_grid/`
- **Other MVTec categories**: Available in `results/models/backbone_0/`

## Model Types

### ONNX Model (Recommended)
- **Faster inference** (2-3x speedup)
- **Smaller memory footprint**
- **Better for production deployment**
- Requires `onnxruntime-gpu` package

### PyTorch Model
- **More flexible** for custom modifications
- **Full access to model internals**
- **Better for research and development**
- Requires full PyTorch environment

## Usage Examples

### Example 1: Basic Single Image Prediction

```python
from predict_single_image import GLASSPredictor

# Initialize predictor
predictor = GLASSPredictor(
    model_path="results/models/backbone_0/wfdd_grid_cloth",
    model_type="onnx",
    device="cuda"
)

# Run prediction
results = predictor.predict("your_image.jpg")

# Print results
print(f"Image Score: {results['image_score']:.4f}")
print(f"Is Anomalous: {results['is_anomalous']}")

# Save visualization
predictor.visualize_results("your_image.jpg", results, "output.png")
```

### Example 2: Custom Threshold

```python
# Run prediction with custom threshold
results = predictor.predict("your_image.jpg")

# Custom threshold (default is 0.5)
custom_threshold = 0.3
is_anomalous = results['image_score'] > custom_threshold
print(f"Anomalous (threshold {custom_threshold}): {is_anomalous}")
```

### Example 3: Batch Processing

```python
from batch_predict import process_batch

# Process all images in a directory
process_batch(
    input_dir="./input_images/",
    output_dir="./output_results/",
    model_path="results/models/backbone_0/wfdd_grid_cloth",
    model_type="onnx"
)
```

## Command Line Options

### Single Image Prediction

```bash
python predict_single_image.py [OPTIONS]

Options:
  --image TEXT              Path to input image (required)
  --model_path TEXT         Path to trained model directory
  --model_type [pytorch|onnx]  Model type to use (default: onnx)
  --device [cuda|cpu]       Device to use (default: cuda)
  --output TEXT             Output path for visualization
  --threshold FLOAT         Anomaly threshold (default: 0.5)
```

### Batch Processing

```bash
python batch_predict.py [OPTIONS]

Options:
  --input_dir TEXT          Directory containing input images (required)
  --output_dir TEXT         Directory to save results (required)
  --model_path TEXT         Path to trained model directory
  --model_type [pytorch|onnx]  Model type to use (default: onnx)
  --device [cuda|cpu]       Device to use (default: cuda)
```

## Output Format

### Single Image Output
- **Image Score**: Float value indicating anomaly likelihood (0-1)
- **Is Anomalous**: Boolean based on threshold
- **Anomaly Map**: Pixel-level anomaly localization
- **Visualization**: 4-panel visualization with original, anomaly map, colored map, and overlay

### Batch Processing Output
- **Individual Results**: Each image gets its own prediction files
- **Summary CSV**: `prediction_summary.csv` with all results
- **Anomaly Maps**: Individual anomaly map images
- **Visualizations**: Combined prediction visualizations

## Understanding Results

### Image Score
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher values indicate higher anomaly likelihood
- **Threshold**: Default 0.5, but can be adjusted based on your needs

### Anomaly Map
- **Purpose**: Shows pixel-level anomaly localization
- **Format**: Grayscale heatmap where brighter areas indicate anomalies
- **Size**: 288x288 pixels (resized from input)

### Visualization Panels
1. **Original Image**: Input image resized to 288x288
2. **Anomaly Map**: Raw anomaly scores
3. **Colored Map**: Anomaly map with jet colormap
4. **Overlay**: Original image with colored anomaly overlay

## Performance Tips

### For Speed
- Use ONNX model (`--model_type onnx`)
- Use GPU (`--device cuda`)
- Process images in batches

### For Accuracy
- Use PyTorch model for fine-tuning
- Adjust threshold based on your dataset
- Consider model trained on similar data

## Troubleshooting

### Common Issues

1. **Model not found**
   ```
   Error: Model path results/models/backbone_0/wfdd_grid_cloth not found
   ```
   **Solution**: Check if the model directory exists and contains checkpoint files

2. **ONNX Runtime not available**
   ```
   Warning: ONNX Runtime not available
   ```
   **Solution**: Install with `pip install onnxruntime-gpu`

3. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use CPU (`--device cpu`) or reduce batch size

4. **Image format not supported**
   ```
   Error: Image file not found
   ```
   **Solution**: Ensure image is in supported format (JPG, PNG, BMP, TIFF)

### Model Compatibility

- **Input Size**: All models expect 288x288 pixel images
- **Color Format**: RGB (3 channels)
- **Data Type**: Float32, normalized with ImageNet mean/std

## Advanced Usage

### Custom Preprocessing

```python
# Custom image preprocessing
from PIL import Image
import numpy as np

# Load and preprocess your image
image = Image.open("your_image.jpg").convert("RGB")
image = image.resize((288, 288))
image_array = np.array(image).astype(np.float32) / 255.0

# Run prediction
results = predictor.predict(image)
```

### Multiple Models

```python
# Compare different models
models = [
    "results/models/backbone_0/wfdd_grid_cloth",
    "results/models/backbone_0/mvtec_carpet",
    "results/models/backbone_0/mvtec_grid"
]

for model_path in models:
    predictor = GLASSPredictor(model_path, "onnx", "cuda")
    results = predictor.predict("your_image.jpg")
    print(f"{model_path}: {results['image_score']:.4f}")
```

## Requirements

### For ONNX Inference
```bash
pip install onnxruntime-gpu opencv-python matplotlib numpy pillow
```

### For PyTorch Inference
```bash
pip install torch torchvision opencv-python matplotlib numpy pillow
```

### Full Environment
```bash
pip install -r requirements.txt
```

## Model Performance

Based on the latest training results:

- **WFDD Grid Cloth Model**:
  - Image AUROC: 99.77%
  - Pixel AUROC: 98.57%
  - Best Epoch: 571

- **MVTec Carpet Model**:
  - Best Epoch: 75

- **MVTec Grid Model**:
  - Best Epoch: 21

Choose the model that best matches your application domain for optimal performance. 