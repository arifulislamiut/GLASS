# Fabric Defect Detection with GLASS on Jetson Orin Nano 8GB

## Overview

This document provides a comprehensive deployment plan for implementing fabric defect detection using the GLASS (Gradient-based Localized Anomaly Synthesis Strategy) algorithm on NVIDIA Jetson Orin Nano 8GB edge device.

## System Architecture

```
[Fabric Sample] → [Camera] → [Jetson Orin Nano] → [GLASS Model] → [Defect Detection Results]
                                     ↓
                            [TensorRT Optimization]
                                     ↓
                              [Real-time Inference]
```

## Hardware Specifications

### Jetson Orin Nano 8GB
- **Memory**: 8GB LPDDR5 shared CPU/GPU (68-102 GB/s bandwidth)
- **GPU**: 1024 CUDA cores, 32 Tensor cores, 625 MHz
- **AI Performance**: 40-67 TOPS (INT8/FP16)
- **Power**: 7-25W configurable (15W optimal)
- **CPU**: 6-core ARM Cortex-A78AE

### Performance Constraints
- Limited 8GB shared memory
- Power budget: 15W for sustained operation
- ARM architecture compatibility required

## Implementation Plan

### Phase 1: Model Optimization (Weeks 1-2)

#### Memory Optimization
- **Input Resolution**: Reduce from 288x288 to 256x256
- **Backbone Network**: Consider lightweight alternatives
  - Current: WideResNet50 (94M parameters)
  - Alternative: EfficientNet-B0 (5.3M parameters)
  - Alternative: MobileNetV3-Large (5.4M parameters)
- **Feature Dimensions**: Reduce from 1536 to 512-768
- **Batch Size**: Set to 1 for inference

#### Model Architecture Modifications
```python
# Optimized GLASS configuration for Jetson
JETSON_CONFIG = {
    'input_size': 256,
    'backbone': 'efficientnet_b0',
    'pretrain_embed_dimension': 768,
    'target_embed_dimension': 768,
    'batch_size': 1,
    'layers_to_extract_from': ['layer3', 'layer4']
}
```

### Phase 2: ONNX and TensorRT Pipeline (Weeks 3-4)

#### Step 1: ONNX Conversion
```bash
# Create fabric-specific ONNX conversion script
python fabric_to_onnx.py \
  --model_path results/models/fabric_glass.pth \
  --input_size 256 \
  --batch_size 1 \
  --output_path fabric_glass.onnx
```

#### Step 2: TensorRT Engine Generation
```bash
# FP16 optimization
trtexec --onnx=fabric_glass.onnx \
        --saveEngine=fabric_glass_fp16.trt \
        --fp16 \
        --workspace=4096 \
        --minShapes=input:1x3x256x256 \
        --maxShapes=input:1x3x256x256 \
        --optShapes=input:1x3x256x256

# INT8 optimization (requires calibration dataset)
trtexec --onnx=fabric_glass.onnx \
        --saveEngine=fabric_glass_int8.trt \
        --int8 \
        --calib=calibration_cache.bin \
        --workspace=4096
```

#### Step 3: Performance Validation
| Precision | Memory Usage | Inference Time | Accuracy Loss |
|-----------|--------------|----------------|---------------|
| FP32      | ~6GB         | ~150ms         | 0%            |
| FP16      | ~4GB         | ~80ms          | <2%           |
| INT8      | ~3GB         | ~50ms          | <5%           |

### Phase 3: Dataset Preparation (Weeks 5-6)

#### Dataset Structure
```
datasets/fabric/
├── train/
│   └── good/                    # 1000+ normal fabric images
├── test/
│   ├── good/                    # 200+ normal test images
│   ├── hole/                    # Hole defects
│   ├── stain/                   # Stain defects
│   ├── wrinkle/                 # Wrinkle defects
│   ├── tear/                    # Tear defects
│   └── color_variation/         # Color inconsistencies
└── ground_truth/
    ├── hole/                    # Pixel-level masks
    ├── stain/
    ├── wrinkle/
    ├── tear/
    └── color_variation/
```

#### Data Collection Guidelines
- **Resolution**: 256x256 pixels (consistent preprocessing)
- **Lighting**: Industrial lighting conditions (fluorescent, LED)
- **Fabric Types**: Cotton, polyester, blends, various textures
- **Defect Severity**: Range from subtle to obvious defects
- **Background**: Consistent industrial inspection setup

#### Training Configuration
```bash
# Fabric-specific training script
python main.py net \
  -b efficientnet_b0 \
  -le layer3 layer4 \
  --pretrain_embed_dimension 768 \
  --target_embed_dimension 768 \
  --meta_epochs 500 \
  --batch_size 8 \
  --imagesize 256 \
  --lr 0.0001 \
  fabric --data_path datasets/fabric
```

### Phase 4: Jetson Environment Setup (Weeks 7-8)

#### System Requirements
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip cmake build-essential
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y v4l-utils # For USB camera support

# Python packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tensorrt pycuda
pip3 install onnxruntime-gpu
pip3 install opencv-python numpy pandas tqdm click
```

#### Docker Deployment (Alternative)
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app
COPY . /app

# Install requirements
RUN pip3 install -r requirements_jetson.txt

# Setup environment
ENV CUDA_VISIBLE_DEVICES=0
ENV TRT_LOGGER_LEVEL=WARNING

CMD ["python3", "fabric_inference_jetson.py"]
```

#### Camera Integration Options
1. **USB Industrial Cameras**
   - Recommended: Basler acA1920-40uc (2MP, USB 3.0)
   - Alternative: FLIR Blackfly S (1.3MP, USB 3.0)

2. **CSI Cameras**
   - Raspberry Pi Camera Module V2 (8MP)
   - IMX219 sensor modules

3. **IP Cameras**
   - Network-based industrial cameras
   - RTSP/HTTP stream support

### Phase 5: Inference Pipeline Implementation (Weeks 9-10)

#### Real-time Processing Pipeline
```python
# fabric_inference_jetson.py structure
import tensorrt as trt
import pycuda.driver as cuda
import cv2
import numpy as np

class FabricDefectDetector:
    def __init__(self, engine_path, input_size=256):
        self.engine_path = engine_path
        self.input_size = input_size
        self.load_engine()
        
    def load_engine(self):
        # Load TensorRT engine
        # Initialize CUDA context
        # Allocate GPU memory
        
    def preprocess(self, image):
        # Resize to input_size
        # Normalize using ImageNet stats
        # Convert to CHW format
        
    def inference(self, image):
        # Copy input to GPU
        # Run inference
        # Copy output from GPU
        
    def postprocess(self, output, threshold=0.5):
        # Generate anomaly map
        # Apply threshold
        # Create binary mask
        
    def detect_defects(self, image):
        # End-to-end detection pipeline
        preprocessed = self.preprocess(image)
        output = self.inference(preprocessed)
        anomaly_map, binary_mask = self.postprocess(output)
        return anomaly_map, binary_mask
```

#### Camera Capture Loop
```python
def main():
    detector = FabricDefectDetector('fabric_glass_fp16.trt')
    cap = cv2.VideoCapture(0)  # USB camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect defects
        anomaly_map, binary_mask = detector.detect_defects(frame)
        
        # Visualize results
        display_frame = visualize_results(frame, anomaly_map, binary_mask)
        cv2.imshow('Fabric Defect Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

### Phase 6: Performance Optimization (Weeks 11-12)

#### Target Performance Metrics
- **Inference Latency**: <100ms per frame
- **Memory Usage**: <6GB total system memory
- **Throughput**: 10-15 FPS sustained
- **Power Consumption**: <20W average
- **Accuracy**: >95% of original model performance

#### Optimization Techniques
1. **CUDA Streams**: Parallel processing
2. **Memory Pooling**: Reduce allocation overhead
3. **Pipeline Parallelism**: Overlap capture and inference
4. **Dynamic Batching**: Process multiple regions simultaneously

#### Benchmarking Script
```python
def benchmark_performance():
    metrics = {
        'latency': [],
        'memory_usage': [],
        'fps': [],
        'power_consumption': []
    }
    
    # Run 1000 inference cycles
    for i in range(1000):
        start_time = time.time()
        result = detector.detect_defects(test_image)
        latency = time.time() - start_time
        
        metrics['latency'].append(latency)
        # Collect other metrics
    
    return analyze_metrics(metrics)
```

## Deployment Configuration Files

### requirements_jetson.txt
```
torch>=1.13.0
torchvision>=0.14.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.62.0
click>=8.0.0
tensorrt>=8.5.0
pycuda>=2022.1
onnxruntime-gpu>=1.12.0
```

### fabric_config.yaml
```yaml
model:
  backbone: efficientnet_b0
  input_size: 256
  embed_dimension: 768
  layers: [layer3, layer4]

inference:
  engine_path: models/fabric_glass_fp16.trt
  batch_size: 1
  threshold: 0.5
  
camera:
  device_id: 0
  resolution: [1920, 1080]
  fps: 30
  
output:
  save_results: true
  result_path: results/fabric_detection/
  display_window: true
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Memory Errors**
   - Reduce input resolution further (256→224)
   - Use INT8 quantization
   - Clear GPU memory between inferences

2. **Low FPS Performance**
   - Enable CUDA streams
   - Optimize image preprocessing
   - Use hardware-accelerated video decode

3. **Model Accuracy Loss**
   - Fine-tune with fabric-specific data
   - Adjust anomaly threshold
   - Use calibration dataset for INT8

4. **Camera Integration Issues**
   - Check V4L2 compatibility
   - Verify USB 3.0 connection
   - Test with `v4l2-ctl --list-devices`

## Testing and Validation

### Test Scenarios
1. **Synthetic Defects**: Controlled fabric samples with known defects
2. **Real Industrial Samples**: Actual fabric defects from manufacturing
3. **Edge Cases**: Various lighting, angles, fabric types
4. **Continuous Operation**: 24/7 stability testing
5. **Environmental Conditions**: Temperature, vibration tolerance

### Success Criteria
- **Detection Accuracy**: >90% for all defect types
- **False Positive Rate**: <5%
- **Processing Speed**: 10+ FPS sustained
- **System Uptime**: >99% over 48-hour test period
- **Power Efficiency**: <20W average consumption

## Maintenance and Updates

### Regular Maintenance Tasks
- Model retraining with new fabric samples (monthly)
- Performance monitoring and optimization
- System updates and security patches
- Camera calibration and cleaning

### Update Procedures
1. Collect new defect samples
2. Retrain model with updated dataset
3. Convert to optimized TensorRT engine
4. Deploy and validate performance
5. Rollback if performance degrades

## Conclusion

This deployment plan provides a comprehensive roadmap for implementing fabric defect detection using GLASS on Jetson Orin Nano 8GB. The approach balances performance requirements with hardware constraints while maintaining high detection accuracy for industrial applications.

Key success factors:
- Proper model optimization for edge deployment
- Efficient TensorRT conversion and optimization
- Robust camera integration and preprocessing
- Comprehensive testing and validation
- Ongoing maintenance and improvement processes

Expected deployment timeline: 12 weeks from initiation to production deployment.