#!/bin/bash

# Convert Keyboard GLASS Model to ONNX Format
# This script converts the trained keyboard model to ONNX for optimized inference

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate glass_env

# Set paths
MODEL_PATH="results/models/backbone_0/keyboard_keyboard/ckpt_best_139.pth"
SCRIPT_PATH="keyboard_to_onnx.py"
OUTPUT_DIR="results/models/backbone_0/keyboard_keyboard"

# Function to show usage
show_usage() {
    echo "Keyboard Model ONNX Conversion"
    echo "=============================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model <path>           Path to model checkpoint (default: $MODEL_PATH)"
    echo "  --output <path>          Output path for ONNX model"
    echo "  --simplified             Also create simplified ONNX model"
    echo "  --device <device>        Device to use (cuda/cpu/auto)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                       # Convert with default settings"
    echo "  $0 --simplified          # Create both full and simplified models"
    echo "  $0 --device cpu          # Use CPU for conversion"
    echo ""
}

# Parse command line arguments
MODEL_ARG=""
OUTPUT_ARG=""
SIMPLIFIED_ARG=""
DEVICE_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_ARG="--model $2"
            shift 2
            ;;
        --output)
            OUTPUT_ARG="--output $2"
            shift 2
            ;;
        --simplified)
            SIMPLIFIED_ARG="--simplified"
            shift
            ;;
        --device)
            DEVICE_ARG="--device $2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please ensure training is complete and model checkpoint exists."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Converting keyboard model to ONNX format..."
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the conversion
python "$SCRIPT_PATH" \
    --model "$MODEL_PATH" \
    $OUTPUT_ARG \
    $SIMPLIFIED_ARG \
    $DEVICE_ARG

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ONNX conversion completed successfully!"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"/*.onnx 2>/dev/null || echo "No ONNX files found"
    echo ""
    echo "You can now use the ONNX model for optimized inference."
else
    echo ""
    echo "❌ ONNX conversion failed!"
    exit 1
fi 