#!/bin/bash

# Keyboard Anomaly Detection Inference Script
# This script provides easy access to run inference on the trained keyboard model

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate glass_env

# Set paths
MODEL_PATH="results/models/backbone_0/keyboard_keyboard/ckpt_best_139.pth"
SCRIPT_PATH="keyboard_inference.py"

# Function to show usage
show_usage() {
    echo "Keyboard Anomaly Detection Inference"
    echo "===================================="
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  camera                    Run real-time camera inference"
    echo "  image <path>              Run inference on a single image"
    echo "  batch <directory>         Run inference on all images in directory"
    echo "  test                      Run inference on test images from dataset"
    echo "  help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 camera                 # Start real-time camera inference"
    echo "  $0 image my_keyboard.jpg  # Analyze single image"
    echo "  $0 batch ./test_images/   # Analyze all images in directory"
    echo "  $0 test                   # Test on keyboard dataset test images"
    echo ""
}

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please ensure training is complete and model checkpoint exists."
    exit 1
fi

# Parse command line arguments
case "$1" in
    "camera")
        echo "Starting real-time camera inference..."
        python "$SCRIPT_PATH" --model "$MODEL_PATH" --camera
        ;;
    "image")
        if [ -z "$2" ]; then
            echo "Error: Please provide image path"
            echo "Usage: $0 image <path_to_image>"
            exit 1
        fi
        echo "Running inference on image: $2"
        python "$SCRIPT_PATH" --model "$MODEL_PATH" --input "$2"
        ;;
    "batch")
        if [ -z "$2" ]; then
            echo "Error: Please provide directory path"
            echo "Usage: $0 batch <directory_path>"
            exit 1
        fi
        echo "Running batch inference on directory: $2"
        python "$SCRIPT_PATH" --model "$MODEL_PATH" --input "$2"
        ;;
    "test")
        echo "Running inference on keyboard test images..."
        python "$SCRIPT_PATH" --model "$MODEL_PATH" --input "datasets/keyboard/test"
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "Error: Unknown option '$1'"
        echo ""
        show_usage
        exit 1
        ;;
esac 