#!/bin/bash
# Setup Python environment for GLASS detection

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Setting up GLASS Python environment"
echo "===================================="

# Check if we're on Jetson
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null)
    print_status "Detected platform: $MODEL"
else
    print_status "Platform: Generic Linux"
fi

# Check for existing virtual environments
VENV_PATHS=(
    "$HOME/venv_detectron2"
    "$HOME/.venv"
    "$HOME/venv"
    "$HOME/glass_env"
)

EXISTING_VENV=""
for path in "${VENV_PATHS[@]}"; do
    if [ -d "$path" ]; then
        EXISTING_VENV="$path"
        print_status "Found existing virtual environment: $path"
        break
    fi
done

if [ -n "$EXISTING_VENV" ]; then
    print_status "Testing existing environment..."
    source "$EXISTING_VENV/bin/activate"
    
    # Test if required packages are available
    python3 -c "import torch, cv2, numpy" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_status "✅ Existing environment is functional"
        echo "Environment ready at: $EXISTING_VENV"
        exit 0
    else
        print_warning "Existing environment missing packages"
    fi
    
    deactivate 2>/dev/null
fi

# Create new virtual environment
VENV_PATH="$HOME/glass_env"
print_status "Creating new virtual environment at $VENV_PATH"

python3 -m venv "$VENV_PATH"
if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment"
    print_error "Try: sudo apt install python3-venv"
    exit 1
fi

print_status "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

print_status "Upgrading pip..."
pip install --upgrade pip

print_status "Installing basic requirements..."
pip install numpy opencv-python pillow

# Try to install PyTorch
print_status "Installing PyTorch..."
if [ -f /proc/device-tree/model ] && grep -q "Jetson" /proc/device-tree/model 2>/dev/null; then
    print_status "Installing PyTorch for Jetson..."
    # Use Jetson-specific PyTorch wheel
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_status "Installing PyTorch for generic platform..."
    pip install torch torchvision torchaudio
fi

# Install additional requirements if available
if [ -f "requirements_jetson.txt" ]; then
    print_status "Installing from requirements_jetson.txt..."
    pip install -r requirements_jetson.txt
elif [ -f "requirements.txt" ]; then
    print_status "Installing from requirements.txt..."
    pip install -r requirements.txt
fi

print_status "Testing installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

if [ $? -eq 0 ]; then
    print_status "✅ Environment setup completed successfully!"
    echo "Virtual environment created at: $VENV_PATH"
    echo "To activate manually: source $VENV_PATH/bin/activate"
else
    print_error "Environment setup failed during testing"
    exit 1
fi