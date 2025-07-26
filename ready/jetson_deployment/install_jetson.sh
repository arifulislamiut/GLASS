#!/bin/bash
# GLASS Fabric Detection - Jetson Orin Nano Installation Script

set -e  # Exit on any error

echo "🚀 GLASS Fabric Detection - Jetson Orin Nano Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if running on Jetson
check_jetson() {
    print_header "Checking Jetson Platform"
    
    if [ ! -f "/proc/device-tree/model" ]; then
        print_error "Not running on Jetson platform!"
        exit 1
    fi
    
    MODEL=$(cat /proc/device-tree/model)
    print_status "Detected: $MODEL"
    
    if [[ "$MODEL" != *"Orin"* ]]; then
        print_warning "This script is optimized for Jetson Orin Nano"
        print_warning "Your device: $MODEL"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check system requirements
check_requirements() {
    print_header "Checking System Requirements"
    
    # Check Ubuntu version
    UBUNTU_VERSION=$(lsb_release -rs)
    print_status "Ubuntu version: $UBUNTU_VERSION"
    
    if [[ "$UBUNTU_VERSION" < "20.04" ]]; then
        print_error "Ubuntu 20.04+ required"
        exit 1
    fi
    
    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    print_status "Available memory: ${MEMORY_GB}GB"
    
    if [ "$MEMORY_GB" -lt 6 ]; then
        print_warning "Low memory detected. System may run slowly."
    fi
    
    # Check available storage
    STORAGE_GB=$(df -BG . | awk 'NR==2{print $4}' | tr -d 'G')
    print_status "Available storage: ${STORAGE_GB}GB"
    
    if [ "$STORAGE_GB" -lt 4 ]; then
        print_error "Insufficient storage space (4GB+ required)"
        exit 1
    fi
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
        print_status "CUDA version: $CUDA_VERSION"
    else
        print_error "CUDA not found! Install JetPack 5.1+"
        exit 1
    fi
}

# Update system packages
update_system() {
    print_header "Updating System Packages"
    
    print_status "Updating package lists..."
    sudo apt update
    
    print_status "Installing system dependencies..."
    sudo apt install -y \
        python3-pip \
        python3-dev \
        python3-opencv \
        libopencv-dev \
        cmake \
        build-essential \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libatlas-base-dev \
        gfortran \
        wget \
        curl \
        unzip
    
    print_status "System packages updated successfully"
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    python3 -m pip install --upgrade pip
    
    # Install PyTorch for Jetson
    print_status "Installing PyTorch for Jetson..."
    
    # Check if PyTorch is already installed
    if python3 -c "import torch" 2>/dev/null; then
        print_status "PyTorch already installed"
    else
        # Install PyTorch wheel for Jetson
        TORCH_URL="https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl"
        TORCH_FILE="torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
        
        print_status "Downloading PyTorch wheel..."
        wget -O $TORCH_FILE $TORCH_URL
        
        print_status "Installing PyTorch..."
        python3 -m pip install $TORCH_FILE
        
        rm $TORCH_FILE
    fi
    
    # Install torchvision
    print_status "Installing torchvision..."
    python3 -m pip install torchvision
    
    # Install other dependencies
    print_status "Installing other Python packages..."
    python3 -m pip install -r requirements_jetson.txt
    
    print_status "Python dependencies installed successfully"
}

# Optimize for Jetson
optimize_jetson() {
    print_header "Optimizing for Jetson Performance"
    
    # Set max performance mode
    print_status "Setting performance mode..."
    sudo nvpmodel -m 0  # Max performance
    sudo jetson_clocks
    
    # Increase swap if needed
    SWAP_SIZE=$(swapon --show --bytes | awk 'NR==2{print $3}' | numfmt --to=iec)
    if [ -z "$SWAP_SIZE" ] || [ "${SWAP_SIZE%G*}" -lt 4 ]; then
        print_status "Creating swap file for better memory management..."
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    
    # Set GPU memory fraction
    print_status "Configuring GPU memory..."
    echo 'export CUDA_DEVICE_MAX_CONNECTIONS=1' | sudo tee -a /etc/environment
    
    print_status "Jetson optimization completed"
}

# Install GLASS system
install_glass() {
    print_header "Installing GLASS System"
    
    # Create installation directory
    INSTALL_DIR="/opt/glass"
    print_status "Creating installation directory: $INSTALL_DIR"
    sudo mkdir -p $INSTALL_DIR
    
    # Copy files
    print_status "Copying GLASS files..."
    sudo cp -r . $INSTALL_DIR/
    sudo chown -R $USER:$USER $INSTALL_DIR
    
    # Make scripts executable
    chmod +x $INSTALL_DIR/*.py
    chmod +x $INSTALL_DIR/scripts/*.py
    
    # Create wrapper scripts with proper Python paths
    print_status "Creating command shortcuts..."
    
    # Create glass-detect wrapper
    cat > /tmp/glass-detect << EOF
#!/bin/bash
cd $INSTALL_DIR
export PYTHONPATH=$INSTALL_DIR/utils:$INSTALL_DIR/scripts:\$PYTHONPATH
python3 $INSTALL_DIR/run_fabric_detection_jetson.py "\$@"
EOF
    sudo mv /tmp/glass-detect /usr/local/bin/glass-detect
    sudo chmod +x /usr/local/bin/glass-detect
    
    # Create glass-verify wrapper
    cat > /tmp/glass-verify << EOF
#!/bin/bash
cd $INSTALL_DIR
export PYTHONPATH=$INSTALL_DIR/utils:$INSTALL_DIR/scripts:\$PYTHONPATH
python3 $INSTALL_DIR/verify_jetson.py "\$@"
EOF
    sudo mv /tmp/glass-verify /usr/local/bin/glass-verify
    sudo chmod +x /usr/local/bin/glass-verify
    
    # Create glass-benchmark wrapper
    cat > /tmp/glass-benchmark << EOF
#!/bin/bash
cd $INSTALL_DIR
export PYTHONPATH=$INSTALL_DIR/utils:$INSTALL_DIR/scripts:\$PYTHONPATH
python3 $INSTALL_DIR/benchmark_jetson.py "\$@"
EOF
    sudo mv /tmp/glass-benchmark /usr/local/bin/glass-benchmark
    sudo chmod +x /usr/local/bin/glass-benchmark
    
    # Create glass-diagnose wrapper for display issues
    cat > /tmp/glass-diagnose << EOF
#!/bin/bash
cd $INSTALL_DIR
python3 $INSTALL_DIR/diagnose_display.py "\$@"
EOF
    sudo mv /tmp/glass-diagnose /usr/local/bin/glass-diagnose
    sudo chmod +x /usr/local/bin/glass-diagnose
    
    # Create glass-detect-fixed wrapper with display fixes
    cat > /tmp/glass-detect-fixed << EOF
#!/bin/bash
cd $INSTALL_DIR
export PYTHONPATH=$INSTALL_DIR/utils:$INSTALL_DIR/scripts:\$PYTHONPATH
python3 $INSTALL_DIR/run_fabric_detection_jetson_fixed.py "\$@"
EOF
    sudo mv /tmp/glass-detect-fixed /usr/local/bin/glass-detect-fixed
    sudo chmod +x /usr/local/bin/glass-detect-fixed
    
    # Create glass-detect-complete wrapper with full PC functionality
    cat > /tmp/glass-detect-complete << EOF
#!/bin/bash
cd $INSTALL_DIR
export PYTHONPATH=$INSTALL_DIR/utils:$INSTALL_DIR/scripts:\$PYTHONPATH
python3 $INSTALL_DIR/run_fabric_detection_jetson_complete.py "\$@"
EOF
    sudo mv /tmp/glass-detect-complete /usr/local/bin/glass-detect-complete
    sudo chmod +x /usr/local/bin/glass-detect-complete
    
    print_status "GLASS system installed successfully"
}

# Setup systemd service (optional)
setup_service() {
    print_header "Setting up System Service (Optional)"
    
    read -p "Install GLASS as system service? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Creating systemd service..."
        
        cat > /tmp/glass-detection.service << EOF
[Unit]
Description=GLASS Fabric Defect Detection Service
After=multi-user.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/glass
ExecStart=/usr/bin/python3 /opt/glass/run_fabric_detection_jetson.py --service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        sudo mv /tmp/glass-detection.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable glass-detection.service
        
        print_status "Service installed. Control with:"
        print_status "  sudo systemctl start glass-detection"
        print_status "  sudo systemctl stop glass-detection"
        print_status "  sudo systemctl status glass-detection"
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    cd /opt/glass
    
    print_status "Running system verification..."
    if python3 verify_jetson.py; then
        print_status "✅ Installation verification successful!"
    else
        print_error "❌ Installation verification failed!"
        exit 1
    fi
}

# Main installation process
main() {
    print_header "Starting GLASS Installation on Jetson"
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        print_error "Run as regular user: ./install_jetson.sh"
        exit 1
    fi
    
    check_jetson
    check_requirements
    update_system
    install_python_deps
    optimize_jetson
    install_glass
    setup_service
    verify_installation
    
    print_header "Installation Complete!"
    print_status "🎉 GLASS Fabric Detection is ready on Jetson!"
    echo
    print_status "Quick commands:"
    print_status "  glass-verify          # System verification"
    print_status "  glass-benchmark       # Performance test"
    print_status "  glass-diagnose        # Display diagnostic"
    print_status "  glass-detect          # Run detection (basic)"
    print_status "  glass-detect-fixed    # Run with display fixes"
    print_status "  glass-detect-complete # Run with full PC functionality"
    print_status "  glass-detect-complete --headless # Run without display"
    echo
    print_status "Full commands:"
    print_status "  cd /opt/glass"
    print_status "  python3 run_fabric_detection_jetson.py"
    echo
    print_status "Documentation: /opt/glass/docs/"
    print_status "Logs: /opt/glass/logs/"
    echo
    print_status "🚀 Ready for fabric defect detection!"
}

# Run main function
main "$@"