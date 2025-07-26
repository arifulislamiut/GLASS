#!/bin/bash
# Auto-start GLASS detection system after boot
# This script activates the virtual environment and starts detection

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[AUTO-START]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Log file for debugging
LOG_FILE="/var/log/glass_auto_start.log"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

print_status "Starting GLASS Auto-Detection System"
log_message "Auto-start script initiated"

# Wait for system to fully boot (30 seconds delay)
print_status "Waiting for system initialization..."
log_message "Waiting 30 seconds for system boot completion"
sleep 30

# Check for virtual environment in common locations
VENV_PATHS=(
    "$HOME/venv_detectron2"
    "$HOME/.venv"
    "$HOME/venv"
    "$HOME/glass_env"
    "$HOME/miniconda3/envs/glass_env"
    "$HOME/anaconda3/envs/glass_env"
)

VENV_PATH=""
for path in "${VENV_PATHS[@]}"; do
    if [ -d "$path" ]; then
        VENV_PATH="$path"
        break
    fi
done

if [ -z "$VENV_PATH" ]; then
    print_warning "No virtual environment found in standard locations"
    log_message "WARNING: No virtual environment found, trying system Python"
    
    # Try without virtual environment
    VENV_PATH=""
else
    print_status "Found virtual environment at $VENV_PATH"
    log_message "Virtual environment found at $VENV_PATH"
fi

# Activate virtual environment if available
if [ -n "$VENV_PATH" ]; then
    print_status "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    log_message "Activated virtual environment at $VENV_PATH"
    
    # Check if detect-defect-complete command exists
    if command -v detect-defect-complete &> /dev/null; then
        print_status "Found detect-defect-complete command"
        log_message "detect-defect-complete command found"
        USE_COMMAND=true
    else
        print_warning "detect-defect-complete command not found"
        log_message "detect-defect-complete command not found"
        USE_COMMAND=false
    fi
else
    print_status "No virtual environment, using system Python"
    log_message "Using system Python"
    USE_COMMAND=false
fi

# Try to find the actual detection script
DETECTION_SCRIPT="$HOME/glass_jetson_deploy/run_fabric_detection_jetson_complete.py"
if [ ! -f "$DETECTION_SCRIPT" ]; then
    # Try current directory
    DETECTION_SCRIPT="./run_fabric_detection_jetson_complete.py"
    if [ ! -f "$DETECTION_SCRIPT" ]; then
        # Try script directory
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        DETECTION_SCRIPT="$SCRIPT_DIR/run_fabric_detection_jetson_complete.py"
        if [ ! -f "$DETECTION_SCRIPT" ]; then
            print_error "No detection script found!"
            log_message "ERROR: No detection script found"
            exit 1
        fi
    fi
fi

print_status "Using detection script: $DETECTION_SCRIPT"
log_message "Using detection script: $DETECTION_SCRIPT"

print_status "Starting detection system..."

# Start detection based on what's available
if [ "$USE_COMMAND" = true ]; then
    print_status "Starting with detect-defect-complete command..."
    log_message "Starting detect-defect-complete command"
    detect-defect-complete &
    DETECTION_PID=$!
else
    print_status "Starting with Python script..."
    log_message "Starting Python script: $DETECTION_SCRIPT"
    
    # Change to script directory
    SCRIPT_DIR="$(dirname "$DETECTION_SCRIPT")"
    cd "$SCRIPT_DIR"
    
    python3 "$(basename "$DETECTION_SCRIPT")" &
    DETECTION_PID=$!
fi

if [ $? -eq 0 ] && [ -n "$DETECTION_PID" ]; then
    print_status "Detection system started successfully with PID: $DETECTION_PID"
    log_message "Detection system started successfully with PID: $DETECTION_PID"
    
    # Save PID for later management
    echo $DETECTION_PID > /var/run/glass_detection.pid
    
    print_status "Auto-start completed. Detection running in background."
    log_message "Auto-start completed successfully"
else
    print_error "Failed to start detection system"
    log_message "ERROR: Failed to start detection system"
    exit 1
fi