#!/bin/bash
# Manage GLASS detection service

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[SERVICE]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

show_usage() {
    echo "GLASS Detection Service Manager"
    echo "==============================="
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start     - Start the detection service"
    echo "  stop      - Stop the detection service" 
    echo "  restart   - Restart the detection service"
    echo "  status    - Show service status"
    echo "  logs      - Show service logs (real-time)"
    echo "  enable    - Enable auto-start on boot"
    echo "  disable   - Disable auto-start on boot"
    echo "  install   - Install the auto-start service"
    echo "  uninstall - Remove the auto-start service"
    echo
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 logs"
    echo "  sudo $0 install"
}

check_service_exists() {
    if ! systemctl list-unit-files | grep -q "glass-detection.service"; then
        print_error "GLASS detection service not installed"
        print_info "Run: sudo $0 install"
        return 1
    fi
    return 0
}

case "$1" in
    start)
        print_status "Starting GLASS detection service..."
        if check_service_exists; then
            sudo systemctl start glass-detection
            if [ $? -eq 0 ]; then
                print_status "✅ Service started successfully"
                sleep 2
                systemctl status glass-detection --no-pager -l
            else
                print_error "Failed to start service"
                exit 1
            fi
        fi
        ;;
        
    stop)
        print_status "Stopping GLASS detection service..."
        if check_service_exists; then
            sudo systemctl stop glass-detection
            if [ $? -eq 0 ]; then
                print_status "✅ Service stopped successfully"
            else
                print_error "Failed to stop service"
                exit 1
            fi
        fi
        ;;
        
    restart)
        print_status "Restarting GLASS detection service..."
        if check_service_exists; then
            sudo systemctl restart glass-detection
            if [ $? -eq 0 ]; then
                print_status "✅ Service restarted successfully"
                sleep 2
                systemctl status glass-detection --no-pager -l
            else
                print_error "Failed to restart service"
                exit 1
            fi
        fi
        ;;
        
    status)
        if check_service_exists; then
            print_status "GLASS detection service status:"
            systemctl status glass-detection --no-pager -l
            echo
            
            # Check if process is actually running
            if [ -f /var/run/glass_detection.pid ]; then
                PID=$(cat /var/run/glass_detection.pid)
                if ps -p $PID > /dev/null 2>&1; then
                    print_status "✅ Detection process running (PID: $PID)"
                else
                    print_warning "⚠️  PID file exists but process not running"
                fi
            else
                print_info "No PID file found"
            fi
        fi
        ;;
        
    logs)
        print_status "Showing GLASS detection service logs (Ctrl+C to exit):"
        if check_service_exists; then
            sudo journalctl -u glass-detection -f
        fi
        ;;
        
    enable)
        print_status "Enabling auto-start on boot..."
        if check_service_exists; then
            sudo systemctl enable glass-detection
            if [ $? -eq 0 ]; then
                print_status "✅ Auto-start enabled"
            else
                print_error "Failed to enable auto-start"
                exit 1
            fi
        fi
        ;;
        
    disable)
        print_status "Disabling auto-start on boot..."
        if check_service_exists; then
            sudo systemctl disable glass-detection
            if [ $? -eq 0 ]; then
                print_status "✅ Auto-start disabled"
            else
                print_error "Failed to disable auto-start"
                exit 1
            fi
        fi
        ;;
        
    install)
        if [ "$EUID" -ne 0 ]; then
            print_error "Installation requires sudo: sudo $0 install"
            exit 1
        fi
        
        print_status "Installing GLASS auto-start service..."
        ./install_auto_start.sh
        ;;
        
    uninstall)
        if [ "$EUID" -ne 0 ]; then
            print_error "Uninstallation requires sudo: sudo $0 uninstall"
            exit 1
        fi
        
        print_status "Removing GLASS auto-start service..."
        
        # Stop and disable service
        systemctl stop glass-detection 2>/dev/null
        systemctl disable glass-detection 2>/dev/null
        
        # Remove service file
        rm -f /etc/systemd/system/glass-detection.service
        
        # Remove PID file
        rm -f /var/run/glass_detection.pid
        
        # Reload systemd
        systemctl daemon-reload
        
        print_status "✅ Service uninstalled successfully"
        ;;
        
    *)
        show_usage
        exit 1
        ;;
esac