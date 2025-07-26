#!/bin/bash
# Install GLASS auto-start service for Jetson

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INSTALL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run with sudo: sudo ./install_auto_start.sh"
    exit 1
fi

print_status "Installing GLASS Auto-Start Service"
echo "=================================="

# Get the actual username (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME="/home/$ACTUAL_USER"

print_status "Installing for user: $ACTUAL_USER"
print_status "User home directory: $USER_HOME"

# Check if auto_start_detection.sh exists
if [ ! -f "auto_start_detection.sh" ]; then
    print_error "auto_start_detection.sh not found in current directory"
    exit 1
fi

# Make auto_start_detection.sh executable
chmod +x auto_start_detection.sh
print_status "Made auto_start_detection.sh executable"

# Update the service file with correct user and paths
SERVICE_FILE="glass-detection.service"
TEMP_SERVICE="/tmp/glass-detection.service"

# Create updated service file with correct user
cat > $TEMP_SERVICE << EOF
[Unit]
Description=GLASS Defect Detection Auto-Start Service
After=multi-user.target
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=forking
User=$ACTUAL_USER
Group=$ACTUAL_USER
WorkingDirectory=$USER_HOME
Environment="HOME=$USER_HOME"
Environment="USER=$ACTUAL_USER"
Environment="DISPLAY=:0"
Environment="XAUTHORITY=$USER_HOME/.Xauthority"
ExecStart=$USER_HOME/glass_jetson_deploy/auto_start_detection.sh
ExecStop=/bin/kill -TERM \$MAINPID
PIDFile=/var/run/glass_detection.pid
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# GPU and system access
SupplementaryGroups=video dialout

[Install]
WantedBy=multi-user.target
EOF

# Copy service file to systemd directory
cp $TEMP_SERVICE /etc/systemd/system/glass-detection.service
print_status "Service file installed to /etc/systemd/system/"

# Create log file with proper permissions
touch /var/log/glass_auto_start.log
chown $ACTUAL_USER:$ACTUAL_USER /var/log/glass_auto_start.log
print_status "Created log file: /var/log/glass_auto_start.log"

# Reload systemd and enable the service
systemctl daemon-reload
print_status "Reloaded systemd daemon"

systemctl enable glass-detection.service
print_status "Enabled glass-detection service"

print_status "✅ Auto-start service installed successfully!"
echo
echo "Service Commands:"
echo "  Start:   sudo systemctl start glass-detection"
echo "  Stop:    sudo systemctl stop glass-detection"
echo "  Status:  sudo systemctl status glass-detection"
echo "  Logs:    sudo journalctl -u glass-detection -f"
echo "  Disable: sudo systemctl disable glass-detection"
echo
echo "Log file: /var/log/glass_auto_start.log"
echo
print_warning "The service will start automatically on next boot"
print_warning "To start now: sudo systemctl start glass-detection"

# Clean up
rm $TEMP_SERVICE