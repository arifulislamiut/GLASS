# GLASS Auto-Start System for Jetson

This system automatically starts the GLASS defect detection after system boot.

## Files Included

- **`auto_start_detection.sh`** - Main auto-start script
- **`glass-detection.service`** - Systemd service definition
- **`install_auto_start.sh`** - Service installation script
- **`manage_detection_service.sh`** - Service management utility

## Installation

1. **Install the auto-start service**:
   ```bash
   sudo ./install_auto_start.sh
   ```

2. **Start the service immediately** (optional):
   ```bash
   sudo systemctl start glass-detection
   ```

## Service Management

Use the management script for easy service control:

```bash
# Check service status
./manage_detection_service.sh status

# Start service
./manage_detection_service.sh start

# Stop service
./manage_detection_service.sh stop

# Restart service
./manage_detection_service.sh restart

# View live logs
./manage_detection_service.sh logs

# Enable auto-start on boot
./manage_detection_service.sh enable

# Disable auto-start on boot
./manage_detection_service.sh disable
```

## Manual Service Commands

```bash
# Start service
sudo systemctl start glass-detection

# Stop service
sudo systemctl stop glass-detection

# Check status
sudo systemctl status glass-detection

# View logs
sudo journalctl -u glass-detection -f

# Enable auto-start
sudo systemctl enable glass-detection

# Disable auto-start
sudo systemctl disable glass-detection
```

## How It Works

1. **Boot Sequence**: Service starts after graphical session is ready
2. **Environment Setup**: Activates `~/venv_detectron2` virtual environment
3. **Command Execution**: Runs `detect-defect-complete` command
4. **Fallback**: If command not found, uses `run_fabric_detection_jetson_complete.py`
5. **Logging**: All activities logged to `/var/log/glass_auto_start.log`

## Configuration

The service automatically detects:
- Virtual environment paths: `~/venv_detectron2`, `~/glass_env`, `~/.venv`, `~/venv`
- Detection script: `detect-defect-complete` or fallback script
- User environment and permissions

### Virtual Environment Detection
The auto-start script checks for virtual environments in this order:
1. `~/venv_detectron2` (original expected path)
2. `~/glass_env` (created by setup_environment.sh)
3. `~/.venv` (standard Python venv location)
4. `~/venv` (common alternative)
5. `~/miniconda3/envs/glass_env` (conda environment)
6. System Python (fallback)

## Troubleshooting

### Check Service Status
```bash
./manage_detection_service.sh status
```

### View Logs
```bash
# Service logs
./manage_detection_service.sh logs

# Boot log
tail /var/log/glass_auto_start.log
```

### Common Issues

1. **Virtual environment not found**:
   - Ensure one exists in the checked paths
   - Service will fall back to system Python if none found
   - Manually run `./setup_environment.sh` if needed

2. **detect-defect-complete not found**:
   - Service will automatically use fallback script
   - Ensure `run_fabric_detection_jetson_complete.py` exists

3. **Display issues**:
   - Service sets `DISPLAY=:0` and `XAUTHORITY`
   - Ensure X11 is running

4. **Permission issues**:
   - Service runs as the installing user
   - Added to `video` and `dialout` groups for GPU/camera access

### Restart After Changes
```bash
sudo systemctl daemon-reload
./manage_detection_service.sh restart
```

## Uninstallation

```bash
sudo ./manage_detection_service.sh uninstall
```

## Boot Behavior

After installation, the detection system will:
1. Wait 30 seconds for system boot completion
2. Activate the virtual environment
3. Start the detection system automatically
4. Run in the background with auto-restart on failure

The system is designed to be robust and handle various deployment scenarios automatically.