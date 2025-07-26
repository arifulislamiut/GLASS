#!/bin/bash
# Get virtual environment path

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
    if [ -d "$path" ] && [ -f "$path/bin/activate" ]; then
        VENV_PATH="$path"
        break
    fi
done

if [ -n "$VENV_PATH" ]; then
    echo "$VENV_PATH"
    exit 0
else
    echo "No virtual environment found" >&2
    exit 1
fi