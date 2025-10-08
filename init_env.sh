#!/bin/bash
# This script recreates the smolvla-310 conda environment and installs the required packages.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Deactivating and removing the old 'smolvla-310' environment ---"
conda deactivate
conda env remove -n smolvla-310 -y

echo "--- Creating the new 'smolvla-310' environment with Python 3.10 ---"
conda create -n smolvla-310 python=3.10 -y

echo "--- Activating the new environment ---"
# Note: conda activate might not work directly in a script.
# This command is for the user to run after the script, but we'll try to source it.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate smolvla-310

echo "--- Installing PyTorch with CUDA support ---"
pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "--- Installing lerobot with smolvla extras ---"
pip install -e "/home/adelli/lerobot[smolvla]"

echo "--- Installing RLBench ---"
pip install -e .

echo "--- Environment setup is complete! ---"
echo "Please run 'conda activate smolvla-310' to activate the new environment."
