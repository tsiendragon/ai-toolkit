#!/bin/bash

# FLUX Kontext LoRA Training Script for Annimate Selfie Dataset
# Usage: ./train.sh

set -e

CONFIG_FILE="config/train_annimate_selfie_lora.yaml"

echo "Starting FLUX Kontext LoRA training..."


# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: No virtual environment found"
fi

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create output directory
mkdir -p output

# Start training
echo "Running: python run.py $CONFIG_FILE"
python run.py "$CONFIG_FILE"

echo "Training completed! Check output/ directory for results."
