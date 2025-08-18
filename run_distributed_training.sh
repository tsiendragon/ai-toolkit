#!/bin/bash

# AI Toolkit Multi-GPU Distributed Training Script
# Created by Tsien at 2025-08-18
# Usage: ./run_distributed_training.sh [config_file] [num_gpus]

set -e

# Default parameters
CONFIG_FILE=${1:-"config/train_id_card_flux_kontext_distributed.yaml"}
NUM_GPUS=${2:-2}

echo "🚀 Starting AI Toolkit Multi-GPU Distributed Training"
echo "📁 Config file: $CONFIG_FILE"
echo "🎮 Number of GPUs: $NUM_GPUS"
echo "👤 Created by: Tsien at 2025-08-18"
echo

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file $CONFIG_FILE not found!"
    echo "📝 Please make sure the config file exists and has 'distributed_training: true'"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found"
fi

# Set environment variables for better performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_P2P_DISABLE=1  # Disable P2P in case of issues

# Create output directory
mkdir -p output

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "❌ Error: accelerate is not installed"
    echo "💿 Please run: pip install accelerate"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -L | head -$NUM_GPUS
    AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
    if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
        echo "⚠️  Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
        echo "🔧 Using $AVAILABLE_GPUS GPUs instead"
        NUM_GPUS=$AVAILABLE_GPUS
    fi
else
    echo "⚠️  nvidia-smi not found, proceeding anyway..."
fi

echo
echo "💡 This script uses the built-in Accelerate integration"
echo "🔧 Starting distributed training..."
echo "📋 Command: accelerate launch --multi_gpu --num_processes $NUM_GPUS run.py $CONFIG_FILE"
echo

# Start distributed training
accelerate launch \
    --multi_gpu \
    --num_processes "$NUM_GPUS" \
    --main_process_port 29500 \
    run.py "$CONFIG_FILE"

echo
echo "🎉 Distributed training completed!"
echo "📂 Check the output/ directory for results"
echo "👤 Script created by Tsien at 2025-08-18"
