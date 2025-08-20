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

# Fix distributed training logging issues - by Tsien at 2025-08-19
export WANDB_DISABLED=true          # Disable wandb to avoid console capture conflicts
export WANDB_MODE=disabled          # Alternative way to disable wandb
export PYTHONUNBUFFERED=1           # Ensure proper stdout flushing

# NCCL Configuration for AWS environment - by Tsien at 2025-08-18
export NCCL_P2P_DISABLE=1           # Disable P2P communication
export NCCL_IB_DISABLE=1            # Disable InfiniBand
export NCCL_TIMEOUT=1800            # Increase timeout to 30 minutes
export TORCH_NCCL_BLOCKING_WAIT=1         # Enable blocking wait
export NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling
export NCCL_DEBUG=WARN              # Production setting - use INFO for debugging

# Fix GPU device mapping warnings - by Tsien at 2025-08-19
# Note: CUDA_VISIBLE_DEVICES will be set per-process by run.py based on LOCAL_RANK
# This ensures each process only sees its own GPU, fixing the mapping warning

# Auto-detect correct network interface - by Tsien at 2025-08-19
NETWORK_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -n1)
if [ -z "$NETWORK_INTERFACE" ]; then
    NETWORK_INTERFACE="ens32"  # Fallback to detected interface
fi
echo "🌐 Using network interface: $NETWORK_INTERFACE"

export NCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"  # Use detected interface
export NCCL_NET_PLUGIN=none         # Disable OFI plugin to avoid warnings
export NCCL_IB_DISABLE=1            # Ensure InfiniBand is disabled
export NCCL_P2P_DISABLE=1           # Ensure P2P is disabled

# AWS specific optimizations
export NCCL_TREE_THRESHOLD=0        # Force ring algorithm for AWS
export NCCL_NET="Socket"            # Use socket communication

# Additional NCCL fallback configurations - by Tsien at 2025-08-19
export NCCL_SOCKET_NTHREADS=1       # Reduce socket threads for stability
export NCCL_NSOCKS_PERTHREAD=1      # One socket per thread
export NCCL_BUFFSIZE=2097152        # 2MB buffer size
export NCCL_NTHREADS=16             # Thread count
export NCCL_LL_THRESHOLD=16384      # Low-latency threshold
export MASTER_ADDR="127.0.0.1"      # Explicit master address for single-node
export MASTER_PORT="29500"          # Explicit master port

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

# Start distributed training with explicit Accelerate config - by Tsien at 2025-08-19
# Create temporary accelerate config for current GPU count
TEMP_CONFIG="/tmp/accelerate_config_${NUM_GPUS}gpu.yaml"
cat > "$TEMP_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: '$(seq -s, 0 $((NUM_GPUS-1)))'
machine_rank: 0
main_process_ip: null
main_process_port: 29500
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "📋 Generated Accelerate config for $NUM_GPUS GPUs:"
echo "   Config file: $TEMP_CONFIG"
echo "   GPU IDs: $(seq -s, 0 $((NUM_GPUS-1)))"
echo

# Launch with explicit config file
accelerate launch \
    --config_file "$TEMP_CONFIG" \
    run.py "$CONFIG_FILE"

# Cleanup temp config
rm -f "$TEMP_CONFIG"

echo
echo "🎉 Distributed training completed!"
echo "📂 Check the output/ directory for results"
echo "👤 Script created by Tsien at 2025-08-18"
