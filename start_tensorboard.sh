#!/bin/bash

# TensorBoard Monitoring Script for AI Toolkit Training
# Created by Tsien at 2025-08-18
# Usage: ./start_tensorboard.sh [log_dir] [port]

set -e

# Default parameters
LOG_DIR=${1:-"output/.tensorboard"}
PORT=${2:-6006}

echo "🔍 Starting TensorBoard for AI Toolkit Training"
echo "📁 Log directory: $LOG_DIR"
echo "🌐 Port: $PORT"
echo "👤 Created by: Tsien at 2025-08-18"
echo

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "⚠️  Log directory $LOG_DIR does not exist yet"
    echo "📝 Creating directory..."
    mkdir -p "$LOG_DIR"
    echo "✅ Directory created. TensorBoard will populate it during training."
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check if tensorboard is installed
if ! command -v tensorboard &> /dev/null; then
    echo "❌ Error: tensorboard is not installed"
    echo "💿 Installing tensorboard..."
    pip install tensorboard
fi

echo
echo "🚀 Starting TensorBoard server..."
echo "🌐 Open your browser and go to: http://localhost:$PORT"
echo "🔄 Press Ctrl+C to stop TensorBoard"
echo

# Start TensorBoard
tensorboard --logdir="$LOG_DIR" --port=$PORT --host=0.0.0.0

echo
echo "👤 TensorBoard script created by Tsien at 2025-08-18"
