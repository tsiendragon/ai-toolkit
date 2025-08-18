# TensorBoard Monitoring Guide for AI Toolkit

**Created by Tsien at 2025-08-18**

This guide explains how to use TensorBoard to monitor your FLUX Kontext LoRA training in real-time.

## 🎯 Overview

TensorBoard provides real-time visualization of:
- Training loss curves
- Learning rate schedules
- Training metrics
- Sample generation progress

## 🚀 Quick Start

### 1. Enable TensorBoard in Config

The configuration is already enabled in `config/train_id_card_flux_kontext_distributed.yaml`:

```yaml
process:
  - type: 'sd_trainer'
    # TensorBoard logging configuration
    log_dir: "output/.tensorboard"

    # logging configuration for TensorBoard and other loggers
    logging:
      log_every: 50  # log metrics every N steps
      project_name: "id_card_flux_kontext_training"
      run_name: "distributed_lora_v1"
```

### 2. Start TensorBoard Server

#### Option A: Using the provided script (Recommended)
```bash
# Start TensorBoard with default settings
./start_tensorboard.sh

# Or specify custom log directory and port
./start_tensorboard.sh output/.tensorboard 6006
```

#### Option B: Manual start
```bash
# Activate virtual environment
source venv/bin/activate

# Install tensorboard if not already installed
pip install tensorboard

# Start TensorBoard
tensorboard --logdir=output/.tensorboard --port=6006
```

### 3. Access TensorBoard

Open your web browser and go to:
- **Local access**: `http://localhost:6006`
- **Remote access**: `http://your-server-ip:6006`

## 📊 Monitoring Your Training

### Key Metrics to Watch

1. **Loss Curves**
   - `loss/loss` - Main training loss
   - `loss/sl_l` - Slider loss (if applicable)
   - `loss/an_l` - Anchor loss (if applicable)

2. **Learning Rate**
   - `lr` - Current learning rate
   - Monitor learning rate schedules

3. **Training Progress**
   - Step progression over time
   - Training speed and stability

### What to Look For

#### ✅ Good Training Signs
- **Steadily decreasing loss**: Loss should generally trend downward
- **Stable learning rate**: LR should follow expected schedule
- **Consistent logging**: Regular metric updates every 50 steps

#### ⚠️ Warning Signs
- **Loss plateauing too early**: May need learning rate adjustment
- **Loss spiking**: Possible instability, consider lowering LR
- **No metric updates**: Check if logging is properly configured

## 🔧 Configuration Options

### Logging Frequency
```yaml
logging:
  log_every: 50  # Log every 50 steps (adjust as needed)
  # log_every: 100  # Less frequent logging (faster training)
  # log_every: 25   # More frequent logging (detailed monitoring)
```

### Project Organization
```yaml
logging:
  project_name: "id_card_flux_kontext_training"  # Project identifier
  run_name: "distributed_lora_v1"                # Experiment name
```

### Log Directory
```yaml
log_dir: "output/.tensorboard"  # Where TensorBoard logs are saved
```

## 🔄 Multi-Experiment Comparison

TensorBoard automatically organizes runs by timestamp. To compare different experiments:

1. **Keep the same log_dir** for related experiments
2. **Change run_name** for each experiment:
   ```yaml
   logging:
     run_name: "experiment_lr_1e4"     # First experiment
     # run_name: "experiment_lr_2e4"   # Second experiment
     # run_name: "experiment_bs_2"     # Third experiment
   ```

3. **Use TensorBoard's comparison features** to analyze results

## 📱 Remote Monitoring

### SSH Tunneling (Recommended for remote servers)
```bash
# On your local machine, create SSH tunnel
ssh -L 6006:localhost:6006 user@remote-server

# Then access http://localhost:6006 on your local browser
```

### Direct Access (if firewall allows)
```bash
# Start TensorBoard with external access
tensorboard --logdir=output/.tensorboard --port=6006 --host=0.0.0.0

# Access via http://your-server-ip:6006
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No dashboards are active"**
   - Wait for training to start and generate logs
   - Check if `log_dir` path is correct
   - Verify logging is enabled in config

2. **Port already in use**
   ```bash
   # Use different port
   ./start_tensorboard.sh output/.tensorboard 6007
   ```

3. **Empty logs directory**
   - Training hasn't started yet
   - Check if `log_every` is set correctly
   - Verify config file syntax

4. **Permission denied**
   ```bash
   # Fix script permissions
   chmod +x start_tensorboard.sh
   ```

### Debugging Steps

1. **Check log directory exists**:
   ```bash
   ls -la output/.tensorboard/
   ```

2. **Verify TensorBoard installation**:
   ```bash
   source venv/bin/activate
   pip install tensorboard
   ```

3. **Test with sample data**:
   ```bash
   # TensorBoard should show existing runs
   tensorboard --logdir=output/.tensorboard
   ```

## 💡 Tips for Better Monitoring

### 1. Optimal Logging Frequency
- **Fast training**: `log_every: 100` (less overhead)
- **Detailed monitoring**: `log_every: 25` (more data points)
- **Balanced**: `log_every: 50` (recommended)

### 2. Organize Experiments
```yaml
# Use descriptive run names
run_name: "flux_kontext_lr1e4_bs1_gpu2"  # Include key parameters
```

### 3. Monitor Resource Usage
- Use `nvidia-smi` alongside TensorBoard
- Watch for memory leaks or GPU utilization drops

### 4. Save Important Runs
TensorBoard logs are automatically saved and persist between sessions.

## 🎯 Expected Training Pattern

For FLUX Kontext LoRA training, expect:

1. **Initial phase** (0-500 steps): Loss decreases rapidly
2. **Stabilization** (500-1500 steps): Loss reduction slows
3. **Fine-tuning** (1500-3000 steps): Gradual improvement
4. **Convergence** (2500+ steps): Loss plateaus at optimal level

## 📞 Support

If TensorBoard isn't working:
1. Check the configuration syntax
2. Verify all file paths exist
3. Ensure proper permissions
4. Check firewall settings for remote access

**Created by Tsien at 2025-08-18**
