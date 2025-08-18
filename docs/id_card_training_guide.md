# ID Card Dataset - FLUX Kontext LoRA Distributed Training Guide

**Created by Tsien at 2025-08-18**

This guide demonstrates how to perform multi-GPU distributed LoRA fine-tuning on an ID Card dataset using the FLUX Kontext Dev model.

## 📁 Data Structure

Ensure your data is organized according to the following structure:
```
/folder/
├── training_images/          # Target images (expected output after training)
│   ├── image1.jpg
│   ├── image1.txt           # Corresponding caption text
│   ├── image2.jpg
│   ├── image2.txt
│   └── ...
└── control_images/          # Control images (original input)
    ├── image1.jpg           # Filenames must match training_images
    ├── image2.jpg
    └── ...
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Ensure accelerate is installed
pip install accelerate
```

### 2. Configure Hugging Face Token
```bash
# Visit https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev to accept the license
# Get READ token: https://huggingface.co/settings/tokens/new
export HF_TOKEN=your_token_here
# Or create .env file
echo "HF_TOKEN=your_token_here" > .env
```

### 3. Launch Distributed Training
```bash
# Using 2 GPUs
./run_distributed_training.sh config/train_id_card_flux_kontext_distributed.yaml 2

# Using 4 GPUs
./run_distributed_training.sh config/train_id_card_flux_kontext_distributed.yaml 4

# Using default configuration (2 GPUs)
./run_distributed_training.sh
```

## ⚙️ Configuration Guide

### Key Configuration Options

#### Distributed Training (New Feature)
```yaml
train:
  distributed_training: true  # 🔥 Enable multi-GPU distributed training
  batch_size: 1              # Batch size per GPU
  gradient_accumulation_steps: 1  # Usually set to 1 for distributed training
```

#### Data Paths
```yaml
datasets:
  - folder_path: "/home/sysop/data/id_card/training_images"
    control_path: "/home/sysop/data/id_card/control_images"
```

#### Model Configuration
```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-Kontext-dev"
  arch: "flux_kontext"
  quantize: true
```

## 📊 Performance Optimization

### Batch Size Calculation
```
Effective batch size = batch_size × num_gpus × gradient_accumulation_steps
Example: 1 × 2 × 1 = 2 (when using 2 GPUs)
```

### Learning Rate Adjustment
You may need to adjust the learning rate for multi-GPU training:
```yaml
train:
  lr: 1e-4  # May need adjustment based on number of GPUs
```

### Memory Optimization
```yaml
model:
  quantize: true     # Enable 8-bit quantization
  low_vram: true     # If memory is insufficient
train:
  gradient_checkpointing: true  # Save VRAM
```

## 🛠️ Troubleshooting

### Common Issues

1. **NCCL Errors**
   ```bash
   export NCCL_P2P_DISABLE=1
   export NCCL_IB_DISABLE=1
   ```

2. **Port Conflicts**
   ```bash
   # Modify the port in the script
   --main_process_port 29501
   ```

3. **Out of Memory**
   - Reduce `batch_size`
   - Enable `quantize: true`
   - Use lower resolution

### Training Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training processes
ps aux | grep python
```

## 📝 Code Modifications

This implementation makes minimal changes to the original project:

### 1. Configuration Option (toolkit/config_modules.py)
```python
# Added distributed training configuration - by Tsien at 2025-08-18
self.distributed_training: bool = kwargs.get('distributed_training', False)
```

### 2. DataLoader Preparation (jobs/process/BaseSDTrainProcess.py)
```python
# Distributed training support - by Tsien at 2025-08-18
if getattr(self.train_config, 'distributed_training', False):
    if hasattr(self, 'data_loader') and self.data_loader is not None:
        self.data_loader = self.accelerator.prepare(self.data_loader)
    if hasattr(self, 'data_loader_reg') and self.data_loader_reg is not None:
        self.data_loader_reg = self.accelerator.prepare(self.data_loader_reg)
```

## 🎯 Expected Results

After training completion, you will obtain:
- LoRA weight files (.safetensors)
- Training sample images
- Training logs and metrics
- Model weights ready for inference

## 📞 Support

If you encounter issues, please check:
1. Are the data paths correct?
2. Is the HF_TOKEN valid?
3. Is there sufficient GPU memory?
4. Is the configuration file format correct?

## 🔧 Implementation Details

### Architecture Overview
- **FLUX Kontext Dev**: Advanced conditional image generation model
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning approach
- **Multi-GPU Support**: Distributed training via Hugging Face Accelerate
- **Configuration-driven**: Enable/disable distributed training via config

### Key Features
- **Minimal Code Changes**: Only adds functionality, preserves existing code
- **Backward Compatible**: Single GPU training remains unchanged
- **Conditional Preparation**: DataLoaders only prepared when distributed training is enabled
- **Automatic Device Assignment**: Accelerate handles GPU allocation automatically

### Training Process
1. **Data Loading**: Paired images loaded from training and control paths
2. **Model Preparation**: FLUX Kontext model prepared with LoRA adapters
3. **Distributed Setup**: If enabled, dataloaders prepared for multi-GPU
4. **Training Loop**: Standard diffusion training with distributed gradient sync
5. **Sampling**: Periodic validation samples generated during training

**Created by Tsien at 2025-08-18**
