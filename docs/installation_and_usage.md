# AI Toolkit Installation and Usage Guide

## System Requirements

- Python 3.10 or higher
- NVIDIA GPU (with sufficient VRAM for your tasks)
- Git
- Node.js 18 or higher (for Web UI)
- At least 24GB VRAM (for FLUX.1 training)

## 1. Install System Dependencies

### Install Node.js and npm

#### Ubuntu/Linux:
```bash
# Install latest Node.js using NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

#### Using NVM (Recommended):
```bash
# Install NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash

# Reload terminal or run
source ~/.bashrc

# Install and use Node.js 20
nvm install 20
nvm use 20
```

#### Windows:
1. Visit [Node.js official website](https://nodejs.org/)
2. Download and install the LTS version
3. Make sure to check "Add to PATH" during installation

### Install Git (if not already installed):
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install git

# CentOS/RHEL
sudo yum install git
```

### Install Python virtual environment tools:
```bash
sudo apt install python3-venv
```

## 2. Clone and Install AI Toolkit

### Linux Installation Steps:
```bash
# Clone the project
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch first
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip3 install -r requirements.txt
```

### Windows Installation Steps:
```bash
# Clone the project
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch first
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install -r requirements.txt
```

## 3. Configure Hugging Face Access (for FLUX.1-dev and FLUX.1-Kontext)

If you want to use FLUX.1-dev or FLUX.1-Kontext models:

1. Login to [Hugging Face](https://huggingface.co/)
2. Accept the license for the models:
   - [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
   - [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
3. Get a READ access [token](https://huggingface.co/settings/tokens/new?)
4. Create a `.env` file in the project root:
```bash
echo "HF_TOKEN=your_token_here" > .env
```

## 4. Install and Run Web UI

### Install UI Dependencies:
```bash
cd ui
npm install
```

### Start the UI:
```bash
# Build and start the UI
npm run build_and_start
```

The UI will start at `http://localhost:8675`.

### Security Configuration (Optional):
If you're running on a cloud server, it's recommended to set an access token:

```bash
# Linux
AI_TOOLKIT_AUTH=your_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=your_secure_password && npm run build_and_start

# Windows PowerShell
$env:AI_TOOLKIT_AUTH="your_secure_password"; npm run build_and_start
```

## 5. Dataset Preparation

### Standard Dataset Format Requirements:

1. **Supported Image Formats**: JPG, JPEG, PNG
2. **File Structure**:
   ```
   your_dataset/
   ├── image1.jpg
   ├── image1.txt
   ├── image2.png
   ├── image2.txt
   └── ...
   ```

3. **Text Files**: Each image should have a corresponding `.txt` file containing the image description

### FLUX Kontext Dataset Format (Image Editing):

FLUX Kontext requires a **paired dataset structure** with control images for image editing tasks:

1. **File Structure**:
   ```
   your_project/
   ├── training_images/          # Target/output images
   │   ├── person1.jpg
   │   ├── person1.txt           # Caption: "A person with an afro"
   │   ├── person2.jpg
   │   ├── person2.txt           # Caption: "A smiling person"
   │   └── ...
   └── control_images/           # Source/input images
       ├── person1.jpg           # Original image before editing
       ├── person2.jpg           # Original image before editing
       └── ...                   # NO caption files needed here
   ```

2. **Key Requirements for FLUX Kontext**:
   - **Control images** must have **matching filenames** with training images
   - Control images are the **source/original** images you want to edit
   - Training images are the **target/edited** versions
   - **Only training images need captions** (control images don't)
   - Recommended resolutions: 512x512 or 768x768 (1024 may cause OOM with 24GB VRAM)

### Dataset Preparation Steps:

#### For Standard Training:

1. **Create dataset folder**:
   ```bash
   mkdir datasets/my_training_data
   ```

2. **Add images and descriptions**:
   - Copy your training images to the folder
   - Create same-named `.txt` files for each image
   - Write image descriptions in the text files

3. **Using trigger words**:
   - Use `[trigger]` placeholder in descriptions
   - The system will automatically replace it with your configured trigger word

#### For FLUX Kontext Training:

1. **Create paired dataset structure**:
   ```bash
   mkdir -p datasets/kontext_editing/{training_images,control_images}
   ```

2. **Organize your images**:
   - Place **edited/target** images in `training_images/`
   - Place **original/source** images in `control_images/`
   - Ensure **matching filenames** between folders

3. **Create captions** (only for training images):
   - Describe what the edited image shows
   - Focus on the differences from the original
   - Example: "A person with blue hair" (if you changed hair color)

### Dataset Examples:

#### Standard Dataset:
```
datasets/cat_photos/
├── cat1.jpg          # An orange cat sitting on a windowsill
├── cat1.txt          # Content: "A [trigger] sitting on a windowsill"
├── cat2.png          # A black cat playing in grass
├── cat2.txt          # Content: "A [trigger] playing in the grass"
```

#### FLUX Kontext Dataset:
```
datasets/portrait_editing/
├── training_images/
│   ├── person1.jpg   # Person with smile
│   ├── person1.txt   # "A smiling person"
│   ├── person2.jpg   # Person with afro hairstyle
│   ├── person2.txt   # "A person with an afro"
└── control_images/
    ├── person1.jpg   # Same person without smile
    ├── person2.jpg   # Same person with original hair
```

## 6. Model Training

### Using Simple Gradio UI:

1. **Start Gradio Interface**:
   ```bash
   # Make sure you're in the virtual environment
   source venv/bin/activate  # Linux
   # or .\venv\Scripts\activate  # Windows

   # Login to Hugging Face (for publishing models)
   huggingface-cli login

   # Start the UI
   python flux_train_ui.py
   ```

2. **Using the Interface**:
   - Upload your training images
   - Set trigger words
   - Configure training parameters
   - Start training

### Using Configuration Files:

#### For Standard FLUX Training:

1. **Copy example configuration**:
   ```bash
   cp config/examples/train_lora_flux_24gb.yaml config/my_training.yaml
   ```

2. **Edit configuration file**:
   ```yaml
   datasets:
     - folder_path: "datasets/my_training_data"  # Your dataset path
       caption_ext: "txt"
       caption_dropout_rate: 0.05
       shuffle_tokens: false
       cache_latents_to_disk: true
       resolution: [512, 768, 1024]
   ```

3. **Start training**:
   ```bash
   python run.py config/my_training.yaml
   ```

#### For FLUX Kontext Training:

1. **Copy Kontext example configuration**:
   ```bash
   cp config/examples/train_lora_flux_kontext_24gb.yaml config/my_kontext_training.yaml
   ```

2. **Edit configuration for Kontext**:
   ```yaml
   datasets:
     - folder_path: "datasets/kontext_editing/training_images"  # Target images
       control_path: "datasets/kontext_editing/control_images"   # Source images
       caption_ext: "txt"
       caption_dropout_rate: 0.05
       shuffle_tokens: false
       cache_latents_to_disk: true
       resolution: [512, 768]  # Lower resolution recommended for 24GB VRAM

   model:
     name_or_path: "black-forest-labs/FLUX.1-Kontext-dev"
     arch: "flux_kontext"
     quantize: true

   sample:
     prompts:
       # Use --ctrl_img to specify control image for sampling
       - "make the person smile --ctrl_img datasets/kontext_editing/control_images/person1.jpg"
       - "give the person an afro --ctrl_img datasets/kontext_editing/control_images/person2.jpg"
   ```

3. **Start Kontext training**:
   ```bash
   python run.py config/my_kontext_training.yaml
   ```

## 7. Running Web UI (Advanced Features)

### Start the Full Web UI:

```bash
cd ui
npm run build_and_start
```

### Web UI Features:
- Create and manage training jobs
- Monitor training progress
- Manage datasets
- Generate sample images
- View training history

### Access the UI:
- Local access: `http://localhost:8675`
- Remote access: `http://your-server-ip:8675`

## 8. Troubleshooting

### Common Issues:

1. **Out of Memory Errors**:
   - Set `low_vram: true` in config file
   - Reduce `batch_size`
   - Use smaller resolutions

2. **PyTorch Installation Problems**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
   ```

3. **NPM Installation Failures**:
   ```bash
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **CUDA Version Mismatch**:
   - Check your CUDA version: `nvidia-smi`
   - Install the corresponding PyTorch version

5. **FLUX Kontext Specific Issues**:
   - Ensure control images have matching filenames with training images
   - Use lower resolutions (512x512 or 768x768) to avoid OOM
   - Make sure you've accepted the license for FLUX.1-Kontext-dev

### Performance Optimization:

1. **Use Quantization**: Set `quantize: true` in config
2. **Gradient Checkpointing**: Set `gradient_checkpointing: true`
3. **Mixed Precision**: Use `dtype: bf16`

## 9. Updating the Project

Regularly update the project to get the latest features:

```bash
cd ai-toolkit
git pull origin main

# Update Python dependencies
source venv/bin/activate
pip install -r requirements.txt

# Update UI dependencies
cd ui
npm install
```

## 10. Getting Help

- **Discord Community**: [Join Official Discord](https://discord.gg/VXmU2f5WEU)
- **GitHub Issues**: [Report Bugs](https://github.com/ostris/ai-toolkit/issues)
- **Documentation**: Check `README.md` and `FAQ.md`

---

This guide covers the basic installation and usage of AI Toolkit, including support for both standard FLUX training and FLUX Kontext image editing. For more advanced features and configuration options, please refer to the project's official documentation.
