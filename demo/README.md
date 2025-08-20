# FLUX Kontext LoRA Demo

一个基于FastAPI的FLUX Kontext Dev模型LoRA checkpoint加载和推理演示应用。

## ✨ 功能特性

- 🔄 **动态模型管理**: 支持实时加载和卸载FLUX Kontext模型和LoRA checkpoint
- 📊 **状态监控**: 实时显示模型状态、GPU内存使用情况
- 🖼️ **批量推理**: 从指定文件夹批量处理图像，支持随机采样
- 🎨 **单个推理**: 上传图像和文本提示进行单个推理
- 🌐 **现代Web界面**: 响应式设计，支持桌面和移动设备
- 📝 **实时日志**: 查看详细的操作日志和错误信息

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA兼容的GPU（推荐24GB+显存）
- 足够的磁盘空间用于模型存储

### 安装和启动

1. **克隆项目**（如果还没有）
   ```bash
   git clone <your-ai-toolkit-repo>
   cd ai-toolkit/demo
   ```

2. **使用Python启动脚本（推荐）**
   ```bash
   python3 start_demo.py
   ```

3. **或使用Shell脚本（Linux/macOS）**
   ```bash
   chmod +x start_demo.sh
   ./start_demo.sh
   ```

4. **手动启动**
   ```bash
   # 安装依赖
   pip install -r requirements.txt

   # 启动应用
   python3 app.py
   ```

5. **访问应用**
   ```
   打开浏览器访问: http://localhost:8000
   ```

## 📋 使用说明

### 1. 模型管理

#### 加载模型
1. 在"模型控制"面板中输入LoRA checkpoint路径
2. 默认路径：`/data/lilong/experiment/id_card_flux_kontext_lora_v1/id_card_flux_kontext_lora_v1/id_card_flux_kontext_lora_v1_000001111.safetensors`
3. 点击"加载模型"按钮
4. 等待加载完成，状态会更新为"已加载"

#### 卸载模型
1. 点击"卸载模型"按钮
2. 确认操作
3. 模型将从GPU内存中卸载

#### 状态监控
- **加载状态**: 显示模型是否已加载
- **当前LoRA**: 显示当前加载的LoRA文件名
- **设备**: 显示使用的计算设备
- **显存使用**: 显示GPU内存占用情况

### 2. 批量推理

1. **设置参数**:
   - **图像文件夹路径**: 包含图像和对应txt文件的文件夹
   - **最大采样数量**: 从文件夹中随机选择的图像数量
   - **随机种子**: 控制随机性的种子值
   - **引导强度**: 控制生成图像与提示的匹配度
   - **推理步数**: 生成质量和速度的平衡

2. **执行推理**:
   - 点击"开始批量推理"
   - 等待处理完成
   - 查看生成结果

### 3. 单个推理

1. **输入参数**:
   - **文本提示**: 描述要生成的图像
   - **控制图像**: 可选，上传作为控制图像的文件
   - **尺寸设置**: 设置生成图像的宽度和高度
   - **推理参数**: 设置种子、引导强度、推理步数

2. **执行推理**:
   - 点击"开始单个推理"
   - 等待生成完成
   - 查看结果图像和参数详情

## 🔧 配置说明

### 默认配置

```python
# 默认LoRA路径
DEFAULT_LORA_PATH = "/data/lilong/experiment/id_card_flux_kontext_lora_v1/id_card_flux_kontext_lora_v1/id_card_flux_kontext_lora_v1_000001111.safetensors"

# 默认图像文件夹
DEFAULT_IMAGE_FOLDER = "/home/sysop/data/id_card/training_images"

# 样本输出目录
SAMPLES_OUTPUT_DIR = "/data/lilong/flux_kyc/samples"

# 默认推理参数
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 576
```

### 环境变量

- `PYTHONPATH`: 自动设置为项目根目录
- `TOKENIZERS_PARALLELISM`: 设置为false避免警告

## 📁 项目结构

```
demo/
├── app.py                 # FastAPI主应用
├── index.html            # 主页面
├── requirements.txt      # Python依赖
├── start_demo.py        # Python启动脚本
├── start_demo.sh        # Shell启动脚本
├── README.md            # 说明文档
└── static/
    ├── css/
    │   └── style.css     # 样式文件
    ├── js/
    │   └── app.js        # JavaScript逻辑
    ├── outputs/          # 本地临时输出目录
    └── uploads/          # 本地临时上传目录

# 样本输出目录 (实际生成图像保存位置)
/data/lilong/flux_kyc/samples/
├── single_*.jpg          # 单个推理生成的图像
├── batch_*.jpg           # 批量推理生成的图像
└── uploads/              # 用户上传的控制图像
```

## 🔌 API接口

### 模型管理
- `GET /api/status` - 获取模型状态
- `POST /api/load_model` - 加载模型
- `POST /api/unload_model` - 卸载模型

### 推理接口
- `POST /api/batch_inference` - 批量推理
- `POST /api/single_inference` - 单个推理
- `POST /api/upload_image` - 上传图像

### 文档
- `GET /docs` - API文档（Swagger UI）
- `GET /redoc` - API文档（ReDoc）

## ⚠️ 注意事项

1. **显存要求**: FLUX Kontext模型需要大量GPU内存，建议使用24GB+显存的GPU
2. **文件路径**: 确保LoRA checkpoint和图像文件路径正确且可访问
3. **权限问题**: 确保应用有读写相关目录的权限，特别是样本输出目录 `/data/lilong/flux_kyc/samples/`
4. **端口占用**: 默认使用8000端口，如被占用会自动寻找其他可用端口
5. **存储空间**: 确保样本输出目录有足够的磁盘空间存储生成的图像

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查LoRA文件路径是否正确
   - 确认GPU内存足够
   - 查看系统日志获取详细错误信息

2. **推理失败**
   - 确保模型已正确加载
   - 检查图像文件夹路径和权限
   - 验证推理参数范围

3. **界面无法访问**
   - 检查端口是否被占用
   - 确认防火墙设置
   - 查看启动日志

### 日志查看

应用提供实时日志显示，包括：
- 模型加载/卸载状态
- 推理进度和结果
- 错误信息和警告
- 系统资源使用情况

## 🤝 贡献

欢迎提交问题报告和功能请求！

## 📄 许可证

本项目遵循与主AI Toolkit项目相同的许可证。
