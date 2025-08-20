#!/bin/bash

# FLUX Kontext LoRA Demo 启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DEMO_DIR")"

print_info "FLUX Kontext LoRA Demo 启动中..."
print_info "Demo目录: $DEMO_DIR"
print_info "项目根目录: $PROJECT_ROOT"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    print_error "Python3 未找到，请先安装Python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
print_info "Python版本: $PYTHON_VERSION"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_info "GPU信息: $CUDA_INFO"
else
    print_warning "NVIDIA GPU未检测到，将使用CPU模式（不推荐）"
fi

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_info "检测到虚拟环境: $VIRTUAL_ENV"
else
    print_warning "未检测到虚拟环境，建议使用虚拟环境"
fi

# 切换到demo目录
cd "$DEMO_DIR"

# 检查依赖安装
print_info "检查依赖..."
if ! python3 -c "import fastapi, uvicorn, torch" &> /dev/null; then
    print_warning "部分依赖未安装，正在安装..."

    # 安装依赖
    if [[ -f "requirements.txt" ]]; then
        print_info "从requirements.txt安装依赖..."
        pip install -r requirements.txt
    else
        print_info "安装基础依赖..."
        pip install fastapi uvicorn torch torchvision transformers diffusers pillow
    fi
else
    print_success "依赖检查通过"
fi

# 检查必要文件
REQUIRED_FILES=("app.py" "index.html" "static/css/style.css" "static/js/app.js")
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "必要文件缺失: $file"
        exit 1
    fi
done
print_success "文件检查通过"

# 创建必要目录
mkdir -p static/outputs static/uploads
mkdir -p "/data/lilong/flux_kyc/samples/uploads"
print_info "创建输出目录完成"
print_info "样本保存目录: /data/lilong/flux_kyc/samples"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
# 使用新的HF_HOME替代已弃用的TRANSFORMERS_CACHE
if [[ -z "$HF_HOME" && -z "$TRANSFORMERS_CACHE" ]]; then
    export HF_HOME="$HOME/.cache/huggingface"
fi

# 检查端口占用
PORT=${1:-8080}
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "端口 $PORT 已被占用，尝试使用其他端口..."
    PORT=$((PORT + 1))
    while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
        PORT=$((PORT + 1))
        if [[ $PORT -gt 8010 ]]; then
            print_error "无法找到可用端口"
            exit 1
        fi
    done
fi

print_success "使用端口: $PORT"

# 打印启动信息
echo ""
print_info "=================== 启动信息 ==================="
print_info "应用地址: http://localhost:$PORT"
print_info "API文档: http://localhost:$PORT/docs"
print_info "项目目录: $DEMO_DIR"
print_info "日志级别: INFO"
print_info "==============================================="
echo ""

# 启动应用
print_info "启动FLUX Kontext LoRA Demo..."
python3 -m uvicorn app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --reload-dir . \
    --log-level info

print_info "应用已退出"
