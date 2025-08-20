#!/usr/bin/env python3
"""
FLUX Kontext LoRA Demo 启动脚本 (跨平台版本)
"""

import os
import sys
import subprocess
import socket
from pathlib import Path

def print_colored(message, color_code):
    """打印带颜色的消息"""
    print(f"\033[{color_code}m{message}\033[0m")

def print_info(message):
    print_colored(f"[INFO] {message}", "0;34")

def print_success(message):
    print_colored(f"[SUCCESS] {message}", "0;32")

def print_warning(message):
    print_colored(f"[WARNING] {message}", "1;33")

def print_error(message):
    print_colored(f"[ERROR] {message}", "0;31")

def check_port_available(port):
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def find_available_port(start_port=8000):
    """找到可用端口"""
    port = start_port
    while port <= start_port + 10:
        if check_port_available(port):
            return port
        port += 1
    return None

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'transformers',
        'diffusers',
        'PIL'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    return missing_packages

def install_dependencies():
    """安装依赖包"""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if requirements_file.exists():
        print_info("从requirements.txt安装依赖...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
    else:
        print_info("安装基础依赖...")
        basic_deps = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "diffusers>=0.24.0",
            "pillow>=10.0.0",
            "python-multipart>=0.0.6"
        ]

        for dep in basic_deps:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

def main():
    """主函数"""
    print_info("FLUX Kontext LoRA Demo 启动中...")

    # 获取目录路径
    demo_dir = Path(__file__).parent.absolute()
    project_root = demo_dir.parent

    print_info(f"Demo目录: {demo_dir}")
    print_info(f"项目根目录: {project_root}")

    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print_info(f"Python版本: {python_version}")

    if sys.version_info < (3, 8):
        print_error("需要Python 3.8或更高版本")
        sys.exit(1)

    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print_info(f"检测到CUDA GPU: {gpu_name} (共{gpu_count}个)")
        else:
            print_warning("未检测到CUDA GPU，将使用CPU模式（不推荐）")
    except ImportError:
        print_warning("PyTorch未安装，稍后将自动安装")

    # 切换到demo目录
    os.chdir(demo_dir)

    # 检查依赖
    print_info("检查依赖...")
    missing_deps = check_dependencies()

    if missing_deps:
        print_warning(f"缺少依赖: {', '.join(missing_deps)}")
        try:
            install_dependencies()
            print_success("依赖安装完成")
        except subprocess.CalledProcessError as e:
            print_error(f"依赖安装失败: {e}")
            sys.exit(1)
    else:
        print_success("依赖检查通过")

    # 检查必要文件
    required_files = [
        "app.py",
        "index.html",
        "static/css/style.css",
        "static/js/app.js"
    ]

    for file_path in required_files:
        if not (demo_dir / file_path).exists():
            print_error(f"必要文件缺失: {file_path}")
            sys.exit(1)

    print_success("文件检查通过")

    # 创建必要目录
    (demo_dir / "static" / "outputs").mkdir(parents=True, exist_ok=True)
    (demo_dir / "static" / "uploads").mkdir(parents=True, exist_ok=True)

    # 创建样本输出目录
    samples_dir = Path("/data/lilong/flux_kyc/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "uploads").mkdir(parents=True, exist_ok=True)
    print_info("创建输出目录完成")
    print_info(f"样本保存目录: {samples_dir}")

    # 设置环境变量
    os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 使用新的HF_HOME替代已弃用的TRANSFORMERS_CACHE
    if "HF_HOME" not in os.environ and "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

    # 查找可用端口
    port = find_available_port(8000)
    if port is None:
        print_error("无法找到可用端口")
        sys.exit(1)

    print_success(f"使用端口: {port}")

    # 打印启动信息
    print("\n" + "=" * 50)
    print_info("FLUX Kontext LoRA Demo 启动信息")
    print_info(f"应用地址: http://localhost:{port}")
    print_info(f"API文档: http://localhost:{port}/docs")
    print_info(f"项目目录: {demo_dir}")
    print("=" * 50 + "\n")

    # 启动应用
    try:
        print_info("启动应用...")
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload",
            "--reload-dir", ".",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print_info("用户中断，正在退出...")
    except subprocess.CalledProcessError as e:
        print_error(f"应用启动失败: {e}")
        sys.exit(1)

    print_info("应用已退出")

if __name__ == "__main__":
    main()
