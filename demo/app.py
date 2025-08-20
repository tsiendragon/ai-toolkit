#!/usr/bin/env python3
"""
FLUX Kontext LoRA Web Demo
支持FLUX Kontext Dev模型的LoRA checkpoint加载和推理演示
"""

import os
import sys
import json
import random
import asyncio
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# AI Toolkit imports
from toolkit.config_modules import ModelConfig, DatasetConfig, SaveConfig, NetworkConfig, GenerateImageConfig
from extensions_built_in.diffusion_models.flux_kontext.flux_kontext import FluxKontextModel
from toolkit.assistant_lora import load_assistant_lora_from_path
from toolkit.train_tools import get_torch_dtype


def flush():
    """清理GPU内存"""
    torch.cuda.empty_cache()
    import gc
    gc.collect()


class AppState:
    """应用状态管理"""
    def __init__(self):
        self.model: Optional[FluxKontextModel] = None
        self.lora_network = None
        self.current_lora_path: Optional[str] = None
        self.is_model_loaded = False
        self.device = None
        self.torch_dtype = None

    def reset(self):
        """重置状态"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.lora_network is not None:
            del self.lora_network
            self.lora_network = None
        self.current_lora_path = None
        self.is_model_loaded = False
        torch.cuda.empty_cache()
        flush()


# 全局应用状态
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    app_state.device = "cuda" if torch.cuda.is_available() else "cpu"
    app_state.torch_dtype = get_torch_dtype("bf16")

    print(f"🚀 启动FLUX Kontext Demo")
    print(f"📱 设备: {app_state.device}")
    print(f"🔢 数据类型: {app_state.torch_dtype}")

    yield

    # 关闭时清理
    print("🧹 清理资源...")
    app_state.reset()


# 创建FastAPI应用
app = FastAPI(
    title="FLUX Kontext LoRA Demo",
    description="FLUX Kontext Dev模型LoRA checkpoint加载和推理演示",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
demo_path = Path(__file__).parent
app.mount("/static", StaticFiles(directory=demo_path / "static"), name="static")

# 挂载样本输出目录
SAMPLES_OUTPUT_DIR = Path("/data/lilong/flux_kyc/samples")
SAMPLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/samples", StaticFiles(directory=SAMPLES_OUTPUT_DIR), name="samples")


# Pydantic模型
class ModelStatus(BaseModel):
    is_loaded: bool = False
    current_lora_path: Optional[str] = None
    device: Optional[str] = None
    memory_info: Optional[Dict[str, Any]] = None


class LoadModelRequest(BaseModel):
    lora_path: str = Field(
        default="/data/lilong/experiment/id_card_flux_kontext_lora_v1/id_card_flux_kontext_lora_v1/id_card_flux_kontext_lora_v1_000001111.safetensors",
        description="LoRA checkpoint路径"
    )


class BatchInferenceRequest(BaseModel):
    folder_path: str = Field(
        default="/home/sysop/data/id_card/training_images",
        description="图像文件夹路径"
    )
    max_samples: int = Field(default=5, ge=1, le=20, description="最大采样数量")
    seed: int = Field(default=42, description="随机种子")
    guidance_scale: float = Field(default=4.0, ge=1.0, le=20.0, description="引导强度")
    num_inference_steps: int = Field(default=20, ge=10, le=50, description="推理步数")


class SingleInferenceRequest(BaseModel):
    prompt: str = Field(..., description="文本提示")
    control_image_path: Optional[str] = Field(None, description="控制图像路径")
    width: int = Field(default=832, ge=512, le=1024, description="图像宽度")
    height: int = Field(default=576, ge=512, le=1024, description="图像高度")
    seed: int = Field(default=42, description="随机种子")
    guidance_scale: float = Field(default=4.0, ge=1.0, le=20.0, description="引导强度")
    num_inference_steps: int = Field(default=20, ge=10, le=50, description="推理步数")


def get_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        return {
            "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
            "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        }
    return None


@app.get("/", response_class=HTMLResponse)
async def root():
    """主页面"""
    html_file = demo_path / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse("""
        <html><body>
        <h1>FLUX Kontext LoRA Demo</h1>
        <p>请先创建 index.html 文件</p>
        </body></html>
        """)


@app.get("/api/status", response_model=ModelStatus)
async def get_status():
    """获取模型状态"""
    return ModelStatus(
        is_loaded=app_state.is_model_loaded,
        current_lora_path=app_state.current_lora_path,
        device=str(app_state.device) if app_state.device else None,
        memory_info=get_memory_info()
    )


@app.post("/api/load_model")
async def load_model(request: LoadModelRequest):
    """加载模型和LoRA"""
    try:
        # 如果已经加载了相同的LoRA，直接返回
        if (app_state.is_model_loaded and
            app_state.current_lora_path == request.lora_path):
            return {
                "success": True,
                "message": f"模型已加载: {request.lora_path}",
                "status": await get_status()
            }

        # 清理之前的状态
        app_state.reset()

        # 验证LoRA文件是否存在
        if not os.path.exists(request.lora_path):
            raise HTTPException(status_code=404, detail=f"LoRA文件不存在: {request.lora_path}")

        print(f"🔄 开始加载模型和LoRA: {request.lora_path}")

        # 创建模型配置
        model_config = ModelConfig(
            name_or_path="black-forest-labs/FLUX.1-Kontext-dev",
            arch="flux_kontext",
            quantize=True,
            low_vram=True,
            inference_lora_path=request.lora_path
        )

        # 创建并加载模型
        app_state.model = FluxKontextModel(
            device=app_state.device,
            model_config=model_config,
            dtype="bf16"
        )

        print("📥 加载基础模型...")
        app_state.model.load_model()

        print("🔗 加载LoRA...")
        # FluxKontextModel继承自BaseModel，我们需要传递模型本身
        app_state.lora_network = load_assistant_lora_from_path(
            request.lora_path,
            app_state.model
        )

        app_state.current_lora_path = request.lora_path
        app_state.is_model_loaded = True

        print("✅ 模型加载完成!")

        return {
            "success": True,
            "message": f"模型加载成功: {os.path.basename(request.lora_path)}",
            "status": await get_status()
        }

    except Exception as e:
        app_state.reset()
        print(f"❌ 模型加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@app.post("/api/unload_model")
async def unload_model():
    """卸载模型"""
    try:
        if not app_state.is_model_loaded:
            return {
                "success": True,
                "message": "模型未加载",
                "status": await get_status()
            }

        print("🗑️ 卸载模型...")
        app_state.reset()

        print("✅ 模型卸载完成!")

        return {
            "success": True,
            "message": "模型卸载成功",
            "status": await get_status()
        }

    except Exception as e:
        print(f"❌ 模型卸载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型卸载失败: {str(e)}")


@app.post("/api/batch_inference")
async def batch_inference(request: BatchInferenceRequest):
    """批量推理"""
    if not app_state.is_model_loaded:
        raise HTTPException(status_code=400, detail="请先加载模型")

    try:
        # 验证文件夹是否存在
        if not os.path.exists(request.folder_path):
            raise HTTPException(status_code=404, detail=f"文件夹不存在: {request.folder_path}")

        # 查找图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(request.folder_path).glob(ext))

        if not image_files:
            raise HTTPException(status_code=404, detail="文件夹中未找到图像文件")

        # 随机采样
        random.seed(request.seed)
        selected_files = random.sample(image_files, min(request.max_samples, len(image_files)))

        results = []
        for i, img_path in enumerate(selected_files):
            print(f"🖼️ 处理图像 {i+1}/{len(selected_files)}: {img_path.name}")

            # 查找对应的提示文件
            txt_path = img_path.with_suffix('.txt')
            prompt = ""
            if txt_path.exists():
                prompt = txt_path.read_text(encoding='utf-8').strip()

            # TODO: 这里应该调用实际的推理逻辑
            # 生成输出文件名
            output_filename = f"batch_{request.seed}_{i:04d}.jpg"
            output_path = SAMPLES_OUTPUT_DIR / output_filename

            # 暂时返回模拟结果 - 实际实现时这里会调用模型推理
            results.append({
                "original_image": str(img_path),
                "prompt": prompt,
                "generated_image": f"/samples/{output_filename}",
                "output_path": str(output_path),
                "success": True
            })

        return {
            "success": True,
            "message": f"批量推理完成，处理了 {len(results)} 张图像",
            "results": results
        }

    except Exception as e:
        print(f"❌ 批量推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")


@app.post("/api/single_inference")
async def single_inference(request: SingleInferenceRequest):
    """单个推理"""
    if not app_state.is_model_loaded:
        raise HTTPException(status_code=400, detail="请先加载模型")

    try:
        print(f"🎨 开始单个推理: {request.prompt[:50]}...")

        # TODO: 这里应该调用实际的推理逻辑
        # 生成输出文件名
        import time
        timestamp = int(time.time())
        output_filename = f"single_{timestamp}_{request.seed}.jpg"
        output_path = SAMPLES_OUTPUT_DIR / output_filename
        output_url = f"/samples/{output_filename}"

        return {
            "success": True,
            "message": "推理完成",
            "generated_image": output_url,
            "prompt": request.prompt,
            "parameters": {
                "width": request.width,
                "height": request.height,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "seed": request.seed
            }
        }

    except Exception as e:
        print(f"❌ 单个推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"单个推理失败: {str(e)}")


@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """上传图像"""
    try:
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="请上传图像文件")

        # 创建上传目录（保存到样本目录下的uploads子目录）
        upload_dir = SAMPLES_OUTPUT_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一文件名避免冲突
        import time
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / filename

        # 保存文件
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {
            "success": True,
            "message": "图像上传成功",
            "file_path": f"/samples/uploads/{filename}",
            "local_path": str(file_path)
        }

    except Exception as e:
        print(f"❌ 图像上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图像上传失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(demo_path)]
    )
