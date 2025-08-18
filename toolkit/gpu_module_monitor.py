"""
GPU模块监控工具
用于监控和记录哪些AI模块当前加载在GPU上
Created by Tsien at 2025-01-27
"""

import torch
from typing import Dict, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion

# GPU模块编码系统 - 使用位标志便于组合
GPU_MODULE_CODES = {
    'VAE_ENCODER': 1,      # VAE编码器
    'VAE_DECODER': 2,      # VAE解码器
    'UNET': 4,             # UNet主模型
    'TEXT_ENCODER': 8,     # 文本编码器
    'ADAPTER': 16,         # 适配器模块
    'REFINER_UNET': 32,    # 精炼器UNet
    'CLIP': 64,            # CLIP模型
    'OTHER': 128,          # 其他模块
}

# 模块名称映射
MODULE_NAMES = {
    1: 'VAE_ENC',
    2: 'VAE_DEC',
    4: 'UNET',
    8: 'TEXT_ENC',
    16: 'ADAPTER',
    32: 'REFINER',
    64: 'CLIP',
    128: 'OTHER',
}

def get_module_on_gpu(module: torch.nn.Module, target_device: Union[str, torch.device] = 'cuda') -> bool:
    """
    检查模块是否在指定GPU设备上

    Args:
        module: PyTorch模块
        target_device: 目标设备，默认为'cuda'

    Returns:
        bool: 模块是否在目标设备上
    """
    if module is None:
        return False

    # 检查模块的第一个参数的设备
    try:
        for param in module.parameters():
            device_str = str(param.device)
            if isinstance(target_device, torch.device):
                target_device = str(target_device)
            return target_device in device_str
    except:
        return False

    return False

def get_gpu_module_state(model_instance) -> int:
    """
    获取当前GPU上加载的模块状态编码

    Args:
        model_instance: 模型实例，可以是StableDiffusion、VAE训练或ESRGAN训练实例

    Returns:
        int: 模块状态编码（位标志组合）
    """
    state_code = 0

    # 检查StableDiffusion模型
    if hasattr(model_instance, 'sd') and model_instance.sd is not None:
        sd = model_instance.sd

        # 检查VAE (通常VAE encoder和decoder是一体的)
        if hasattr(sd, 'vae') and sd.vae is not None:
            if get_module_on_gpu(sd.vae):
                # VAE通常包含encoder和decoder，这里简化为同时标记
                state_code |= GPU_MODULE_CODES['VAE_ENCODER']
                state_code |= GPU_MODULE_CODES['VAE_DECODER']

        # 检查UNet
        if hasattr(sd, 'unet') and sd.unet is not None:
            if get_module_on_gpu(sd.unet):
                state_code |= GPU_MODULE_CODES['UNET']

        # 检查Text Encoder
        if hasattr(sd, 'text_encoder') and sd.text_encoder is not None:
            # 处理单个或多个text encoder
            if isinstance(sd.text_encoder, list):
                for te in sd.text_encoder:
                    if get_module_on_gpu(te):
                        state_code |= GPU_MODULE_CODES['TEXT_ENCODER']
                        break
            else:
                if get_module_on_gpu(sd.text_encoder):
                    state_code |= GPU_MODULE_CODES['TEXT_ENCODER']

        # 检查Adapter
        if hasattr(sd, 'adapter') and sd.adapter is not None:
            if get_module_on_gpu(sd.adapter):
                state_code |= GPU_MODULE_CODES['ADAPTER']

        # 检查Refiner UNet
        if hasattr(sd, 'refiner_unet') and sd.refiner_unet is not None:
            if get_module_on_gpu(sd.refiner_unet):
                state_code |= GPU_MODULE_CODES['REFINER_UNET']

        # 检查CLIP (如果单独存在)
        if hasattr(sd, 'clip') and sd.clip is not None:
            if get_module_on_gpu(sd.clip):
                state_code |= GPU_MODULE_CODES['CLIP']

    # 检查独立的VAE模型 (TrainVAEProcess)
    elif hasattr(model_instance, 'vae') and model_instance.vae is not None:
        if get_module_on_gpu(model_instance.vae):
            state_code |= GPU_MODULE_CODES['VAE_ENCODER']
            state_code |= GPU_MODULE_CODES['VAE_DECODER']

        # 检查目标VAE
        if hasattr(model_instance, 'target_latent_vae') and model_instance.target_latent_vae is not None:
            if get_module_on_gpu(model_instance.target_latent_vae):
                state_code |= GPU_MODULE_CODES['OTHER']  # 标记为其他模块

    # 检查ESRGAN模型 (TrainESRGANProcess)
    elif hasattr(model_instance, 'model') and model_instance.model is not None:
        if get_module_on_gpu(model_instance.model):
            state_code |= GPU_MODULE_CODES['OTHER']  # ESRGAN标记为其他模块

        # 检查判别器
        if hasattr(model_instance, 'critic') and model_instance.critic is not None:
            critic_model = getattr(model_instance.critic, 'critic', None)
            if critic_model is not None and get_module_on_gpu(critic_model):
                state_code |= GPU_MODULE_CODES['ADAPTER']  # 判别器标记为adapter

    return state_code

def decode_gpu_module_state(state_code: int) -> str:
    """
    将模块状态编码解码为可读字符串

    Args:
        state_code: 模块状态编码

    Returns:
        str: 可读的模块状态描述
    """
    if state_code == 0:
        return "NONE"

    active_modules = []
    for code, name in MODULE_NAMES.items():
        if state_code & code:
            active_modules.append(name)

    return "+".join(active_modules)

def log_gpu_module_state_to_tensorboard(
    writer,
    model_instance,
    step: int,
    tag_prefix: str = "gpu_modules"
) -> None:
    """
    将GPU模块状态记录到TensorBoard

    Args:
        writer: TensorBoard SummaryWriter
        model_instance: 模型实例，可以是StableDiffusion、VAE训练或ESRGAN训练实例
        step: 当前训练步数
        tag_prefix: TensorBoard标签前缀
    """
    if writer is None:
        return

    try:
        # 获取模块状态编码
        state_code = get_gpu_module_state(model_instance)

        # 记录数值编码
        writer.add_scalar(f"{tag_prefix}/state_code", state_code, step)

        # 记录每个模块的状态
        for code, name in MODULE_NAMES.items():
            module_active = 1 if (state_code & code) else 0
            writer.add_scalar(f"{tag_prefix}/module_{name.lower()}", module_active, step)

        # 添加文本描述 (可选)
        state_desc = decode_gpu_module_state(state_code)
        writer.add_text(f"{tag_prefix}/description", state_desc, step)

    except Exception as e:
        print(f"⚠️ [GPU_MODULE_MONITOR] 记录GPU模块状态失败: {e}")

def get_gpu_memory_info() -> Dict[str, float]:
    """
    获取GPU内存使用信息

    Returns:
        Dict[str, float]: GPU内存信息
    """
    if not torch.cuda.is_available():
        return {"used_mb": 0, "total_mb": 0, "free_mb": 0, "utilization": 0}

    try:
        used_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
        free_memory = total_memory - used_memory
        utilization = (used_memory / total_memory) * 100

        return {
            "used_mb": used_memory,
            "total_mb": total_memory,
            "free_mb": free_memory,
            "utilization": utilization
        }
    except:
        return {"used_mb": 0, "total_mb": 0, "free_mb": 0, "utilization": 0}

def log_gpu_memory_to_tensorboard(
    writer,
    step: int,
    tag_prefix: str = "gpu_memory"
) -> None:
    """
    将GPU内存使用情况记录到TensorBoard

    Args:
        writer: TensorBoard SummaryWriter
        step: 当前训练步数
        tag_prefix: TensorBoard标签前缀
    """
    if writer is None:
        return

    try:
        memory_info = get_gpu_memory_info()

        for key, value in memory_info.items():
            writer.add_scalar(f"{tag_prefix}/{key}", value, step)

    except Exception as e:
        print(f"⚠️ [GPU_MODULE_MONITOR] 记录GPU内存状态失败: {e}")

# 使用示例和说明
"""
使用示例:

# 在训练循环中
state_code = get_gpu_module_state(self)  # 传递训练实例
print(f"当前GPU模块: {decode_gpu_module_state(state_code)} (编码: {state_code})")

# 记录到TensorBoard (已自动集成到训练流程中)
log_gpu_module_state_to_tensorboard(self.writer, self, self.step_num)
log_gpu_memory_to_tensorboard(self.writer, self.step_num)

模块编码说明 (位标志，可组合):
- 0: 无模块在GPU
- 1: VAE Encoder (01)
- 2: VAE Decoder (10)
- 3: VAE完整 (11)
- 4: UNet (100)
- 8: Text Encoder (1000)
- 16: Adapter (10000)
- 32: Refiner UNet (100000)
- 64: CLIP (1000000)
- 128: 其他模块 (10000000)

常见组合示例:
- 3: VAE Encoder + Decoder (VAE训练)
- 12: UNet + Text Encoder (SD训练基础)
- 15: VAE + UNet + Text Encoder (完整SD训练)
- 31: 所有SD基础模块
- 128: ESRGAN等其他模型训练

TensorBoard中的监控指标:
■ 模块状态图表:
  - gpu_modules/state_code: 总体编码数值
  - gpu_modules/module_vae_enc: VAE编码器状态 (0/1)
  - gpu_modules/module_vae_dec: VAE解码器状态 (0/1)
  - gpu_modules/module_unet: UNet状态 (0/1)
  - gpu_modules/module_text_enc: Text Encoder状态 (0/1)
  - gpu_modules/module_adapter: Adapter状态 (0/1)
  - gpu_modules/module_refiner: Refiner状态 (0/1)
  - gpu_modules/module_clip: CLIP状态 (0/1)
  - gpu_modules/module_other: 其他模块状态 (0/1)

■ 内存监控图表:
  - gpu_memory/used_mb: GPU已用内存(MB)
  - gpu_memory/total_mb: GPU总内存(MB)
  - gpu_memory/free_mb: GPU可用内存(MB)
  - gpu_memory/utilization: GPU内存使用率(%)

■ 文本日志:
  - gpu_modules/description: 可读的模块状态描述

✅ 实时监控效果:
现在每个训练step都会自动记录GPU模块加载状态，你可以在TensorBoard中：
1. 实时观察哪些模块在GPU上
2. 监控模块切换模式的变化
3. 追踪GPU内存使用趋势
4. 诊断内存不足或模块加载问题

🎯 使用场景:
- 调试GPU内存管理
- 优化模块加载策略
- 监控分布式训练中的模块分布
- 分析不同训练阶段的资源使用
"""
