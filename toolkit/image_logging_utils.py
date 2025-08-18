"""
图像日志记录工具函数
用于在 TensorBoard 中记录训练图像和控制图像

Created by Tsien at 2025-08-18
"""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union, List, TYPE_CHECKING
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将张量转换为 PIL 图像

    Args:
        tensor: 形状为 [C, H, W] 或 [H, W, C] 的图像张量，值范围 [0, 1] 或 [-1, 1]

    Returns:
        PIL Image 对象
    """
    # 确保张量在 CPU 上
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 处理不同的输入格式
    if tensor.dim() == 4:
        # 如果是批次，取第一个
        tensor = tensor[0]

    if tensor.dim() == 3:
        # [C, H, W] 格式
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:
            # 通道在第一维
            pass
        elif tensor.shape[2] == 3 or tensor.shape[2] == 1:
            # [H, W, C] 格式，转换为 [C, H, W]
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    # 归一化到 [0, 1] 范围
    if tensor.min() < 0:
        # 假设范围是 [-1, 1]
        tensor = (tensor + 1.0) / 2.0

    # 确保在 [0, 1] 范围内
    tensor = torch.clamp(tensor, 0, 1)

    # 转换为 PIL 图像
    if tensor.shape[0] == 1:
        # 灰度图像
        tensor = tensor.squeeze(0)
        image = Image.fromarray((tensor * 255).numpy().astype(np.uint8), mode='L')
    else:
        # RGB 图像
        image = torchvision.transforms.ToPILImage()(tensor)

    return image


def create_image_grid(
    training_images: torch.Tensor,
    control_images: Optional[torch.Tensor] = None,
    max_images: int = 8,
    grid_cols: int = 4
) -> torch.Tensor:
    """
    创建训练图像和控制图像的网格显示

    Args:
        training_images: 训练图像张量 [B, C, H, W]
        control_images: 控制图像张量 [B, C, H, W]，可选
        max_images: 最大显示图像数量
        grid_cols: 网格列数

    Returns:
        网格图像张量
    """
    batch_size = training_images.shape[0]
    num_images = min(batch_size, max_images)

    images_to_show = []

    for i in range(num_images):
        # 添加训练图像
        train_img = training_images[i]
        images_to_show.append(train_img)

        # 添加对应的控制图像（如果有）
        if control_images is not None and i < control_images.shape[0]:
            control_img = control_images[i]
            images_to_show.append(control_img)

    # 创建网格
    if len(images_to_show) > 0:
        grid = torchvision.utils.make_grid(
            images_to_show,
            nrow=grid_cols,
            normalize=True,
            value_range=(-1, 1) if training_images.min() < 0 else (0, 1),
            padding=2,
            pad_value=1.0
        )
        return grid
    else:
        # 返回空白图像
        return torch.zeros(3, 256, 256)


def log_training_images(
    writer: SummaryWriter,
    batch: 'DataLoaderBatchDTO',
    step: int,
    max_images: int = 8,
    tag_prefix: str = "training"
) -> None:
    """
    在 TensorBoard 中记录训练图像和控制图像

    Args:
        writer: TensorBoard SummaryWriter
        batch: 训练批次数据
        step: 当前训练步数
        max_images: 最大记录图像数量
        tag_prefix: 标签前缀
    """
    try:
        # 获取训练图像
        training_images = None
        if batch.tensor is not None:
            training_images = batch.tensor
        elif batch.latents is not None:
            # 如果只有 latents，我们无法直接可视化，跳过
            print(f"⚠️ [IMAGE_LOG] 批次只有 latents，无法直接可视化训练图像")
            return

        if training_images is None:
            print(f"⚠️ [IMAGE_LOG] 批次中没有可用的训练图像")
            return

        # 获取控制图像（如果有）
        control_images = batch.control_tensor if hasattr(batch, 'control_tensor') else None

        # 确保图像在合理范围内
        num_images = min(training_images.shape[0], max_images)

        # 记录训练图像
        if training_images is not None:
            train_grid = create_image_grid(
                training_images[:num_images],
                None,  # 单独显示训练图像
                max_images=num_images,
                grid_cols=4
            )
            writer.add_image(f"{tag_prefix}/training_images", train_grid, step)

        # 记录控制图像（如果有）
        if control_images is not None:
            control_grid = create_image_grid(
                control_images[:num_images],
                None,  # 单独显示控制图像
                max_images=num_images,
                grid_cols=4
            )
            writer.add_image(f"{tag_prefix}/control_images", control_grid, step)

            # 创建并记录组合视图（训练图像和控制图像并排）
            combined_grid = create_image_grid(
                training_images[:num_images],
                control_images[:num_images],
                max_images=num_images,
                grid_cols=4  # 每行4对图像
            )
            writer.add_image(f"{tag_prefix}/training_control_pairs", combined_grid, step)

        print(f"📸 [IMAGE_LOG] 已记录 {num_images} 对训练图像到 TensorBoard (step {step})")

    except Exception as e:
        print(f"❌ [IMAGE_LOG] 记录图像时出错: {e}")


def log_individual_images(
    writer: SummaryWriter,
    batch: 'DataLoaderBatchDTO',
    step: int,
    max_images: int = 8
) -> None:
    """
    记录单个图像对（便于详细查看）

    Args:
        writer: TensorBoard SummaryWriter
        batch: 训练批次数据
        step: 当前训练步数
        max_images: 最大记录图像数量
    """
    try:
        training_images = batch.tensor if batch.tensor is not None else None
        control_images = batch.control_tensor if hasattr(batch, 'control_tensor') and batch.control_tensor is not None else None

        if training_images is None:
            return

        num_images = min(training_images.shape[0], max_images)

        for i in range(num_images):
            # 记录单个训练图像
            train_img = training_images[i:i+1]  # 保持批次维度
            train_grid = torchvision.utils.make_grid(
                train_img,
                normalize=True,
                value_range=(-1, 1) if train_img.min() < 0 else (0, 1)
            )
            writer.add_image(f"individual/training_image_{i}", train_grid, step)

            # 记录对应的控制图像（如果有）
            if control_images is not None and i < control_images.shape[0]:
                control_img = control_images[i:i+1]  # 保持批次维度
                control_grid = torchvision.utils.make_grid(
                    control_img,
                    normalize=True,
                    value_range=(-1, 1) if control_img.min() < 0 else (0, 1)
                )
                writer.add_image(f"individual/control_image_{i}", control_grid, step)

    except Exception as e:
        print(f"❌ [IMAGE_LOG] 记录单个图像时出错: {e}")


def should_log_images(step: int, log_images_every: int, log_images: bool = True) -> bool:
    """
    判断是否应该在当前步骤记录图像

    Args:
        step: 当前训练步数
        log_images_every: 图像记录间隔
        log_images: 是否启用图像记录

    Returns:
        是否应该记录图像
    """
    if not log_images:
        return False

    if log_images_every <= 0:
        return False

    return step % log_images_every == 0 and step > 0
