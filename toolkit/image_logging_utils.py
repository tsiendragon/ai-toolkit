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

    重要：此函数应该只在主进程(rank-0)中调用，以避免分布式训练中的冲突

    Args:
        writer: TensorBoard SummaryWriter
        batch: 训练批次数据
        step: 当前训练步数
        max_images: 最大记录图像数量
        tag_prefix: 标签前缀
    """
    try:
        # 安全检查：避免非主进程记录日志 - by Tsien at 2025-08-18
        import os
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank != 0:
            return  # 非主进程直接返回，不记录日志

        # 安全检查批次对象
        if batch is None:
            print(f"⚠️ [IMAGE_LOG] 批次对象为空")
            return

        # 获取训练图像 - 支持latent缓存模式 - by Tsien at 2025-01-27
        training_images = None

        # 方式1：直接从tensor获取（非缓存模式）
        if hasattr(batch, 'tensor') and batch.tensor is not None:
            training_images = batch.tensor
            print(f"✅ [IMAGE_LOG] 使用直接图像tensor进行记录")

        # 方式2：从latent缓存模式按需加载（新功能）
        elif hasattr(batch, 'latents') and batch.latents is not None:
            print(f"🔄 [IMAGE_LOG] 检测到latent缓存模式，尝试按需加载原始图像...")

            # 尝试使用新的按需加载功能
            if hasattr(batch, 'get_images_for_tensorboard'):
                try:
                    training_images = batch.get_images_for_tensorboard(max_images=max_images)
                    if training_images is not None:
                        print(f"✅ [IMAGE_LOG] 成功从latent缓存模式加载 {training_images.shape[0]} 张图像用于TensorBoard")
                    else:
                        print(f"⚠️ [IMAGE_LOG] 按需加载图像失败，跳过图像记录")
                        return
                except Exception as e:
                    print(f"⚠️ [IMAGE_LOG] 按需加载图像时出错: {e}")
                    return
            else:
                print(f"⚠️ [IMAGE_LOG] 当前版本不支持latent缓存模式下的图像记录")
                print(f"⚠️ [IMAGE_LOG] 如需启用图像记录，请在配置中设置 cache_latents_to_disk: false")
                return

        # 方式3：无法获取图像
        else:
            # 检查batch的所有属性，用于调试
            available_attrs = []
            for attr in dir(batch):
                if not attr.startswith('_'):
                    try:
                        value = getattr(batch, attr, None)
                        if value is not None and not callable(value):
                            if isinstance(value, torch.Tensor):
                                available_attrs.append(f"{attr}(tensor:{list(value.shape)})")
                            else:
                                available_attrs.append(f"{attr}({type(value).__name__})")
                    except:
                        pass
            # print(f"⚠️ [IMAGE_LOG] 批次中没有可用的训练图像。可用属性: {available_attrs}")
            return

        if training_images is None:
            print(f"⚠️ [IMAGE_LOG] 训练图像为空")
            return

        # 获取控制图像（如果有）
        control_images = None
        if hasattr(batch, 'control_tensor') and batch.control_tensor is not None:
            control_images = batch.control_tensor

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

    重要：此函数应该只在主进程(rank-0)中调用，以避免分布式训练中的冲突

    Args:
        writer: TensorBoard SummaryWriter
        batch: 训练批次数据
        step: 当前训练步数
        max_images: 最大记录图像数量
    """
    try:
        # 安全检查：避免非主进程记录日志 - by Tsien at 2025-08-18
        import os
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank != 0:
            return  # 非主进程直接返回，不记录日志

        # 安全检查批次对象 - 支持latent缓存模式 - by Tsien at 2025-01-27
        if batch is None:
            return

        training_images = None

        # 支持latent缓存模式的图像获取
        if hasattr(batch, 'tensor') and batch.tensor is not None:
            training_images = batch.tensor
        elif hasattr(batch, 'latents') and batch.latents is not None and hasattr(batch, 'get_images_for_tensorboard'):
            try:
                training_images = batch.get_images_for_tensorboard(max_images=max_images)
            except Exception as e:
                print(f"⚠️ [IMAGE_LOG] 个别图像记录时按需加载失败: {e}")
                return

        control_images = None
        if hasattr(batch, 'control_tensor') and batch.control_tensor is not None:
            control_images = batch.control_tensor

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


def log_sample_images(
    writer: SummaryWriter,
    images: List[Image.Image],
    prompts: List[str],
    step: int,
    max_images: int = 8,
    tag_prefix: str = "samples",
    control_images: Optional[List[Image.Image]] = None
) -> None:
    """
    在 TensorBoard 中记录训练过程中生成的样本图像

    重要：此函数应该只在主进程(rank-0)中调用，以避免分布式训练中的冲突

    Args:
        writer: TensorBoard SummaryWriter
        images: 生成的 PIL 图像列表
        prompts: 对应的提示词列表
        step: 当前训练步数
        max_images: 最大记录图像数量
        tag_prefix: 标签前缀
        control_images: 控制图像列表（原图），用于对比显示
    """
    try:
        # 安全检查：避免非主进程记录日志
        import os
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank != 0:
            return  # 非主进程直接返回，不记录日志

        if not images or len(images) == 0:
            print(f"⚠️ [SAMPLE_LOG] 没有样本图像需要记录")
            return

        # 限制记录的图像数量
        num_images = min(len(images), max_images)
        images_to_log = images[:num_images]
        prompts_to_log = prompts[:num_images] if prompts else [f"sample_{i}" for i in range(num_images)]

        # 将 PIL 图像转换为张量
        image_tensors = []
        for i, pil_img in enumerate(images_to_log):
            try:
                # 确保图像是 RGB 模式
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # 转换为张量 [C, H, W]，值范围 [0, 1]
                img_array = np.array(pil_img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                image_tensors.append(img_tensor)
            except Exception as e:
                print(f"⚠️ [SAMPLE_LOG] 转换图像 {i} 时出错: {e}")
                continue

        if not image_tensors:
            print(f"⚠️ [SAMPLE_LOG] 没有有效的图像张量")
            return

        # 创建网格显示
        if len(image_tensors) > 1:
            # 多个图像，创建网格
            grid_cols = min(4, len(image_tensors))  # 最多4列
            grid = torchvision.utils.make_grid(
                image_tensors,
                nrow=grid_cols,
                normalize=False,  # 已经归一化到 [0,1]
                padding=2,
                pad_value=1.0
            )
            writer.add_image(f"{tag_prefix}/generated_samples", grid, step)
        else:
            # 单个图像
            writer.add_image(f"{tag_prefix}/generated_sample", image_tensors[0], step)

        # 如果有控制图像，创建对比拼接显示
        if control_images and len(control_images) > 0:
            try:
                comparison_tensors = []
                control_tensors = []

                # 转换控制图像为张量
                for i, ctrl_img in enumerate(control_images[:num_images]):
                    try:
                        if ctrl_img.mode != 'RGB':
                            ctrl_img = ctrl_img.convert('RGB')
                        # 调整控制图像尺寸与生成图像一致
                        if i < len(images_to_log):
                            target_size = images_to_log[i].size
                            ctrl_img = ctrl_img.resize(target_size, Image.Resampling.LANCZOS)

                        ctrl_array = np.array(ctrl_img).astype(np.float32) / 255.0
                        ctrl_tensor = torch.from_numpy(ctrl_array).permute(2, 0, 1)
                        control_tensors.append(ctrl_tensor)
                    except Exception as e:
                        print(f"⚠️ [SAMPLE_LOG] 转换控制图像 {i} 时出错: {e}")
                        continue

                # 创建对比拼接图像（原图-编辑图并排）
                for i, (gen_tensor, ctrl_tensor) in enumerate(zip(image_tensors, control_tensors)):
                    if i >= len(prompts_to_log):
                        break
                    try:
                        # 拼接图像：左边原图，右边编辑图
                        comparison_tensor = torch.cat([ctrl_tensor, gen_tensor], dim=2)  # 水平拼接
                        comparison_tensors.append(comparison_tensor)
                    except Exception as e:
                        print(f"⚠️ [SAMPLE_LOG] 拼接图像 {i} 时出错: {e}")
                        continue

                # 记录对比网格
                if comparison_tensors:
                    if len(comparison_tensors) > 1:
                        comparison_grid = torchvision.utils.make_grid(
                            comparison_tensors,
                            nrow=min(2, len(comparison_tensors)),  # 每行最多2个对比图
                            normalize=False,
                            padding=4,
                            pad_value=1.0
                        )
                        writer.add_image(f"{tag_prefix}/before_after_comparison", comparison_grid, step)
                    else:
                        writer.add_image(f"{tag_prefix}/before_after_comparison", comparison_tensors[0], step)

                # 移除单个对比图像记录 - 只保留网格视图 - by Tsien at 2025-01-27
                # 这样可以避免重复记录相同的图像内容

                print(f"📸 [SAMPLE_LOG] 已记录 {len(comparison_tensors)} 张对比图像到 TensorBoard")

            except Exception as e:
                print(f"⚠️ [SAMPLE_LOG] 创建对比图像时出错: {e}")

        # 记录单个生成图像（便于详细查看）
        for i, (img_tensor, prompt) in enumerate(zip(image_tensors, prompts_to_log)):
            # 清理提示词作为标签（移除特殊字符）
            clean_prompt = prompt.replace("/", "_").replace("\\", "_")[:50]  # 限制长度
            writer.add_image(f"{tag_prefix}/individual/sample_{i}_{clean_prompt}", img_tensor, step)

        print(f"📸 [SAMPLE_LOG] 已记录 {len(image_tensors)} 张样本图像到 TensorBoard (step {step})")

    except Exception as e:
        print(f"❌ [SAMPLE_LOG] 记录样本图像时出错: {e}")


def should_log_images(step: int, log_images_every: int, log_images: bool = True) -> bool:
    """
    判断是否应该在当前步骤记录图像

    Args:
        step: 当前训练步数
        log_images_every: 图像记录间隔
        log_images: 是否启用图像记录

    Returns:
        bool: 是否应该记录图像
    """
    if not log_images:
        return False

    if log_images_every <= 0:
        return False

    return step % log_images_every == 0 and step > 0


def should_log_samples(step: int, sample_every: int, log_samples: bool = True) -> bool:
    """
    判断是否应该在当前步骤记录样本图像

    Args:
        step: 当前训练步数
        sample_every: 样本生成间隔（当生成样本时才记录）
        log_samples: 是否启用样本记录

    Returns:
        bool: 是否应该记录样本
    """
    if not log_samples:
        return False

    if sample_every <= 0:
        return False

    return step % sample_every == 0 and step > 0
