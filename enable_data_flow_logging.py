#!/usr/bin/env python3
"""
启用数据流程详细日志记录

使用方法：
1. 在训练开始前运行此脚本
2. 或者在训练脚本开头导入此模块

作用：
- 设置详细的 logging 配置
- 专门为数据流程相关的模块设置日志级别
- 提供清晰的日志格式
"""

import logging
import sys
from datetime import datetime

def setup_data_flow_logging(log_level=logging.INFO, log_file=None):
    """
    设置数据流程的详细日志记录

    Args:
        log_level: 日志级别 (logging.DEBUG, logging.INFO, 等)
        log_file: 可选的日志文件路径
    """

    # 创建自定义格式器
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 为关键模块设置特定的日志级别
    key_modules = [
        'toolkit.data_loader',
        'toolkit.dataloader_mixins',
        'jobs.process.BaseSDTrainProcess',
        'accelerate',
        'torch.utils.data'
    ]

    for module in key_modules:
        logger = logging.getLogger(module)
        logger.setLevel(log_level)

    # 记录启动信息
    logging.info("=" * 80)
    logging.info("🚀 数据流程详细日志记录已启用")
    logging.info(f"📝 日志级别: {logging.getLevelName(log_level)}")
    if log_file:
        logging.info(f"📁 日志文件: {log_file}")
    logging.info(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 80)

def enable_debug_logging(log_file=None):
    """启用 DEBUG 级别的详细日志"""
    setup_data_flow_logging(logging.DEBUG, log_file)

def enable_info_logging(log_file=None):
    """启用 INFO 级别的日志"""
    setup_data_flow_logging(logging.INFO, log_file)

if __name__ == "__main__":
    # 如果直接运行此脚本，启用详细日志记录
    import argparse

    parser = argparse.ArgumentParser(description="启用数据流程详细日志记录")
    parser.add_argument("--debug", action="store_true", help="启用 DEBUG 级别日志")
    parser.add_argument("--log-file", help="日志文件路径")

    args = parser.parse_args()

    if args.debug:
        enable_debug_logging(args.log_file)
    else:
        enable_info_logging(args.log_file)

    print("✅ 数据流程日志记录已配置完成")
    print("现在可以运行您的训练脚本来查看详细的数据流程日志")
