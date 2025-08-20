"""
Safe safetensors utilities for distributed training
Provides pickle-safe wrappers for safetensors operations
Created by Tsien at 2025-08-19
"""
import os
import pickle
import logging
from typing import Optional, Dict, Any, Union
import torch

logger = logging.getLogger(__name__)

class PickleSafeException(Exception):
    """Pickle-safe exception that can be serialized across processes"""
    def __init__(self, message: str, original_exception_type: str = None):
        super().__init__(message)
        self.message = message
        self.original_exception_type = original_exception_type or "UnknownError"

    def __reduce__(self):
        # Ensure pickle can serialize this exception
        return (PickleSafeException, (self.message, self.original_exception_type))

def safe_load_file(filepath: str, device: str = 'cpu') -> Optional[Dict[str, torch.Tensor]]:
    """
    Safely load safetensors file with pickle-safe error handling

    Args:
        filepath: Path to safetensors file
        device: Device to load tensors to

    Returns:
        Dict of tensors or None if error

    Raises:
        PickleSafeException: If loading fails, with pickle-safe error
    """
    try:
        from safetensors.torch import load_file
        return load_file(filepath, device=device)
    except Exception as e:
        # Convert any safetensors error to a pickle-safe exception
        error_msg = f"Failed to load safetensors file {filepath}: {str(e)}"
        logger.warning(f"[SAFETENSORS] {error_msg}")

        # For distributed training, raise a pickle-safe exception
        if 'LOCAL_RANK' in os.environ:
            raise PickleSafeException(error_msg, type(e).__name__)
        else:
            # In single-GPU mode, re-raise original exception
            raise e

def safe_save_file(tensors: Dict[str, torch.Tensor], filepath: str, metadata: Optional[Dict[str, str]] = None) -> bool:
    """
    Safely save safetensors file with pickle-safe error handling

    Args:
        tensors: Dictionary of tensors to save
        filepath: Output file path
        metadata: Optional metadata dict

    Returns:
        True if successful, False otherwise

    Raises:
        PickleSafeException: If saving fails, with pickle-safe error
    """
    try:
        from safetensors.torch import save_file

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if metadata is not None:
            save_file(tensors, filepath, metadata=metadata)
        else:
            save_file(tensors, filepath)
        return True

    except Exception as e:
        # Convert any safetensors error to a pickle-safe exception
        error_msg = f"Failed to save safetensors file {filepath}: {str(e)}"
        logger.warning(f"[SAFETENSORS] {error_msg}")

        # For distributed training, raise a pickle-safe exception
        if 'LOCAL_RANK' in os.environ:
            raise PickleSafeException(error_msg, type(e).__name__)
        else:
            # In single-GPU mode, re-raise original exception
            raise e

def is_safetensors_file_valid(filepath: str) -> bool:
    """
    Check if a safetensors file is valid without loading it fully

    Args:
        filepath: Path to safetensors file

    Returns:
        True if file exists and appears valid
    """
    if not os.path.exists(filepath):
        return False

    try:
        # Try to load just the metadata to check if file is valid
        from safetensors import safe_open
        with safe_open(filepath, framework="pt", device="cpu"):
            pass
        return True
    except Exception as e:
        logger.debug(f"[SAFETENSORS] File {filepath} appears invalid: {e}")
        return False

def cleanup_corrupted_cache_file(filepath: str) -> None:
    """
    Remove a corrupted safetensors cache file

    Args:
        filepath: Path to potentially corrupted file
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"[SAFETENSORS] Removed corrupted cache file: {filepath}")
    except Exception as e:
        logger.warning(f"[SAFETENSORS] Failed to remove corrupted file {filepath}: {e}")
