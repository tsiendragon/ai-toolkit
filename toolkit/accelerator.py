import os
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None

# Store original init_process_group for restoration
_original_init_process_group = None

def _monkey_patch_init_process_group():
    """Monkey patch torch.distributed.init_process_group to add device_id based on LOCAL_RANK"""
    global _original_init_process_group

    if _original_init_process_group is None:
        _original_init_process_group = dist.init_process_group

    def patched_init_process_group(*args, **kwargs):
        # Add device_id if LOCAL_RANK is set and device_id is not already specified
        if 'LOCAL_RANK' in os.environ and 'device_id' not in kwargs:
            local_rank = int(os.environ['LOCAL_RANK'])
            kwargs['device_id'] = local_rank
            print(f"[rank{local_rank}] Adding device_id={local_rank} to init_process_group")

        return _original_init_process_group(*args, **kwargs)

    # Replace the function
    dist.init_process_group = patched_init_process_group

def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        # Monkey patch init_process_group to add device_id parameter - by Tsien at 2025-08-19
        if 'LOCAL_RANK' in os.environ:
            _monkey_patch_init_process_group()

        # Use standard timeout for distributed training
        kwargs_handlers = []
        if 'LOCAL_RANK' in os.environ:
            from datetime import timedelta
            init_kwargs = InitProcessGroupKwargs(
                backend='nccl',
                timeout=timedelta(minutes=30)
            )
            kwargs_handlers.append(init_kwargs)

        global_accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
    return global_accelerator

def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model
