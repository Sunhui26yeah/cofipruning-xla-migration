"""PyTorch/XLA helpers for running CoFiPruning on TPU."""
import torch

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    IS_XLA = True
except ImportError:
    xm = None
    IS_XLA = False

if IS_XLA:
    # transformers 4.17 still calls the pre-PJRT names, removed in torch_xla 2.9
    if not hasattr(xm, "xrt_world_size"):
        xm.xrt_world_size = xr.world_size
    if not hasattr(xm, "get_ordinal"):
        xm.get_ordinal = xr.global_ordinal


def mark_step():
    if IS_XLA:
        xm.mark_step()
