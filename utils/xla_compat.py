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

if IS_XLA:
    # the CUDA reference (torch 1.10) runs float32 matmuls in TF32 by default; XLA's default on TPU is a single
    # bf16 pass, "high" (bf16_3x) is the closest setting at or above TF32 accuracy
    torch_xla.backends.set_mat_mul_precision("high")


def xla_log_fields():
    """Compile/execute counts and aten:: CPU-fallback counters, logged with the training loss."""
    if not IS_XLA:
        return {}
    import torch_xla.debug.metrics as met
    compile_data = met.metric_data("CompileTime")
    exec_data = met.metric_data("ExecuteTime")
    fallbacks = {name: met.counter_value(name) for name in met.counter_names() if name.startswith("aten::")}
    return {"xla_compiles": compile_data[0] if compile_data else 0,
            "xla_executions": exec_data[0] if exec_data else 0,
            "xla_aten_fallbacks": fallbacks}


def dump_xla_metrics(path):
    if IS_XLA:
        import torch_xla.debug.metrics as met
        with open(path, "w") as f:
            f.write(met.metrics_report())
