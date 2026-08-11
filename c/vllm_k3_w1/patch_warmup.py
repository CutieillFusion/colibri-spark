"""Persist worker warmup boundaries outside Ray and Docker logs."""

import functools
import os
import socket
from datetime import datetime, timezone


def _event(phase, state):
    try:
        import torch

        log_dir = os.environ.get("K3_LOG_DIR", "/k3w1/logs")
        os.makedirs(log_dir, exist_ok=True)
        path = f"{log_dir}/warmup-{socket.gethostname()}-{os.getpid()}.log"
        available = "unknown"
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    available = line.split(":", 1)[1].strip()
                    break
        allocated = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        stamp = datetime.now(timezone.utc).isoformat()
        with open(path, "a", buffering=1) as output:
            output.write(
                f"{stamp} phase={phase} state={state} "
                f"mem_available={available} cuda_allocated_gib={allocated:.3f} "
                f"cuda_reserved_gib={reserved:.3f}\n"
            )
    except OSError:
        pass


def _wrap(function, phase):
    if getattr(function, "_k3_warmup_trace", False):
        return function

    @functools.wraps(function)
    def traced(*args, **kwargs):
        _event(phase, "start")
        try:
            result = function(*args, **kwargs)
        except BaseException:
            _event(phase, "exception")
            raise
        _event(phase, "end")
        return result

    traced._k3_warmup_trace = True
    return traced


def apply():
    """Install idempotent phase tracing around GPU worker warmup."""
    from vllm.v1.worker import gpu_worker

    gpu_worker.kernel_warmup = _wrap(gpu_worker.kernel_warmup, "kernel_warmup")
    gpu_worker.warmup_kernels = _wrap(gpu_worker.warmup_kernels, "scheduler_warmup")
    worker = gpu_worker.Worker
    worker.compile_or_warm_up_model = _wrap(
        worker.compile_or_warm_up_model, "compile_or_warm_up_model"
    )
