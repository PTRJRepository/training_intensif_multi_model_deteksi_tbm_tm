"""Runtime helpers for selecting the best training device on this machine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TrainingRuntime:
    """Resolved runtime settings for a training run."""

    requested_device: str
    resolved_device: str
    requested_amp: Optional[bool]
    amp: bool
    workers: int
    gpu_name: Optional[str]
    total_vram_gb: Optional[float]
    cuda_available: bool


def _default_workers() -> int:
    """Pick a safe worker count for Windows laptops."""
    cpu_count = os.cpu_count() or 4
    return max(2, min(6, cpu_count // 2))


def _normalize_device(device: Optional[str]) -> str:
    if not device:
        return "auto"
    return str(device).strip().lower()


def resolve_training_runtime(
    requested_device: Optional[str] = "auto",
    requested_amp: Optional[bool] = None,
    requested_workers: Optional[int] = None,
) -> TrainingRuntime:
    """Resolve device, AMP, and worker defaults from local CUDA availability."""
    normalized_device = _normalize_device(requested_device)
    cuda_available = torch.cuda.is_available()

    if normalized_device in {"", "auto", "cuda", "cuda:0", "gpu"}:
        resolved_device = "0" if cuda_available else "cpu"
    elif normalized_device == "cpu":
        resolved_device = "cpu"
    else:
        resolved_device = requested_device or "cpu"

    gpu_name = None
    total_vram_gb = None
    if resolved_device != "cpu" and cuda_available:
        gpu_index = 0
        if str(resolved_device).isdigit():
            gpu_index = int(str(resolved_device))
        elif str(resolved_device).startswith("cuda:"):
            gpu_index = int(str(resolved_device).split(":", maxsplit=1)[1])

        props = torch.cuda.get_device_properties(gpu_index)
        gpu_name = props.name
        total_vram_gb = round(props.total_memory / (1024**3), 2)

    if requested_amp is None:
        amp = resolved_device != "cpu"
    else:
        amp = bool(requested_amp and resolved_device != "cpu")

    workers = requested_workers if requested_workers is not None else _default_workers()
    workers = max(0, workers)
    if resolved_device != "cpu":
        workers = min(workers, 6)

    return TrainingRuntime(
        requested_device=normalized_device,
        resolved_device=str(resolved_device),
        requested_amp=requested_amp,
        amp=amp,
        workers=workers,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        cuda_available=cuda_available,
    )


def print_runtime_summary(runtime: TrainingRuntime) -> None:
    """Print a short runtime summary before training starts."""
    print("\n[RUNTIME]")
    print(f"  CUDA available: {runtime.cuda_available}")
    print(f"  Requested device: {runtime.requested_device}")
    print(f"  Resolved device: {runtime.resolved_device}")
    if runtime.gpu_name:
        print(f"  GPU: {runtime.gpu_name} ({runtime.total_vram_gb} GB VRAM)")
    print(f"  AMP: {runtime.amp}")
    print(f"  Workers: {runtime.workers}")
