"""Pick the compute device: CUDA when available (e.g. Colab), otherwise CPU."""

from __future__ import annotations


def get_device() -> str:
    """Return "cuda" if a CUDA GPU is usable, else "cpu". Works without torch installed."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
