"""Device policy shared by training entry points; no optional accelerator fallback."""
import torch


def select_device(value: str) -> torch.device:
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)
