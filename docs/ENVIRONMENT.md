# Environment Setup

This project uses `pyproject.toml` for the Python package and general project dependencies.

PyTorch is intentionally not installed by the base `pyproject.toml` dependencies because CUDA-enabled PyTorch must be selected through the correct package channel. Installing `torch` with plain `pip install -e .` or from the wrong conda channel can result in a CPU-only build.

## Recommended CUDA Environment on Windows

Create and activate the environment:

```bash
mamba create -n masterthesis python=3.11 -y
mamba activate masterthesis
```

Install CUDA-enabled PyTorch from the official PyTorch CUDA wheel index:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then install the project package and remaining dependencies:

```bash
pip install -e .
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected:

```text
torch.cuda.is_available() -> True
torch.version.cuda -> 12.4
```

## If CPU PyTorch Was Installed Accidentally

If `torch.cuda.is_available()` is `False` and `torch.version.cuda` is `None`, remove the CPU builds first:

```bash
python -m pip uninstall torch torchvision torchaudio -y
mamba remove pytorch torchvision torchaudio cpuonly libtorch -y
```

Then reinstall CUDA PyTorch using the official PyTorch CUDA wheel index:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Finally reinstall the project:

```bash
pip install -e .
```

## CPU-Only Fallback

For CPU-only development:

```bash
pip install -e ".[torch-cpu]"
```
