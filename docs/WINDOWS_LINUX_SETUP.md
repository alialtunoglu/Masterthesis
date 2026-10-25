# Windows, Linux and WSL2

Use Python 3.10–3.12 and an environment with the project dependencies already installed. Install a compatible PyTorch/torchvision build separately for the hardware. Do not replace a working CUDA installation to repair imports.

Inside the activated environment and project checkout:

```console
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The editable install registers the source checkout; repeat it in a new environment or after moving the checkout. It does not install missing dependencies. `pytest` belongs to the optional development dependencies. No shell `PYTHONPATH` setting is required. Do not use a local file named `src.py` or `app.py` that shadows the installed packages.

Canonical imports are `src.training`, `src.datasets`, `src.models`, etc., and `app.*`. CLI module entry points work from a different current directory after installation:

```console
python -m src.training.train_baseline --help
python -m src.training.train_teacher --help
python -m src.training.train_kd --help
python -m src.training.train_multi_kd --help
python -m app.queue_worker --help
```

`--help` never starts training. Running the queue worker without `--help` can start queued experiments. Opening Streamlit alone does not authorize a new training run.

The workspace defaults to the editable checkout, not the caller's current directory. For a separately installed package or another workspace, explicitly set `MASTERTHESIS_PROJECT_ROOT` to the workspace directory. Child processes receive this workspace root, not an injected import path. The workspace still needs its configs, dataset registry, data and checkpoints; installing Python code does not supply experiment inputs.

Relative SQLite MLflow paths resolve against that workspace. Existing databases, artifact URIs and historical records are not rewritten. Use the existing bundle import/export workflow to transfer experiments; copying a SQLite file does not relocate absolute artifact URIs.

Device selection supports `auto`, `cpu`, `cuda`. `auto` retains CUDA-first, otherwise CPU behavior; explicit unavailable CUDA fails clearly. MPS/macOS is not supported in this change. NVIDIA inventory uses `nvidia-smi`; CPU operation does not require it. Linux/WSL file locks use `fcntl`, Windows locks use `msvcrt`; subprocesses use the same Python executable as the launcher. Linux DataLoader defaults and all training parameters remain unchanged; the earlier lab-specific worker workaround is not applied globally.

Run Windows and Linux acceptance tests in their own environments. A WSL test is a Linux test, not proof of native Windows support. Hardware-specific full training is not part of the import test suite. Changing machines does not guarantee bitwise-identical numerical training results.
