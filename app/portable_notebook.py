"""Generate portable Colab/Kaggle notebooks from the selected local command."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepositoryExportStatus:
    ready: bool
    message: str
    repository_url: str = ""
    commit: str = ""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


def repository_export_status(root: Path) -> RepositoryExportStatus:
    try:
        repository_url = _git(root, "remote", "get-url", "origin")
    except (subprocess.SubprocessError, OSError):
        return RepositoryExportStatus(False, "Notebook export için public bir origin remote gerekli.")
    if not repository_url.startswith("https://github.com/"):
        return RepositoryExportStatus(False, "origin public bir HTTPS GitHub adresi olmalı.")
    if _git(root, "status", "--porcelain"):
        return RepositoryExportStatus(False, "Notebook export için çalışma ağacı temiz olmalı.")
    try:
        with urllib.request.urlopen(repository_url.removesuffix(".git"), timeout=5) as response:
            if response.status != 200:
                raise OSError
        commit = _git(root, "rev-parse", "HEAD")
        branch = _git(root, "branch", "--show-current")
        remote = _git(root, "ls-remote", "--heads", "origin", branch).split()
    except (subprocess.SubprocessError, OSError):
        return RepositoryExportStatus(False, "origin erişilebilir ve public olmalı.")
    if not branch or not remote or remote[0] != commit:
        return RepositoryExportStatus(False, "Yerel HEAD önce origin üzerindeki aktif branch'e push edilmeli.")
    return RepositoryExportStatus(True, "Notebook export hazır.", repository_url, commit)


def _cell(cell_type: str, source: str) -> dict:
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _dataset_setup(dataset: str) -> str:
    if dataset == "appleleaf9":
        return """from pathlib import Path
import shutil, subprocess
target = Path('datasets/AppleLeaf9/raw')
if not target.exists():
    subprocess.run(['git', 'clone', 'https://github.com/JasonYangCode/AppleLeaf9', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '0af37f2fe1a1ce068d03e607ff5c7fd3b334f1c2'], check=True)
    shutil.rmtree(target / '.git')
"""
    if dataset == "plantvillage":
        return """from pathlib import Path
import subprocess
target = Path('datasets/PlantVillage')
if not target.exists():
    subprocess.run(['git', 'clone', 'https://github.com/spMohanty/PlantVillage-Dataset', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '7f7ecc7e1eaca78107e3affe7cb5abd9427e139a'], check=True)
"""
    if dataset == "plantpathology2021":
        return """from pathlib import Path
import subprocess, sys, zipfile
target = Path('datasets/PlantPathology2021')
target.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)
subprocess.run(['kaggle', 'competitions', 'download', '-c', 'plant-pathology-2021-fgvc8', '-p', str(target)], check=True)
archives = list(target.glob('*.zip'))
while archives:
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
        archive.unlink()
    archives = list(target.glob('*.zip'))
"""
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_portable_notebook(
    *,
    repository_url: str,
    source_commit: str,
    stage: str,
    dataset: str,
    model: str,
    command: list[str],
) -> bytes:
    run_command = ["python", *command[1:]]
    clone = f"""import os, subprocess, sys
subprocess.run(['git', 'clone', {repository_url!r}, 'MasterThesis'], check=True)
os.chdir('MasterThesis')
subprocess.run(['git', 'checkout', {source_commit!r}], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
"""
    train = f"""import subprocess, sys
command = {run_command!r}
command[0] = sys.executable
subprocess.run(command, check=True)
"""
    package = f"""import subprocess, sys
subprocess.run([
    sys.executable, 'scripts/package_external_run.py',
    '--stage', {stage!r}, '--dataset', {dataset!r}, '--model', {model!r},
    '--source-commit', {source_commit!r}, '--output', '/content' if 'COLAB_RELEASE_TAG' in __import__('os').environ else '.',
], check=True)
"""
    notebook = {
        "cells": [
            _cell("markdown", f"# {dataset} · {model}\nGenerated from commit `{source_commit}`."),
            _cell("code", clone),
            _cell("markdown", "## Dataset preparation\nKaggle verisi için API credentials gerekir."),
            _cell("code", _dataset_setup(dataset)),
            _cell("markdown", "## Training"),
            _cell("code", train),
            _cell("markdown", "## Export result bundle\nZIP ve checkpoint ayrı oluşturulur."),
            _cell("code", package),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=2).encode("utf-8")


def render_notebook_download(
    *, stage: str, dataset: str, model: str, command: list[str], key: str
) -> None:
    """Render a guarded Streamlit notebook download control."""
    import streamlit as st

    root = Path(__file__).resolve().parents[1]
    status = repository_export_status(root)
    if "--dry-run" in command:
        status = RepositoryExportStatus(False, "Dry-run sonuç paketi üretmez; notebook için dry-run'ı kapatın.")
    st.subheader("Colab / Kaggle Notebook")
    st.caption(status.message)
    payload = b""
    if status.ready:
        payload = build_portable_notebook(
            repository_url=status.repository_url,
            source_commit=status.commit,
            stage=stage,
            dataset=dataset,
            model=model,
            command=command,
        )
    st.download_button(
        "Portable .ipynb indir",
        data=payload,
        file_name=f"{stage}__{dataset}__{model}.ipynb",
        mime="application/x-ipynb+json",
        disabled=not status.ready,
        key=key,
    )
