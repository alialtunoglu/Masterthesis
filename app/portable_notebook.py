"""Generate portable Colab/Kaggle notebooks from the selected local command."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import streamlit as st


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
    try:
        pending_changes = _git(root, "status", "--porcelain")
    except (subprocess.SubprocessError, OSError):
        return RepositoryExportStatus(False, "Git deposu okunamadı; repository durumu doğrulanamıyor.")
    if pending_changes:
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


@st.cache_data(ttl=30, show_spinner=False)
def cached_export_status(root: Path) -> RepositoryExportStatus:
    """Probe the repository at most once per TTL.

    Streamlit reruns the page on every widget change, and the uncached probe
    shells out to git four times and reaches the network twice, so without this
    each keystroke costs the better part of a second.
    """
    return repository_export_status(root)


def _cell(cell_type: str, source: str) -> dict:
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _output_resolver_source() -> str:
    """Source defining the pure bundle-output resolver used by the notebook."""
    return """from pathlib import Path


KAGGLE_ENVIRONMENT_NAMES = (
    'KAGGLE_KERNEL_RUN_TYPE', 'KAGGLE_URL_BASE', 'KAGGLE_DATA_PROXY_URL',
)


def resolve_bundle_output(environment, mount, drive_root=Path('/content/drive')):
    \"\"\"Return (output_directory, message) for the result bundle.\"\"\"
    if any(environment.get(name) for name in KAGGLE_ENVIRONMENT_NAMES):
        return '/kaggle/working', (
            'Kaggle: bundle /kaggle/working altina yazilacak. Kalici olmasi icin '
            'defteri Save Version ile calistirin.'
        )
    if 'COLAB_RELEASE_TAG' in environment:
        try:
            mount('/content/drive')
        except Exception as error:
            return '/content', (
                'UYARI: Google Drive baglanamadi (' + str(error) + '). Bundle '
                '/content altina yazilacak ve calisma zamani kapaninca silinecek; '
                'oturum bitmeden indirin.'
            )
        target = drive_root / 'MyDrive' / 'MasterThesis' / 'bundles'
        target.mkdir(parents=True, exist_ok=True)
        return str(target), 'Drive baglandi. Bundle su dizine yazilacak: ' + str(target)
    return '.', 'Yerel calisma dizinine yazilacak.'
"""


def _output_setup_source() -> str:
    """Source that resolves BUNDLE_OUTPUT from scratch.

    Emitted into every cell that needs the destination rather than shared
    through the kernel: training runs unattended for hours and a Colab
    reconnect in the meantime wipes the variable while leaving the earlier
    cell's output on screen.
    """
    return _output_resolver_source() + """

import os

try:
    from google.colab import drive as _colab_drive
except ImportError:
    def _mount(mountpoint):
        raise RuntimeError('google.colab bu ortamda yok.')
else:
    _mount = _colab_drive.mount

BUNDLE_OUTPUT, _output_message = resolve_bundle_output(os.environ, _mount)
print(_output_message)
"""


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
import os, shutil, subprocess, sys, zipfile
target = Path('datasets/PlantPathology2021')
target.mkdir(parents=True, exist_ok=True)
required = (target / 'train.csv', target / 'train_images')
mounted = Path('/kaggle/input/plant-pathology-2021-fgvc8')

if not all(path.exists() for path in required):
    if all((mounted / path.name).exists() for path in required):
        for destination in required:
            if not destination.exists():
                source = mounted / destination.name
                destination.symlink_to(source, target_is_directory=source.is_dir())
        print(f'Kaggle input kullanılıyor: {mounted}')
    else:
        is_kaggle_notebook = any(
            os.environ.get(name)
            for name in ('KAGGLE_KERNEL_RUN_TYPE', 'KAGGLE_URL_BASE', 'KAGGLE_DATA_PROXY_URL')
        )
        if is_kaggle_notebook:
            raise RuntimeError(
                'Kaggle Notebook içinde Add Input ile Plant Pathology 2021 - FGVC8 '
                'yarışma verisini ekleyip hücreyi yeniden çalıştırın.'
            )
        credential_files = (
            Path.home() / '.kaggle' / 'access_token',
            Path.home() / '.kaggle' / 'kaggle.json',
        )
        authenticated = bool(os.environ.get('KAGGLE_API_TOKEN')) or any(
            path.exists() for path in credential_files
        )
        if not authenticated:
            try:
                from google.colab import files
            except ImportError:
                files = None
            if files is None:
                raise RuntimeError(
                    'Kaggle kimlik doğrulaması bulunamadı. KAGGLE_API_TOKEN ayarlayın '
                    'veya ~/.kaggle/access_token ya da ~/.kaggle/kaggle.json ekleyin.'
                )
            print('Kaggle API credentials gerekli. Açılan pencereden kaggle.json yükleyin.')
            uploaded = files.upload()
            if 'kaggle.json' not in uploaded:
                raise RuntimeError('kaggle.json yüklenmedi; dataset indirme işlemi durduruldu.')
            credential = credential_files[1]
            credential.parent.mkdir(parents=True, exist_ok=True)
            credential.write_bytes(uploaded['kaggle.json'])
            credential.chmod(0o600)

        if shutil.which('kaggle') is None:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)
        command = [
            'kaggle', 'competitions', 'download', '-c',
            'plant-pathology-2021-fgvc8', '-p', str(target),
        ]
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode:
            output = (result.stdout or '').lower()
            if 'authenticate' in output or 'unauthorized' in output or '401' in output:
                hint = 'Kaggle API token geçersiz veya eksik.'
            elif 'forbidden' in output or '403' in output or 'rule' in output:
                hint = (
                    'Plant Pathology 2021 yarışma kurallarını aynı Kaggle hesabıyla '
                    'kabul edin.'
                )
            else:
                hint = 'Kaggle indirmesi başarısız oldu; yukarıdaki CLI çıktısını inceleyin.'
            raise RuntimeError(hint)

        archives = list(target.glob('*.zip'))
        while archives:
            for archive in archives:
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(target)
                archive.unlink()
            archives = list(target.glob('*.zip'))

if not all(path.exists() for path in required):
    missing = ', '.join(str(path) for path in required if not path.exists())
    raise FileNotFoundError(f'Plant Pathology dataset eksik: {missing}')
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
    train = f"""from pathlib import Path
import os, subprocess, sys
command = {run_command!r}
command[0] = sys.executable
environment = os.environ.copy()
project_src = str(Path('src').resolve())
existing_pythonpath = environment.get('PYTHONPATH')
environment['PYTHONPATH'] = (
    project_src if not existing_pythonpath else project_src + os.pathsep + existing_pythonpath
)
environment['PYTHONUNBUFFERED'] = '1'
process = subprocess.Popen(
    command,
    env=environment,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
)
assert process.stdout is not None
for line in process.stdout:
    print(line, end='', flush=True)
returncode = process.wait()
if returncode:
    raise RuntimeError(f'Eğitim başarısız oldu. Exit code: {{returncode}}')
"""
    output_setup = _output_setup_source()
    package = _output_setup_source() + f"""
import subprocess, sys

if not Path('scripts/package_external_run.py').exists():
    # A kernel restart loses the clone cell's chdir but keeps the disk.
    for _candidate in (Path.cwd() / 'MasterThesis', Path('/content/MasterThesis')):
        if (_candidate / 'scripts' / 'package_external_run.py').exists():
            os.chdir(_candidate)
            print('Calisma dizini geri yuklendi:', _candidate)
            break
    else:
        raise RuntimeError(
            'Repo bulunamadi. Colab calisma zamani geri donusturulmus olabilir; '
            'bu durumda egitim ciktilari da silinmistir ve notebook bastan '
            'calistirilmalidir.'
        )
subprocess.run([
    sys.executable, 'scripts/package_external_run.py',
    '--stage', {stage!r}, '--dataset', {dataset!r}, '--model', {model!r},
    '--source-commit', {source_commit!r}, '--output', BUNDLE_OUTPUT,
], check=True)
print('Bundle dizini:', BUNDLE_OUTPUT)
"""
    notebook = {
        "cells": [
            _cell("markdown", f"# {dataset} · {model}\nGenerated from commit `{source_commit}`."),
            _cell("code", clone),
            _cell(
                "markdown",
                "## Result destination\n"
                "Sonuç bundle'ının nereye yazılacağı burada belirlenir. "
                "Colab'da Drive bağlama onayını **şimdi** verin; eğitim bittiğinde "
                "bilgisayar başında olmanız gerekmez. Kaggle'da bundle "
                "`/kaggle/working` altına yazılır, kalıcı olması için defteri "
                "Save Version ile çalıştırın.",
            ),
            _cell("code", output_setup),
            _cell(
                "markdown",
                "## Dataset preparation\nKaggle bağlı input varsa doğrudan kullanılır; "
                "indirme gerekirse Colab güvenli biçimde `kaggle.json` yüklemenizi ister.",
            ),
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
    root = Path(__file__).resolve().parents[1]
    status = cached_export_status(root)
    if "--dry-run" in command:
        status = RepositoryExportStatus(False, "Dry-run sonuç paketi üretmez; notebook için dry-run'ı kapatın.")
    st.subheader("Colab / Kaggle Notebook")
    caption_column, recheck_column = st.columns([4, 1])
    caption_column.caption(status.message)
    if recheck_column.button("Yeniden kontrol et", key=f"{key}_recheck"):
        cached_export_status.clear()
        st.rerun()
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
