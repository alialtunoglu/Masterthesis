# Portable Notebook Dataset Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generated Plant Pathology notebooks prepare their dataset safely in Kaggle, Google Colab, and local Jupyter without embedding credentials.

**Architecture:** Keep `app.portable_notebook._dataset_setup()` as the single dataset dispatcher and emit one self-contained Plant Pathology setup cell. The cell reuses prepared data first, then Kaggle-mounted input, then authenticated Kaggle CLI; Colab may request a temporary `kaggle.json` upload.

**Tech Stack:** Python standard library, Kaggle CLI, `unittest`, Jupyter notebook JSON.

## Global Constraints

- Do not add a Python dependency to the project.
- Never serialize, print, commit, or package Kaggle credential contents.
- Do not download datasets or start training during tests.
- Preserve the pinned AppleLeaf9 and PlantVillage GitHub sources and commits.
- Keep the change within `app/portable_notebook.py` and `tests/test_portable_notebook.py`.

---

### Task 1: Generate an environment-aware Plant Pathology setup cell

**Files:**
- Modify: `tests/test_portable_notebook.py`
- Modify: `app/portable_notebook.py:56-139`

**Interfaces:**
- Consumes: `_dataset_setup(dataset: str) -> str` and `build_portable_notebook(*, repository_url: str, source_commit: str, stage: str, dataset: str, model: str, command: list[str]) -> bytes`.
- Produces: a self-contained Plant Pathology code cell that resolves `datasets/PlantPathology2021/train.csv` and `datasets/PlantPathology2021/train_images`.

- [ ] **Step 1: Write failing notebook-source tests**

Add tests which build a Plant Pathology notebook and assert that its generated source:

```python
def _notebook_source(payload: bytes) -> str:
    notebook = json.loads(payload)
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_plant_pathology_notebook_handles_kaggle_colab_and_local_auth(self):
    source = _notebook_source(build_portable_notebook(
        repository_url="https://github.com/example/project.git",
        source_commit="abc123",
        stage="teacher",
        dataset="plantpathology2021",
        model="resnet50",
        command=["python", "src/training/train_teacher.py", "--config", "config.json"],
    ))
    self.assertIn("/kaggle/input/plant-pathology-2021-fgvc8", source)
    self.assertIn("google.colab", source)
    self.assertIn("files.upload()", source)
    self.assertIn("KAGGLE_API_TOKEN", source)
    self.assertIn(".kaggle/access_token", source)
    self.assertIn(".kaggle/kaggle.json", source)
    self.assertIn("train.csv", source)
    self.assertIn("train_images", source)
    self.assertNotIn("KAGGLE_KEY =", source)
```

Add a second test asserting the setup checks prepared data before the `kaggle competitions download` statement, and retain assertions for the two pinned GitHub datasets.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
conda run -n masterthesis python -m unittest tests.test_portable_notebook -v
```

Expected: the new Plant Pathology assertions fail because the source only invokes Kaggle CLI.

- [ ] **Step 3: Emit the minimal environment-aware setup source**

Update only the `plantpathology2021` branch of `_dataset_setup()` so the emitted cell:

```python
target = Path("datasets/PlantPathology2021")
required = (target / "train.csv", target / "train_images")
mounted = Path("/kaggle/input/plant-pathology-2021-fgvc8")

if not all(path.exists() for path in required):
    if all((mounted / path.name).exists() for path in required):
        target.mkdir(parents=True, exist_ok=True)
        for destination in required:
            destination.symlink_to(
                mounted / destination.name,
                target_is_directory=(mounted / destination.name).is_dir(),
            )
    else:
        credential_files = (
            Path.home() / ".kaggle" / "access_token",
            Path.home() / ".kaggle" / "kaggle.json",
        )
        authenticated = bool(os.environ.get("KAGGLE_API_TOKEN")) or any(
            path.exists() for path in credential_files
        )
        if not authenticated and importlib.util.find_spec("google.colab"):
            from google.colab import files
            uploaded = files.upload()
            if "kaggle.json" not in uploaded:
                raise RuntimeError("kaggle.json yüklenmedi.")
            credential_files[1].parent.mkdir(parents=True, exist_ok=True)
            credential_files[1].write_bytes(uploaded["kaggle.json"])
            credential_files[1].chmod(0o600)
            authenticated = True
        if not authenticated:
            raise RuntimeError("Kaggle kimlik doğrulaması gerekli.")

if not all(path.exists() for path in required):
    raise FileNotFoundError("Plant Pathology train.csv veya train_images bulunamadı.")
```

Use `Path.symlink_to()` for Kaggle-mounted files, `shutil.which("kaggle")` to avoid unnecessary installs, `importlib.util.find_spec("google.colab")` for Colab detection, and the supported credential locations from the approved design. Catch `subprocess.CalledProcessError` only to raise an actionable message that distinguishes authentication from competition authorization while preserving the original command output.

- [ ] **Step 4: Update the dataset-preparation markdown copy**

Replace the unconditional “Kaggle verisi için API credentials gerekir.” text with a statement explaining that Kaggle-mounted input is reused and Colab requests credentials only when download is necessary.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
conda run -n masterthesis python -m unittest tests.test_portable_notebook -v
```

Expected: all portable notebook tests pass.

- [ ] **Step 6: Run the relevant regression suite**

Run:

```powershell
conda run -n masterthesis python -m unittest discover -s tests -p "test_*notebook*.py" -v
```

Expected: all discovered notebook tests pass without network or dataset access.

- [ ] **Step 7: Validate a generated notebook artifact without executing it**

Run a short Python command that calls `build_portable_notebook` with a Plant Pathology teacher command, parses the returned bytes with `json.loads`, and compiles every generated code cell with `compile(source, "<notebook-cell>", "exec")`.

Expected: valid notebook JSON and no `SyntaxError`.

- [ ] **Step 8: Commit the implementation**

```powershell
git add app/portable_notebook.py tests/test_portable_notebook.py
git commit -m "Fix portable Plant Pathology dataset setup"
```
