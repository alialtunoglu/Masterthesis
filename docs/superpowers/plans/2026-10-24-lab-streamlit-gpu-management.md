# Laboratory Streamlit GPU Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Manage the laboratory GPU queue and notebook attempts in Streamlit, and expose verified folders for manual result transfer.

**Architecture:** Extend the native queue with physical GPU selection and a lifetime OS lock shared with the notebook runner. Adapt notebook states into read-only jobs, reuse native progress parsing and the existing result bundle exporter/importer. No new service or dependency.

**Tech Stack:** Python 3.11, psutil, standard-library process/file locks, Streamlit 1.59.2, unittest/AppTest, existing masterthesis conda environment.

**Spec:** `docs/superpowers/specs/2026-10-24-lab-streamlit-gpu-management-design.md`

## Global Constraints

- One process per physical GPU; existing global parallelism limit remains an additional limit.
- Share `lab_state/gpu_<index>.lock` with notebook runners, without changing running jobs.
- No training, queue promotion, dataset mutation, remote access, Git publishing, or new environment during implementation.
- Preserve existing research settings and user changes; RustDesk/manual transfer remain the workflow.

## Review Focus

- Explicit GPU absent or busy: reject invalid selection, keep busy jobs queued; never spill onto another requested GPU.
- Notebook PID reused or inaccessible: show an uncertain/interrupted state; do not offer unsafe termination.
- Existing notebook holds GPU lock before state is complete: lock still blocks admission.
- Concurrent/failed completion: never export another run or present missing checkpoints as a complete transfer.
- Local Windows without NVIDIA tools: keep CPU/legacy queue and tests usable without initializing CUDA.

### Task 1: GPU-aware queue admission and process ownership

**Files:** Create `app/gpu_runtime.py`, `app/job_process.py`, `tests/test_gpu_scheduling.py`; modify `app/experiment_runner.py`.

**Interfaces:** `discover_gpus() -> list[dict]`; `gpu_lock(root, gpu_id)` context manager; `gpu_is_available(root, gpu_id) -> bool`; `choose_gpu(request, inventory, running_jobs, root) -> str | None`; queued metadata `gpu_requested`, resolved `gpu_id`. The process wrapper runs the original command while holding the lock and emits an explicit exit record.

- [x] Write tests for two-GPU assignment, unavailable explicit GPU, busy GPU queue retention, native/notebook lock contention and CPU fallback; verify RED.
- [x] Implement inventory without importing torch, cross-platform locking and shared wrapper lifetime; integrate enqueue/admission/status handling.
- [x] Run focused tests plus existing queue/recovery tests; verify GREEN and record the result.

### Task 2: Notebook adapter and completed transfer service

**Files:** Create `app/lab_runs.py`, `app/run_transfer.py`, `tests/test_lab_run_management.py`; modify `app/experiment_runner.py`, `app/kd_launcher.py`.

**Interfaces:** `list_lab_jobs(root) -> list[dict]` returns virtual jobs with `execution_origin=lab_notebook`, immutable config/log references and process identity. `prepare_run_transfer(root, job) -> dict` returns verified export paths for the exact MLflow run only. Portable teacher candidates retain source IDs and checkpoint hashes.

- [x] Write tests for dry/full attempts, malformed state, dead/reused PID, correct external stop file, completed export and duplicate export reuse; verify RED.
- [x] Implement read-only adaptation, admission visibility, safe-stop routing and validated teacher fallback. Reuse the existing bundle service and exact run matching.
- [x] Run focused tests and existing external-bundle/KD/recovery tests; verify GREEN.

### Task 3: Streamlit controls, verification and laboratory guide

**Files:** Create `app/gpu_view.py`, `tests/test_lab_streamlit_ui.py`, `docs/LAB_STREAMLIT.md`; modify standard/KD launchers, Overview, Job Monitor, Settings and shared UI labels.

**Interfaces:** `render_gpu_selection(key) -> str` supplies launcher metadata. Monitor displays GPU, origin, dry/full mode, preparation/epoch state and last log update; virtual jobs disable unsafe native removal/resume/force-stop actions. A confirmed export action displays the complete folder and exact local inbox destination.

- [x] Write AppTests for GPU selection, common notebook visibility, safe controls and transfer availability; verify RED.
- [x] Implement native Streamlit controls and concise manual transfer instructions; no custom CSS or remote service.
- [x] Run focused AppTests, the full unittest suite with the established 8 MiB Windows test-thread stack, compile and diff checks. Review the full change and document actual limitations.
- [x] Hand off changed files/branch and safe deployment steps without stopping or starting a real job.

## Verification

- Full masterthesis suite: **263 tests passed / 77.246 seconds**.
- New tests: 9 GPU scheduling/process tests, 13 notebook/transfer tests, 6 Streamlit AppTests.
- Syntax: 37 application/test Python files parsed. Scoped diff whitespace check clean.
- Real two-GPU WSL training and live desktop/mobile browser review were not performed.
- No training or worker started, no existing dataset/checkpoint/results changed by this feature.
