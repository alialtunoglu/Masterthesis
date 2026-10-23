# Baseline Role and Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate training approach, student/teacher role and CNN/ViT family across experiment creation and monitoring; backfill legacy job metadata and support DeiT-Tiny/Small student baselines.

**Architecture:** Preserve routing/storage `stage` values (`baseline`, `teacher_cnn`, `teacher_vision_transformer`, existing KD stages). Add orthogonal `training_method`, `model_role`, `architecture_family`. One lightweight model catalog and taxonomy helper feed job metadata, labels and filters. Existing training/results/checkpoints remain in place.

**Tech Stack:** Streamlit, Python, installed timm1.0.28, masterthesis conda environment.

**Spec:** User-approved hierarchy Baseline → Student/Teacher → CNN/ViT; user explicitly chose DeiT-Tiny and DeiT-Small and unchanged KD constraints.

## Global Constraints
- No real training, weight downloads, worker start or deletion.
- Preserve dirty working tree; no commit/merge/push.
- Back up original job bytes before metadata updates; unknown/conflicting jobs are not guessed.
- Do not rewrite historical CSV metrics, checkpoints or MLflow run identities.
- Student ViT uses the existing AdamW/no-scheduler student baseline protocol, fixed224 input; no hidden teacher-only training flags.

## Review Focus
- Legacy jobs without stage must appear under student baseline only when command establishes their pipeline.
- Teacher baseline must never become a student run or change artifact paths.
- Architecture must come from model registry, not the student/teacher role.
- New student ViT notebooks must import through baseline allowlist and retain class/split validation.
- Migration must be idempotent, backed up, and preserve unrelated fields.

### Task 1: Catalog and classification
- [x] Add failing tests for CNN/ViT classification, all five legacy stages, unknown/conflicting commands, and missing-stage baseline.
- [x] Create `src/models/model_catalog.py` with lightweight role/family sets; keep factory public constants backward-compatible.
- [x] Create `src/tracking/experiment_taxonomy.py`: classify/enrich job metadata without changing known stage values.
- [x] Integrate enqueue/resume metadata, labels and read-only legacy fallback; test actual queued JSON in a temporary directory.

### Task 2: Student ViT baseline
- [x] Add failing model/config tests for `deit_tiny_patch16_224` and `deit_small_patch16_224` with six-class output and no pretrained downloads.
- [x] Register installed timm models, add six dataset-specific JSONs under `configs/baseline/vision_transformers/<dataset>/`.
- [x] Persist taxonomy in new baseline/teacher configs, result rows and MLflow params; baseline CLI accepts student models only.
- [x] Extend external-import model allowlist; preserve original result paths and bundle validation.

### Task 3: Interface
- [x] Add AppTests for Baseline → role → family selection, Student ViT configs, job role/family filters and legacy baseline visibility.
- [x] Update `app/pages/0_New_Experiment.py`, add student-ViT legacy-compatible page, update shared launcher to separate fixed ViT input from teacher-specific options.
- [x] Update labels and results role/family filtering while retaining old routes, artifacts/removal/import.
- [x] Keep all training confirmations and notebook/recovery paths; verify existing AppTests.

### Task 4: Migration and verification
- [x] Add/test `scripts/backfill_job_metadata.py` with dry-run default, explicit `--apply`, byte backups and a manifest.
- [x] Run dry-run, inspect proposed counts, then apply to user-authorized existing job records.
- [x] Verify no stage missing among recognized jobs, byte-preserving backups, unchanged non-taxonomy fields and idempotent second dry-run.
- [x] Run full unittest suite in masterthesis; browser desktop/mobile review without executing jobs; independent final review.

## File map
Catalog/factory; tracking taxonomy; job runner; baseline/teacher CLI; external importer; standard launcher; overview/monitor/results and new-experiment page; six configs; migration script; focused taxonomy/migration/UI/model tests. No KD strategy/trainer changes.
