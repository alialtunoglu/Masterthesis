# Dependent Job Filters Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. Checkboxes track completion.

**Goal:** Dataset-first filters show only relevant baseline/single-KD/multi-KD fields and prevent impossible combinations.
**Architecture:** Extract one small native Streamlit filter renderer into app/job_filters.py. Each field derives options from the preceding subset; strict training services and job files are unchanged.
**Tech Stack:** Existing Streamlit1.59, masterthesis conda Python, unittest/AppTest.
**Spec:** User-approved chat design: dataset → method → type-specific fields → status; disabled unavailable method buttons; compatible dropdown options only; invalid selections reset to Tümü; clear all filters.

## Global Constraints
- No training, worker start, artifact/job writes, package installation or publishing.
- Preserve current dirty branch and existing identity/recovery behavior.
- Baseline: role → family → model. Single KD: student → teacher → KD type.
- Multi KD: student → teacher count → exact unordered teacher combination → KD type → aggregation.
- Include finished/failed/stopped/cancelled/running/queued jobs in available choices.
- Preserve unknown/conflicting records under Tümü; do not guess their training method.

## Review Focus
- Dataset changes remove stale KD/teacher/status selections without choosing another model.
- Exact teacher pairs never match supersets and teacher ordering is irrelevant.
- Missing metadata and empty subsets never crash or hide all records by default.
- Clear filters resets search and conditional fields, not job files.
- Only the currently relevant branch renders role/family or teacher fields.

### Task1: Filter behavior and integration
- [x] Add AppTests for unavailable methods, conditional fields, cascading options, exact teacher combinations, stale state and clear-all. Observe RED.
- [x] Implement render_job_filters(jobs:list[dict])->list[dict] in app/job_filters.py with native controls and no writes.
- [x] Replace monitor's independent filters; preserve free-text search and job selection. Update older UI tests for method buttons. Observe GREEN.

### Task2: Verification
- [x] Run full unittest suite in masterthesis, then desktop/mobile browser checks without consequential actions.
- [x] Fresh independent final review and ordinary-extension documentation check; fix material findings with a regression test.
- [x] Leave branch uncommitted/unpublished, preserve existing server, and remove only agent test process/tab.
