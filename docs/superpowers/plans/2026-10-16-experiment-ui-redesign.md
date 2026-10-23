# Experiment UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make experiment creation, monitoring and comparison discoverable without changing training behavior.

**Architecture:** Keep native Streamlit and existing launchers/services. Register legacy routes in a hidden navigation registry, expose four task-oriented sidebar destinations and one New experiment action. Use a small shared UI helper for compatible preset selection and queue handoff; keep all validation in existing services.

**Tech Stack:** Python, Streamlit 1.59, pandas, unittest/Streamlit AppTest; existing masterthesis conda environment.

**Spec:** `docs/superpowers/specs/2026-10-16-experiment-ui-design.md`

## Global Constraints
- No training, worker start, record removal or package installation during verification.
- Preserve dirty working-tree changes, notebook/import/recovery and safe removal contracts.
- Work on `codex/ui-ux-redesign`, not a worktree. Do not commit unrelated changes.
- No new UI framework or dependencies, fake metrics, cross-dataset averages or decorative AI-slop styling.

## Review Focus
Legacy route compatibility, model/config synchronization, confirmation gates, selected-job handoff, conditional job controls, dataset-local results and student-role filtering. Verify all existing tests and browser layouts without starting jobs.

## File map
- `app/streamlit_app.py`: four-section shell and registered legacy routes.
- `app/pages/0_New_Experiment.py`: training-family selector reusing existing launchers.
- `app/pages/1_Dashboard.py`, `app/pages/10_Settings.py`: task overview and relocated technical controls.
- `app/experiment_ui.py`: compatible preset selector, queue handoff and compact result projection.
- Existing launcher pages `2`, `5`, `6`, `8`, `9`: selection order, progressive disclosure and handoff.
- `app/pages/3_Job_Monitor.py`: identity/status first, contextual actions/metrics.
- `app/pages/4_Results_Explorer.py`, `app/comparison_view.py`: dataset-scoped readable results and role-correct comparison.
- `tests/test_experiment_ui.py`, `tests/test_streamlit_launcher_pages.py`: new acceptance and regression coverage.
- `PRODUCT.md`, `DESIGN.md`, `.impeccable/design.json`: confirmed product and native-interface decisions.

### Task 1: Navigation and overview
- [x] Add AppTest assertions for four primary links and the new experiment entry; observe failure before implementation.
- [x] Register hidden legacy routes, build sidebar, new-experiment dispatcher and overview with real jobs.
- [x] Relocate worker/GPU/split/MLflow controls to Settings; retain explicit confirmation.
- [x] Verify new routes and legacy `AppTest.switch_page()` behavior.

### Task 2: Standard launchers
- [x] Test selection behavior: dataset/model precede compatible preset and changing either excludes incompatible configs.
- [x] Implement shared selector, retain existing command builders, reduce baseline/CNN duplication.
- [x] Collapse technical parameters/command preview; retain common epoch/batch/LR fields, notebook export, recovery and training confirmation.
- [x] Queue success records selected job and opens monitor; test with patched job boundary only.

### Task 3: Distillation forms
- [x] Add assertions for readable fixed-scope explanation and Teacher → Student identity.
- [x] Group training and strategy details, surface compatibility failures, collapse successful validation/profile details.
- [x] Retain uploaded config validation, aggregation normalization, snapshots and all launch gates.
- [x] Verify single/multi KD AppTest and existing strategy/config tests.

### Task 4: Monitoring
- [x] Test that completed jobs have no stop/cancel actions and only recorded metrics render.
- [x] Move worker controls out; show job identity, state and epoch progress before technical details.
- [x] Keep search-by-job-ID, filters, recovery actions and log access. Respect explicitly selected newly queued job.
- [x] Verify running/queued/finished/stopped fixtures without external mutations.

### Task 5: Results and comparison
- [x] Test compact column projection and student choices excluding teacher-only rows.
- [x] Move filters into their relevant tab; require a concrete dataset before model chart aggregation.
- [x] Show model/metrics first, expose full data in an expander, preserve artifact/import/removal flows.
- [x] Verify comparison report downloads and removal regressions.

### Task 6: Verification and handoff
- [x] Run targeted AppTests: `python -m unittest tests.test_experiment_ui tests.test_streamlit_launcher_pages tests.test_comparison_view` (use discovery if tests is not a package).
- [x] Run complete suite with the masterthesis interpreter; record any pre-existing failures by name.
- [x] Start only local Streamlit UI for batched desktop/mobile browser verification; do not click execution/removal controls.
- [x] Run Impeccable detector once, obtain fresh finish review, apply material fixes as a batch.
- [x] Update plan/ledger and report implemented behavior, verification and remaining risks.

## Execution ledger
- Planning: existing graph MCP unavailable (`Transport closed`); source files inspected directly.
- Workspace: user-owned dirty changes retained; created branch without resetting files.


- Final verification: masterthesis full unittest suite **145 passed / 51.747s**; targeted UI suite **12 passed**; diff whitespace check clean.
- Browser evidence: 1280×720 desktop and genuine 390×844 mobile. The temporary viewport override was reset; no real training, worker start, import or deletion action was exercised.
- Impeccable detector: one run, no findings. Independent full review returned five fixes; the verdict pass scored **all five resolved**, disposition **ship** at that fix-list scope (not a new whole-surface audit).
- Documentation: PRODUCT.md, canonical DESIGN.md and schemaVersion2 sidecar recorded native Streamlit ownership. No new CSS, dependency, imagery or motion.
- Integration: left uncommitted on codex/ui-ux-redesign to avoid including pre-existing user changes. No merge or push.
- Decisions and tradeoffs: branch instead of worktree preserves the user's workflow but offers less physical isolation; native Streamlit avoids a new frontend but retains framework layout limits; shared launcher rendering reduces duplication but shares regression risk across three pages; no automatic commit avoids mixing unrelated changes but leaves versioning to the next integration step.
- Deferred review findings: none. Not validated: live GPU training, remote notebook execution, exhaustive keyboard/contrast audit.
