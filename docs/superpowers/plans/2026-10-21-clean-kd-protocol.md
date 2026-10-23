# Clean KD Protocol Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Publish a separate, leakage-free PlantPathology2021 revision and make the student CE control apply the same cosine/early-stopping protocol as KD, without starting training.

**Architecture:** Reuse the existing imagefolder dataset registry and content-addressed splits. Prepare an independent physical imagefolder (copies, not links) so the existing validation, portable export, and all launchers work unchanged. Reuse KD optimizer/scheduler factories for baseline; preserve legacy defaults and results.

**Tech Stack:** Python 3.11, conda masterthesis, PyTorch, unittest, existing Streamlit UI.

**Spec:** Approved solution in chat; evidence and exclusion policy: reports/preflight_2026-10-21/README.md.

## Global Constraints
- Do not start a real training job, CLI dry-run or MLflow run.
- Do not delete/relabel source images or overwrite the legacy split/results.
- Exclude every record in conflicting SHA256 groups; approved current exclusion is 40 records in 20 groups.
- New ID: plantpathology2021_clean; content revision and split seed42 remain pinned.
- Control: mobilenet_v3_small, 224px, batch32, max30 epochs, AdamW LR0.0003/WD0.0001, cosine, val_macro_f1, patience7, pretrained/augmentation on.
- No new packages, worktree, unrelated refactor, commit, merge, or push. Work on codex/clean-kd-protocol; preserve existing dirty changes.

## Review Focus
- Interrupted preparation must not overwrite source or silently trust changed copied images.
- Same bytes with different labels must exclude both records, not arbitrarily retain a label.
- Identical-label duplicates must never leak across new splits.
- UI scheduler/patience values must reach the actual baseline trainer, not only JSON.
- Legacy teachers and results must remain incompatible with the new dataset identity.

### Task 1: Versioned clean dataset
**Files:** scripts/prepare_clean_plantpathology.py; tests/test_clean_plantpathology.py; generated configs/datasets/plantpathology2021_clean.json and datasets registry.
**Interfaces:** prepare_clean_dataset(root: Path) -> dict containing dataset_ref, counts, exclusions and source hashes; no source mutations.
- [x] Write fixture tests for conflicting duplicates, unchanged sources, deterministic rerun, modified destination rejection and split hash disjointness.
- [x] Run tests and observe missing implementation failures.
- [x] Implement preparation using contained_path, sha256_file, DatasetDefinition, scan_dataset, register_dataset_revision and create_registered_split. Reuse imagefolder; reject links; existing destination bytes must match.
- [x] Run fixtures, then prepare actual clean version and record its pinned identity/exclusion manifest. Full pixel validation is intentionally retained by the registry.

### Task 2: Effective baseline protocol and UI
**Files:** src/training/train_baseline.py; app/standard_launcher.py; app/experiment_runner.py; configs/presets/student_baseline/cnn.json; tests/test_baseline_protocol.py.
**Interfaces:** baseline command builder optional scheduler/patience flags; existing signatures/default calls remain valid.
- [x] Write failing tests: cosine config accepted, actual scheduler LR changes, early stopping ends at expected epoch through baseline main, invalid options rejected, launcher snapshot roundtrip.
- [x] Reuse TrainingConfig/create_optimizer/create_scheduler; forward scheduler, patience and monitor to fit; honor augmentation config.
- [x] Add baseline-only scheduler and patience UI fields; keep effective config/command aligned; default preset declares cosine explicitly.
- [x] Run focused tests; keep legacy scheduler none and absent patience backward-compatible.

### Task 3: Readiness and handoff
**Files:** generated baseline/teacher clean configs; reports/preflight_2026-10-21/clean_readiness.md; tests/test_clean_plantpathology.py and test_baseline_protocol.py.
- [x] Create one student CE and three teacher configs pinned to the clean dataset_ref, without starting jobs.
- [x] Verify source content/legacy split and eleven legacy checkpoint hashes unchanged, clean content/class/split identities valid, hash overlaps zero, selected config parser and model inference work without training. No production result writes or MLflow runs were performed.
- [x] Run full unittest discovery in masterthesis, record every failure/skip and focused regression results.
- [x] Review complete changes and write exact UI/CLI handoff; state old teachers need clean retraining and KD is not ready until compatible new teacher checkpoints exist.

## Execution record
- Approved plan executed inline; no extra approval round needed for the already-approved solution.
- Ruling: copy approximately 13.9 GiB into a separate imagefolder rather than add a new manifest layout or allow filesystem links. This reuses existing safety/transfer validation; costs additional local disk and validation time.
- Ruling: retain existing full-pixel registry validation even though it is slower than the initial header-only audit.
- Baseline before changes: 72 targeted tests passed in the preceding preflight. Full-suite baseline is captured separately before implementation.
- Full-suite baseline: 200 tests passed before changes (full_suite_before.log).
- Ruling: a real baseline-main fixture exposed a Windows recovery temporary path longer than 260 characters. New baseline/teacher runs use compact unique IDs, keeping descriptive dataset/model fields in hierarchy and metadata. Old paths remain unchanged. This is necessary for the new clean dataset's runnable recovery configuration.
- Task 1 RED: missing preparation implementation; GREEN: three fixture tests passed. Actual 17,237-image copy and full registry validation running.
- Task 2 RED: cosine rejected, invalid patience accepted, main ran 6 instead of 3 epochs. GREEN: five protocol tests passed; compact-path regression in progress.
- Final review: independent read-only reviewer found destination link containment (Important) and negative/bool label IDs (regraded Important at the source trust boundary). Both reproduced RED; clean-dataset boundary tests and complete focused suite GREEN (12 tests).
- Final: minor (deferred): interrupted copying can leave a partial destination file that retry rejects rather than overwrites. Safe failure is retained and documented; automatic recovery of the preparation copy is outside this change.
- Ruling: AppTest's effective-command check runs in a separate process and never clicks the training button; CLI roundtrip and existing immutable-snapshot tests cover transport without launching a real job.
- Task 2 complete: seven baseline protocol tests, including real synthetic main/fit with cosine + patience and Windows recovery, plus five curation tests passed. Full suite running.
- Full-suite verification: 212 tests passed in 52.628s. Native Windows AppTest worker stack overflow (exit 0xC00000FD) occurred in ordinary all-modules discovery; individual UI test passed. Increasing test-runner thread stack to 8 MiB made the complete unchanged suite pass. No application/global OS setting was changed.
- Task 1 complete: 17,237 independent image copies, 40 conflicting records excluded without deletion, registered content revision 629dfca94858f0bd429bd289b2a83a59dd4beed293e602780193d10afdb6af08 and split 955c38474715ffd540121ab2be7699430e22f9956c110d7cd1617b4b8d007b72 published. Counts: 12,067 / 2,584 / 2,586.
- Task 3 complete: verify_clean_readiness.py passed; clean hash intersections zero, all four pretrained caches/forward passes and pinned config parsers passed, shared baseline/KD training protocol equal, source content + legacy split + eleven checkpoint hashes unchanged. Actual clean-dataset AppTest selected correct config/cosine/patience without clicking train. Readiness: CE and CNN teacher training only; KD awaits new clean teacher runs.
- Final verification after real clean publication and final code edits: 212/212 tests passed in 51.743s with the documented Windows runner stack setting; git diff --check passed (line-ending notices only). No real training processes started, no commit/merge/push performed.
