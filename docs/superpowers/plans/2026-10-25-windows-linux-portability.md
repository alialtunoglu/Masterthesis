# Windows/Linux portability implementation plan

**Goal:** Installed checkout runs tests and CLI/Streamlit/queue entry points without a manually configured PYTHONPATH on Windows, Linux and WSL2.

**Architecture:** Keep directories in place. Use `src.*` and `app.*` as canonical package identities, install both packages, and launch subprocesses as modules. Resolve the workspace independently of installation location. Keep existing CUDA/CPU behavior; no MPS support.

**Constraints:** Do not upgrade PyTorch/CUDA or install runtime dependencies. Preserve user edits, datasets, checkpoints, splits and result files. No model training or scientific algorithm changes. Work in WSL checkout, branch only, no worktree or subagents.

- [x] Capture baseline test failures without PYTHONPATH; preserve existing unrelated changes.
- [x] Add import/entry-point tests; canonicalize package imports and package configuration; editable install with `--no-deps --no-build-isolation`.
- [x] Standardize module commands, child environments, project root and relative MLflow resolution; cover spaces and different cwd.
- [x] Consolidate CPU/CUDA device resolution without changing auto preference; preserve unavailable-CUDA errors.
- [x] Run complete WSL suite without PYTHONPATH; build/import checks and Windows verification where feasible; record hardware/platform limits.
- [ ] Review exact staged changes; push feature branch, merge verified result into main, retest and push main. Delete only merged task branches.

## Decisions and verification ledger

- Existing Windows and WSL main and remote main are 59780974. No unpublished semantic source changes; generated egg-info and EOL differences are pre-existing and not part of the initial push. Protected result deletions are preserved, not staged.
- User explicitly excluded macOS/MPS and explicitly authorized branch/push/merge/merged-branch cleanup.
- Baseline: 261 passed, 2 failed (worker imports) with no PYTHONPATH. New isolated-process tests reproduced missing src/app package imports (RED).
- Canonical package decision: retain directories, use src.* and app.*; do not introduce a third package tree or compatibility module aliases. Console/module entry points and test patch targets migrate together.
- Source/scientific review: strategy, trainer and recovery changes are import-only. Dataloader, optimizer/scheduler, loss formulas, seeds, precision, split and model definitions are unchanged. Native CPU/CUDA selection is consolidated, not expanded to MPS.
- Full WSL suite after implementation and final review: 267 passed, 117 subtests passed in 70.62s, PYTHONPATH unset. Final review covered foreign absolute SQLite paths without rewriting historical tracking URIs.
- PyTorch remains 2.6.0+cu124, CUDA build 12.4. A tiny CUDA tensor sum returned 5.0; no dataset training started. Editable installation and wheel build used --no-deps --no-build-isolation.
- Native Windows Python 3.12 smoke test passed against built wheel: imports, workspace with spaces, msvcrt lock contention/release, CPU subprocess exit propagation. Native Windows lacks PyTorch; full Windows ML tests were not run. No Windows environment or dependencies were installed.
- No agents or worktrees used per user preference. Review performed directly; no claim of independent review.
