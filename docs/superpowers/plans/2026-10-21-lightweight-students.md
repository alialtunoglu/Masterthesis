# Lightweight Students Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with test-driven-development and one final review.

**Goal:** Add ShuffleNetV2 x0.5 and MicroNet-M0 alongside MobileNetV3-Small, without starting experiments.

**Architecture:** Extend the existing shared model catalog/factory and feature registry. Vendor the MIT-licensed author's MicroNet implementation at commit `ed89b2615e96e945f3dba64645ce8a68b0366222`, adapting only imports/configuration, safe pretrained loading, transfer head and singleton-batch behavior. Existing generic trainers, export, metadata, UI and comparison paths remain the consumers.

**Tech Stack:** Existing conda masterthesis, PyTorch/torchvision, unittest, Streamlit AppTest; no new package/environment.

**Spec:** User-approved three-student proposal in this conversation: MobileNetV3-Small reference, ShuffleNetV2 x0.5 and one validated MicroNet M0/M1. Choose M0 as the smallest official variant, retaining pretrained support.

## Global Constraints

- Work on the existing `codex/clean-kd-protocol` branch; no worktree, commit, push or merge, preserving unrelated dirty changes.
- No real training, queue mutation or MLflow run. Only synthetic short forward/backward tests and read-only data/config checks.
- Preserve all original data/results. Two new clean baseline configs use the same immutable dataset_ref and protocol as the existing reference.
- Use author-published source/weights with attribution; never silently fall back to random initialization. Cache weights, pin checksum, deserialize tensors with weights_only=True.
- Inference head is a raw Linear classifier for project CE, not the author's ImageNet output BatchNorm/activation. Retain pretrained feature layers. Document this transfer adaptation.

## Review Focus

- Pretrained download failure, corruption or mismatched backbone must raise, not return random weights (Task 1).
- Small custom class count must not inherit the 1000-class ImageNet head (Task 1).
- Singleton final train batch must not crash MicroNet's classifier BatchNorm; use running statistics only for that BN case, preserving all samples (Task 1).
- KD feature hooks and gradients must work for both new students, single and multi teacher (Task 2).
- UI/preset/snapshot/export must retain selected model and clean ref; switching student must not reuse the other's config (Task 2).

## Task 1: Shared model implementation

**Files:** `src/models/{model_catalog.py,model_factory.py,micronet.py,_micronet/*}`, `tests/test_lightweight_students.py`.

**Interfaces:** Existing `create_model(name, num_classes, pretrained)` and `get_model_spec(name)`; names `shufflenet_v2_x0_5`, `micronet_m0`.

- [x] Write/run failing tests: catalog student/CNN role, six-class shape, parameter reduction relative to reference, pretrained strict backbone loading and no silent fallback, singleton train backward, checkpoint roundtrip.
- [x] Implement official torchvision ShuffleNet and vendored/pinned MicroNet-M0; replace output heads after pretrained loading. Keep feature layer checkpoint keys compatible.
- [x] Run focused tests; verify actual official pretrained weights load and no_grad output on synthetic input. No dataset training.

## Task 2: Platform integration and verification

**Files:** `src/training/kd_features.py`, `app/pages/2_Baseline_Experiments.py`, two `configs/baseline/*_plantpathology2021_clean.json`, `docs/LIGHTWEIGHT_STUDENTS.md`, tests.

**Interfaces:** Catalog drives baseline CLI, KD config validation, taxonomy and KD selection. Baseline page must also derive CNN student membership from that catalog. KD nodes: ShuffleNet `conv5`, MicroNet `features` (spatial tensors).

- [x] Write/run failing feature/gradient, config equality, AppTest selection/command; extend existing portable snapshot and actual external import roundtrip verification. Never click training or dry-run.
- [x] Add nodes, catalog-driven baseline choices and two exact clean protocol configs; document 224x224 ImageNet normalization, head adaptation, weight provenance and six-baseline plan.
- [x] Run focused and complete unittest suites in masterthesis, using 8MiB Python test-thread stack on Windows as previously established; record results and `git diff --check`.
- [x] Final review scoped only to this change, fix important findings with regression tests. No commit/merge/push.

## Execution record

- Source inspection: official repository MIT license; M0 script uses stem4, groups(2,2), out640 and DYShiftMax. Official weight link HTTP responds 200 (4,163,581 bytes); HTTPS endpoints timed out in this environment. Pin retrieved weight digest, use tensor-only loading; transport limitation must be disclosed.
- Pre-existing scope: extensive unrelated changes already present; no files reverted.
- Ruling: user explicitly approved applying the three-student proposal; implement in place on the existing feature branch, without another approval cycle or commits. Only a final read-only reviewer is delegated under executing-plans.
- Task 1 complete: unsupported model/module errors reproduced first; 5 model tests GREEN. Real author checkpoint loaded strictly (entire original ImageNet state) and six-class synthetic CUDA forward succeeded. New weights only, no package/environment changes.
- Task 2 integration RED: missing feature registry, clean configs, import allowance and UI options. After implementation, 9 focused tests GREEN in 15.008s. One test initially used positional features for the kwargs-only multi-logit interface; corrected to the trainer's existing keyword contract, no production API change.
- Ruling: retain author's MIT operators locally rather than runtime torch.hub source execution; keeps portable runs independent of mutable upstream code and avoids new yacs dependency. Costs vendored source size.
- Ruling: use strict original-head loading then raw Linear transfer head, and running-stat fallback only for singleton classifier BN. Documents intentional transfer adaptation; not an exact ImageNet evaluation replica.
- Ruling: fixed checkpoint SHA is measured from the retrieved official HTTP file, not independently authenticated. Tensor-only load and pinned digest prevent arbitrary pickle execution/subsequent substitutions; initial-source authenticity limitation is documented.
- Initial full suite after implementation: 221/221 passed in 58.729s. Additional import roundtrip and notebook cell execution checks added before final verification (not constant membership-only assertions).
- Notebook test correction: registered clean data requires the real dataset_package (as the production UI already supplies). The test now generates with real metadata, compiles every cell, and executes the serialized metadata assignment; it does not mount/copy the real images or pretend to execute cloud training.
- Final review: independent read-only reviewer found no Critical or Important issue. Minor (deferred): loss tests use total CE+KD gradients rather than isolating feature/relation-only gradient connectivity. Current production capture is connected; focused isolated-gradient regression can be added later.
- Final Ruling: hardware accuracy/convergence, latency/energy/peak RAM are not validated — no real training or hardware benchmark authorized; report only functional checks and parameter/operation estimates.
- Final Ruling: HTTP initial-file authenticity and published-notebook support limits remain explicit in docs; no claim of independent checksum authentication or working cloud run before publication.
- Final Ruling: unrelated pre-existing dirty files and exhaustive upstream equivalence are outside review scope; inspected pinned source and strict official checkpoint compatibility for this integration. Existing full-suite coverage is rerun by the primary agent.
- Final verification: 222/222 tests passed in 62.191s, exit 0, `reports/preflight_2026-10-21/lightweight_full_suite.log`; focused 10/10 (5 model + 5 integration) passed. Scoped `git diff --check` clean (CRLF notices only). Streamlit already running on port8501; no server/worker/train started. Existing branch kept as-is, without commit/merge/push. Owned temporary download removed after validating its digest; official pretrained caches retained for training.
