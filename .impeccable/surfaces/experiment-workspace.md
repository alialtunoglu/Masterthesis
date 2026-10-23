# Experiment workspace

Mode: Operate. Code-led evolution of the existing native Streamlit interface.

## Direction contract
Researchers should recognize the next action without decoding the project's file layout. Keep the existing models, results, safeguards and truthful metric semantics. No hero, decorative cards, gradients, emoji navigation or animation.

## First viewport
Sidebar contains one primary Yeni deney button and five destinations, including Veri Setleri. Overview shows actual activity and a next action. New experiment shows training type, dataset/model, compatible preset and basic epoch/batch/LR controls before technical details.

## Signature interaction
The preset list follows the model/data selection. Queue success opens the exact job with follow-running disabled, so another active run cannot steal focus.

Baseline selection separates training method, Student/Teacher role and CNN/ViT
architecture before dataset/model/config selection. Legacy storage stages stay
unchanged; job filters expose method, role and architecture independently.
Student ViT offers DeiT-Tiny and DeiT-Small. All registered student/teacher models
use compatible datasets and KD methods guarded by class/split/checkpoint validation.
Dataset registration is explicit: inspect → validate → publish immutable split.
Teacher selection follows dataset identity/input size; expensive model preflight
is an explicit button, not a side effect of every widget rerun.

## Detail policy
Successful compatibility details, commands, paths, logs and advanced parameters are expandable. Errors remain visible. Completed jobs do not show inapplicable stop controls. Result tables start with model and metrics and never average across datasets.

Job filters are dataset-first and cascading. Unavailable training methods remain
visible but disabled. Baseline exposes role/family/model; KD exposes student and
teacher identities, with teacher count/exact unordered combinations for multi-KD.
Only compatible downstream options are shown; stale choices reset to Tümü.

## Verification scope
AppTest for all launcher routes and safety gates; full existing suite; live local browser review. No training/worker/deletion actions during browser checks.
