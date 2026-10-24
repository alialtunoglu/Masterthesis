# Laboratory experiment management

Approved in chat: GPU selection, common monitoring including existing notebook runs,
and manual transfer of completed result packages. RustDesk remains the access method.

## Design

Reuse the current Streamlit launchers, queue, native trainers and verified external
bundle exporter/importer. Add GPU-aware admission (automatic or an explicit physical
GPU index), initially one process per GPU. Keep the existing global queue limit as
an additional limit. GPU jobs must share the existing laboratory notebook lock
(`lab_state/gpu_<index>.lock`), held for the entire child process lifetime.

Notebook attempts are read-only virtual jobs discovered from `lab_state/*/state.json`;
they retain their own logs, config, MLflow store and stop-request path. Never rewrite
these attempts as native jobs, kill a reused PID, or treat a disappeared process as
a successfully finished experiment. The UI distinguishes preparation, epoch progress,
dry run, completion, error and unavailable process identity.

Successful native runs can be explicitly exported through the existing bundle service.
Only verified completed ZIP/checkpoint pairs are shown as transferable. Imports keep
using the existing inbox and idempotent MLflow registration. No Git synchronization,
remote networking, automatic training, dataset changes or four-way concurrency.

Existing laboratory teachers can be selected using the already verified portable
teacher metadata, without inventing local MLflow runs or copying an active database.

Preserve native Streamlit widgets, Turkish action labels, existing research protocols,
user changes and running processes. Use the existing masterthesis environment only.

## Alternatives

- A notebook-only monitor is smaller but leaves launch and GPU management fragmented.
- The selected approach extends the existing application and manual bundle transfer.
- A remote control server/Git synchronization was declined and remains out of scope.

## Verification and handoff

Use isolated tests for inventory/locks, GPU scheduling, notebook identity/stop routing,
export integrity/idempotency and Streamlit controls. Run the full suite. Real two-GPU
WSL execution is verified by the user after deployment, not claimed from Windows tests.
