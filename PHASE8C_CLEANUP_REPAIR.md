# Phase 8C cleanup and manifest repair

Phase 8C adds restart-safe cleanup and manifest repair for acquisition jobs.

- A job with a mandatory `.part` cleanup warning remains `STALE` and cannot be claimed or resumed until cleanup is safe. Item claim and next-item selection also reject pending cleanup state.
- Cleanup recovery returns counts for successful cleanup, continued warnings, claim conflicts, database errors, eligible backlog, delayed backlog, and processed item IDs. Retry attempts use bounded exponential backoff.
- On Linux, `.part` cleanup traverses `projects/acquisition/jobs/<job>/parts` with directory file descriptors, checks directory identity with `fstat`, checks the entry with `stat(..., follow_symlinks=False)`, removes it with `unlink(dir_fd=...)`, and fsyncs the parts directory before committing the result. Unsafe traversal is rejected; the path fallback is development-only on Windows.
- If a POSIX directory component disappears during fd traversal, cleanup records an explicit absent-artifact result and never falls back to an absolute-path unlink. A missing leaf still fsyncs the held parts directory; an fsync failure remains a fixed cleanup warning and is retried.
- Manifest writes return an explicit result. A warning, ambiguous database commit, or unsafe reference repair moves the job to `manifest_repair_state=pending`; terminal job completion is not recorded until a repair succeeds. Startup and the periodic scheduler reconcile pending repairs.
- Normal, canceled, and unexpected worker terminal paths all use the manifest completion flow. If failure auditing or count recomputation cannot complete safely, a claim-conditional fallback records `STALE` plus `manifest_repair_state=pending` for later reconciliation. The pending record preserves the intended terminal status and fixed error code in a fixed-format warning marker; a repair worker audits unfinished items before writing the manifest.
- Cancellation remains `CANCELED` through repair, including cancellation before the next item and cancellation during an item. The manifest status and database status are written from the same preserved intent, and a lost repair claim cannot finalize the job for an older worker generation.
- The cleanup backoff calculation uses logarithmic thresholds instead of a raw `max/base` ratio, so subnormal positive bases remain finite and every retry delay is clamped to the configured range.
- The cleanup scheduler is single-start, stoppable, interval-based, and uses the configured batch and time budgets. Migration `0035_phase8c_cleanup_repair_scheduler` stores cleanup attempt counts and manifest repair state.

The scheduler is started by `create_app()` after startup reconciliation and is stopped at process exit. It does not upload or delete Google Drive files.
