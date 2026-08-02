# Phase 8C cleanup and manifest repair

Phase 8C adds restart-safe cleanup and manifest repair for acquisition jobs.

- A job with a mandatory `.part` cleanup warning remains `STALE` and cannot be claimed or resumed until cleanup is safe. Item claim and next-item selection also reject pending cleanup state.
- Cleanup recovery returns counts for successful cleanup, continued warnings, claim conflicts, database errors, eligible backlog, delayed backlog, and processed item IDs. Retry attempts use bounded exponential backoff.
- On Linux, `.part` cleanup traverses `projects/acquisition/jobs/<job>/parts` with directory file descriptors, checks directory identity with `fstat`, checks the entry with `stat(..., follow_symlinks=False)`, removes it with `unlink(dir_fd=...)`, and fsyncs the parts directory before committing the result. Unsafe traversal is rejected; the path fallback is development-only on Windows.
- Manifest writes return an explicit result. A warning, ambiguous database commit, or unsafe reference repair moves the job to `manifest_repair_state=pending`; terminal job completion is not recorded until a repair succeeds. Startup and the periodic scheduler reconcile pending repairs.
- The cleanup scheduler is single-start, stoppable, interval-based, and uses the configured batch and time budgets. Migration `0035_phase8c_cleanup_repair_scheduler` stores cleanup attempt counts and manifest repair state.

The scheduler is started by `create_app()` after startup reconciliation and is stopped at process exit. It does not upload or delete Google Drive files.
