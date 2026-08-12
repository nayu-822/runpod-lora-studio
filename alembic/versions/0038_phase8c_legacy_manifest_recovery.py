"""restore safe terminal manifest repairs from the pre-0037 worker"""

from __future__ import annotations

import re
from collections import defaultdict

import sqlalchemy as sa
from alembic import op

revision = "0038_phase8c_legacy_manifest_recovery"
down_revision = "0037_phase8c_manifest_repair_backfill"
branch_labels = None
depends_on = None

_SUCCESS_ITEM_STATUSES = {
    "imported",
    "linked_existing",
    "skipped",
}
_FAILURE_ITEM_STATUSES = {
    "failed",
    "canceled",
}
_ERROR_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def _safe_error_code(value: str | None) -> str | None:
    if value and _ERROR_CODE_RE.fullmatch(value):
        return value
    return None


def _legacy_target(
    statuses: list[str], existing_error_code: str | None
) -> tuple[str, str | None] | None:
    if not statuses or any(
        status not in _SUCCESS_ITEM_STATUSES | _FAILURE_ITEM_STATUSES
        for status in statuses
    ):
        return None
    has_success = any(status in _SUCCESS_ITEM_STATUSES for status in statuses)
    has_failure = any(status in _FAILURE_ITEM_STATUSES for status in statuses)
    if not has_failure:
        return "completed", None
    if has_success:
        return "partially_completed", _safe_error_code(existing_error_code)
    return None


def upgrade() -> None:
    connection = op.get_bind()
    jobs = (
        connection.execute(
            sa.text(
                """
                SELECT id, error_code
                FROM image_acquisition_jobs
                WHERE status = 'failed'
                  AND error_code = 'INCOMPLETE_ITEM_STATE'
                  AND completed_at IS NOT NULL
                  AND manifest_relative_path IS NOT NULL
                  AND manifest_repair_state IS NULL
                  AND manifest_repair_attempted_at IS NULL
                  AND manifest_target_status IS NULL
                  AND manifest_target_error_code IS NULL
                  AND worker_id IS NULL
                  AND claim_token IS NULL
                  AND current_item_id IS NULL
                  AND active_key IS NULL
                  AND cancellation_requested = 0
                  AND (
                      manifest_warning IS NULL
                      OR manifest_warning = 'MANIFEST_WRITE_FAILED'
                  )
                """
            )
        )
        .mappings()
        .all()
    )
    if not jobs:
        return

    job_ids = {str(row["id"]) for row in jobs}
    item_statuses: dict[str, list[str]] = defaultdict(list)
    for row in connection.execute(
        sa.text(
            """
            SELECT job_id, status
            FROM image_acquisition_job_items
            """
        )
    ).mappings():
        job_id = str(row["job_id"])
        if job_id in job_ids:
            item_statuses[job_id].append(str(row["status"]))

    for row in jobs:
        job_id = str(row["id"])
        target = _legacy_target(item_statuses[job_id], row["error_code"])
        if target is None:
            continue
        target_status, target_error_code = target
        connection.execute(
            sa.text(
                """
                UPDATE image_acquisition_jobs
                SET status = 'stale',
                    completed_at = NULL,
                    worker_id = NULL,
                    claim_token = NULL,
                    current_item_id = NULL,
                    heartbeat_at = NULL,
                    active_key = NULL,
                    manifest_warning = 'MANIFEST_WRITE_FAILED',
                    manifest_repair_state = 'pending',
                    manifest_repair_attempted_at = CURRENT_TIMESTAMP,
                    manifest_target_status = :target_status,
                    manifest_target_error_code = :target_error_code,
                    error_code = :target_error_code,
                    error_summary = :target_error_code,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
                  AND status = 'failed'
                  AND error_code = 'INCOMPLETE_ITEM_STATE'
                  AND completed_at IS NOT NULL
                  AND manifest_relative_path IS NOT NULL
                  AND manifest_repair_state IS NULL
                  AND manifest_repair_attempted_at IS NULL
                  AND manifest_target_status IS NULL
                  AND manifest_target_error_code IS NULL
                  AND worker_id IS NULL
                  AND claim_token IS NULL
                  AND current_item_id IS NULL
                  AND active_key IS NULL
                  AND cancellation_requested = 0
                  AND (
                      manifest_warning IS NULL
                      OR manifest_warning = 'MANIFEST_WRITE_FAILED'
                  )
                """
            ),
            {
                "id": job_id,
                "target_status": target_status,
                "target_error_code": target_error_code,
            },
        )


def downgrade() -> None:
    # This is a corrective data migration. Keeping the repair intent on
    # downgrade allows a later upgrade to resume the same restart-safe path.
    pass
