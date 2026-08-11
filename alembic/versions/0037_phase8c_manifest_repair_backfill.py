"""reclassify ambiguous manifest repair intents from 0036"""

from __future__ import annotations

import re
from collections import defaultdict

import sqlalchemy as sa
from alembic import op

revision = "0037_phase8c_manifest_repair_backfill"
down_revision = "0036_phase8c_manifest_repair_intent"
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


def _reclassify(
    statuses: list[str], existing_error_code: str | None
) -> tuple[str, str | None]:
    if statuses and all(status in _SUCCESS_ITEM_STATUSES for status in statuses):
        return "completed", None
    if statuses and all(status in _FAILURE_ITEM_STATUSES for status in statuses):
        return "failed", _safe_error_code(existing_error_code)
    if statuses and any(status in _SUCCESS_ITEM_STATUSES for status in statuses):
        if any(status in _FAILURE_ITEM_STATUSES for status in statuses):
            return "partially_completed", _safe_error_code(existing_error_code)
    return "failed", "INCOMPLETE_ITEM_STATE"


def upgrade() -> None:
    connection = op.get_bind()
    jobs = (
        connection.execute(
            sa.text(
                """
            SELECT id, cancellation_requested, error_code
            FROM image_acquisition_jobs
            WHERE manifest_repair_state IS NOT NULL
              AND manifest_warning = 'MANIFEST_WRITE_FAILED'
              AND manifest_target_status = 'failed'
              AND manifest_target_error_code = 'INCOMPLETE_ITEM_STATE'
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
        if bool(row["cancellation_requested"]):
            target_status, target_error_code = "canceled", "CANCELED"
        else:
            target_status, target_error_code = _reclassify(
                item_statuses[job_id], row["error_code"]
            )
        connection.execute(
            sa.text(
                """
                UPDATE image_acquisition_jobs
                SET manifest_target_status = :target_status,
                    manifest_target_error_code = :target_error_code
                WHERE id = :id
                """
            ),
            {
                "id": job_id,
                "target_status": target_status,
                "target_error_code": target_error_code,
            },
        )


def downgrade() -> None:
    # This is a corrective data migration. Keeping the repaired target values
    # on downgrade allows a 0036 -> 0037 re-upgrade to remain idempotent.
    pass
