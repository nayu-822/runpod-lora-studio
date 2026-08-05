"""persist manifest repair target intent"""

from __future__ import annotations

import re

import sqlalchemy as sa
from alembic import op

revision = "0036_phase8c_manifest_repair_intent"
down_revision = "0035_phase8c_cleanup_repair_scheduler"
branch_labels = None
depends_on = None

_INTENT_PREFIX = "MANIFEST_REPAIR_PENDING:"
_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "partially_completed",
    "canceled",
}
_ERROR_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def _legacy_status(warning: str | None) -> str | None:
    if not warning or not warning.startswith(_INTENT_PREFIX):
        return None
    value = warning[len(_INTENT_PREFIX) :]
    return value if value in _TERMINAL_STATUSES else None


def _safe_error_code(value: str | None) -> str | None:
    if value and _ERROR_CODE_RE.fullmatch(value):
        return value
    return None


def upgrade() -> None:
    op.add_column(
        "image_acquisition_jobs",
        sa.Column("manifest_target_status", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "image_acquisition_jobs",
        sa.Column("manifest_target_error_code", sa.String(length=64), nullable=True),
    )

    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            """
            SELECT id, status, cancellation_requested, error_code,
                   manifest_warning, manifest_repair_state
            FROM image_acquisition_jobs
            WHERE manifest_repair_state IS NOT NULL
               OR manifest_warning LIKE :intent_prefix
            """
        ),
        {"intent_prefix": f"{_INTENT_PREFIX}%"},
    ).mappings()
    for row in rows:
        status = _legacy_status(row["manifest_warning"])
        error_code = _safe_error_code(row["error_code"])
        if bool(row["cancellation_requested"]):
            status = "canceled"
            error_code = "CANCELED"
        elif status is None:
            # Keep the row repairable. The repair worker will audit every
            # non-terminal item before it is allowed to write a manifest.
            status = "failed"
            error_code = "INCOMPLETE_ITEM_STATE"
        connection.execute(
            sa.text(
                """
                UPDATE image_acquisition_jobs
                SET manifest_target_status = :target_status,
                    manifest_target_error_code = :target_error_code,
                    manifest_warning = :warning
                WHERE id = :id
                """
            ),
            {
                "id": row["id"],
                "target_status": status,
                "target_error_code": error_code,
                "warning": "MANIFEST_WRITE_FAILED",
            },
        )


def downgrade() -> None:
    op.drop_column("image_acquisition_jobs", "manifest_target_error_code")
    op.drop_column("image_acquisition_jobs", "manifest_target_status")
