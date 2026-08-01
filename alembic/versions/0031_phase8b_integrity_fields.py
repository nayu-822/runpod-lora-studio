"""add Phase 8B audit fields"""

import sqlalchemy as sa
from alembic import op

revision = "0031_phase8b_integrity_fields"
down_revision = "0030_phase8b_acquisition_downloads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_acquisition_job_items",
        sa.Column("last_attempted_at", sa.DateTime(timezone=True), nullable=True),
    )
    for column in (
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("requested_range_start", sa.Integer(), nullable=True),
        sa.Column("retry_after_seconds", sa.Float(), nullable=True),
        sa.Column("response_etag_fingerprint", sa.String(64), nullable=True),
        sa.Column("response_last_modified_fingerprint", sa.String(64), nullable=True),
        sa.Column(
            "worker_generation", sa.Integer(), nullable=False, server_default="0"
        ),
    ):
        op.add_column("image_acquisition_attempts", column)


def downgrade() -> None:
    for name in (
        "worker_generation",
        "response_last_modified_fingerprint",
        "response_etag_fingerprint",
        "retry_after_seconds",
        "requested_range_start",
        "http_status",
    ):
        op.drop_column("image_acquisition_attempts", name)
    op.drop_column("image_acquisition_job_items", "last_attempted_at")
