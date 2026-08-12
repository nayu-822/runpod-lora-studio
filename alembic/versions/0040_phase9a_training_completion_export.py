"""add durable Phase 9A training completion exports"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0040_phase9a_training_completion_export"
down_revision = "0039_phase8c_manifest_orphan_scan"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_completion_exports",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column(
            "training_job_id",
            sa.String(length=36),
            sa.ForeignKey("training_jobs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.String(length=36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("storage_transfer_job_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("current_stage", sa.String(length=64), nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("claim_token", sa.String(length=128), nullable=True),
        sa.Column(
            "worker_generation", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("preview_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("export_relative_path", sa.Text(), nullable=True),
        sa.Column("export_manifest_relative_path", sa.Text(), nullable=True),
        sa.Column("export_manifest_sha256", sa.String(length=64), nullable=True),
        sa.Column("export_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("remote_relative_path", sa.Text(), nullable=True),
        sa.Column("remote_completion_manifest_relative_path", sa.Text(), nullable=True),
        sa.Column("completion_manifest_sha256", sa.String(length=64), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("id", name="uq_training_completion_exports_id"),
        sa.UniqueConstraint(
            "training_job_id", name="uq_training_completion_exports_training_job"
        ),
    )
    op.create_index(
        "ix_training_completion_exports_status_heartbeat",
        "training_completion_exports",
        ["status", "heartbeat_at", "updated_at", "id"],
        unique=False,
    )
    op.create_index(
        "ix_training_completion_exports_project_status",
        "training_completion_exports",
        ["project_id", "status", "updated_at"],
        unique=False,
    )
    op.create_index(
        "ix_training_completion_exports_storage_job",
        "training_completion_exports",
        ["storage_transfer_job_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_completion_exports_storage_job",
        table_name="training_completion_exports",
    )
    op.drop_index(
        "ix_training_completion_exports_project_status",
        table_name="training_completion_exports",
    )
    op.drop_index(
        "ix_training_completion_exports_status_heartbeat",
        table_name="training_completion_exports",
    )
    op.drop_table("training_completion_exports")
