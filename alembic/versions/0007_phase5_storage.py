"""add Phase 5 model and storage transfer tables"""

import sqlalchemy as sa
from alembic import op

revision = "0007_phase5_storage"
down_revision = "0006_phase4_dataset_snapshots"
branch_labels = None
depends_on = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "managed_models",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("display_name", sa.String(512), nullable=False),
        sa.Column("model_type", sa.String(32), nullable=False),
        sa.Column("remote_name", sa.String(128), nullable=False),
        sa.Column("remote_relative_path", sa.Text(), nullable=False),
        sa.Column("remote_file_name", sa.String(512), nullable=False),
        sa.Column("remote_size_bytes", sa.Integer(), nullable=False),
        sa.Column("remote_modified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("remote_hash_type", sa.String(32), nullable=True),
        sa.Column("remote_hash_value", sa.String(512), nullable=True),
        sa.Column("local_path", sa.Text(), nullable=True),
        sa.Column("local_size_bytes", sa.Integer(), nullable=True),
        sa.Column("local_sha256", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("rclone_version", sa.String(128), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("downloaded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        *_timestamps(),
        sa.UniqueConstraint(
            "remote_name", "remote_relative_path", name="uq_managed_model_remote_path"
        ),
    )
    op.create_index("ix_managed_models_status", "managed_models", ["status"])

    op.create_table(
        "model_transfers",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("managed_model_id", sa.String(36), nullable=False),
        sa.Column("direction", sa.String(16), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("destination_path", sa.Text(), nullable=False),
        sa.Column("expected_size_bytes", sa.Integer(), nullable=False),
        sa.Column("transferred_size_bytes", sa.Integer(), nullable=False),
        sa.Column("expected_hash", sa.String(512), nullable=True),
        sa.Column("actual_hash", sa.String(64), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("rclone_exit_code", sa.Integer(), nullable=True),
        sa.Column("rclone_version", sa.String(128), nullable=True),
        sa.Column("settings_snapshot", sa.Text(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["managed_model_id"], ["managed_models.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_model_transfers_managed_model_id", "model_transfers", ["managed_model_id"]
    )
    op.create_index("ix_model_transfers_status", "model_transfers", ["status"])

    op.create_table(
        "storage_transfer_jobs",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("project_id", sa.String(36), nullable=True),
        sa.Column("snapshot_id", sa.String(36), nullable=True),
        sa.Column("training_run_id", sa.String(36), nullable=True),
        sa.Column("transfer_type", sa.String(32), nullable=False),
        sa.Column("source_kind", sa.String(16), nullable=False),
        sa.Column("destination_kind", sa.String(16), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("current_step", sa.String(128), nullable=False),
        sa.Column("item_count", sa.Integer(), nullable=False),
        sa.Column("processed_item_count", sa.Integer(), nullable=False),
        sa.Column("succeeded_item_count", sa.Integer(), nullable=False),
        sa.Column("failed_item_count", sa.Integer(), nullable=False),
        sa.Column("skipped_item_count", sa.Integer(), nullable=False),
        sa.Column("total_bytes", sa.Integer(), nullable=False),
        sa.Column("transferred_bytes", sa.Integer(), nullable=False),
        sa.Column("cancel_requested", sa.Integer(), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("manifest_path", sa.Text(), nullable=True),
        *_timestamps(),
    )
    op.create_index(
        "ix_storage_transfer_jobs_project_id", "storage_transfer_jobs", ["project_id"]
    )
    op.create_index(
        "ix_storage_transfer_jobs_snapshot_id", "storage_transfer_jobs", ["snapshot_id"]
    )
    op.create_index(
        "ix_storage_transfer_jobs_status", "storage_transfer_jobs", ["status"]
    )

    op.create_table(
        "storage_transfer_items",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("transfer_job_id", sa.String(36), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("item_type", sa.String(32), nullable=False),
        sa.Column("direction", sa.String(16), nullable=False),
        sa.Column("expected_size", sa.Integer(), nullable=False),
        sa.Column("transferred_size", sa.Integer(), nullable=False),
        sa.Column("source_sha256", sa.String(64), nullable=True),
        sa.Column("destination_hash_type", sa.String(32), nullable=True),
        sa.Column("destination_hash_value", sa.String(512), nullable=True),
        sa.Column("verification_status", sa.String(32), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["transfer_job_id"], ["storage_transfer_jobs.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "transfer_job_id", "relative_path", name="uq_storage_transfer_item_path"
        ),
    )
    op.create_index(
        "ix_storage_transfer_items_transfer_job_id",
        "storage_transfer_items",
        ["transfer_job_id"],
    )

    op.create_table(
        "project_storage_settings",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False, unique=True),
        sa.Column("project_remote_root", sa.Text(), nullable=False),
        sa.Column("snapshot_remote_root", sa.Text(), nullable=False),
        sa.Column("training_remote_root", sa.Text(), nullable=False),
        sa.Column("artifact_remote_root", sa.Text(), nullable=False),
        sa.Column("selected_managed_model_id", sa.String(36), nullable=True),
        sa.Column("overwrite_policy", sa.String(32), nullable=False),
        sa.Column("verification_policy", sa.String(32), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("project_storage_settings")
    op.drop_index(
        "ix_storage_transfer_items_transfer_job_id", table_name="storage_transfer_items"
    )
    op.drop_table("storage_transfer_items")
    op.drop_index("ix_storage_transfer_jobs_status", table_name="storage_transfer_jobs")
    op.drop_index(
        "ix_storage_transfer_jobs_snapshot_id", table_name="storage_transfer_jobs"
    )
    op.drop_index(
        "ix_storage_transfer_jobs_project_id", table_name="storage_transfer_jobs"
    )
    op.drop_table("storage_transfer_jobs")
    op.drop_index("ix_model_transfers_status", table_name="model_transfers")
    op.drop_index("ix_model_transfers_managed_model_id", table_name="model_transfers")
    op.drop_table("model_transfers")
    op.drop_index("ix_managed_models_status", table_name="managed_models")
    op.drop_table("managed_models")
