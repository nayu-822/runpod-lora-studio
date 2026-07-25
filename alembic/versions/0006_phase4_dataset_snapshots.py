"""add Phase 4 dataset snapshot tables"""

import sqlalchemy as sa
from alembic import op

revision = "0006_phase4_dataset_snapshots"
down_revision = "0005_phase3_tagging_caption"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_snapshots",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("snapshot_version", sa.String(32), nullable=False),
        sa.Column("generator_version", sa.String(64), nullable=False),
        sa.Column("source_project_version", sa.String(64), nullable=False),
        sa.Column("source_tagger_run_id", sa.String(36), nullable=True),
        sa.Column("source_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("target_image_count", sa.Integer(), nullable=False),
        sa.Column("copied_image_count", sa.Integer(), nullable=False),
        sa.Column("failed_image_count", sa.Integer(), nullable=False),
        sa.Column("warning_count", sa.Integer(), nullable=False),
        sa.Column("total_size_bytes", sa.Integer(), nullable=False),
        sa.Column("snapshot_root", sa.Text(), nullable=False),
        sa.Column("dataset_toml_path", sa.Text(), nullable=False),
        sa.Column("manifest_path", sa.Text(), nullable=False),
        sa.Column("report_path", sa.Text(), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=True),
        sa.Column("dataset_toml_sha256", sa.String(64), nullable=True),
        sa.Column("content_sha256", sa.String(64), nullable=True),
        sa.Column("settings_snapshot", sa.Text(), nullable=False),
        sa.Column("validation_summary", sa.Text(), nullable=False),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_tagger_run_id"], ["tagger_runs.id"], ondelete="SET NULL"
        ),
    )
    op.create_index(
        "ix_dataset_snapshots_project_id", "dataset_snapshots", ["project_id"]
    )
    op.create_index("ix_dataset_snapshots_status", "dataset_snapshots", ["status"])

    op.create_table(
        "dataset_snapshot_items",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("image_id", sa.String(36), nullable=False),
        sa.Column("source_image_path", sa.Text(), nullable=False),
        sa.Column("snapshot_image_relative_path", sa.Text(), nullable=False),
        sa.Column("caption_relative_path", sa.Text(), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("source_image_sha256", sa.String(64), nullable=False),
        sa.Column("snapshot_image_sha256", sa.String(64), nullable=False),
        sa.Column("source_file_size", sa.Integer(), nullable=False),
        sa.Column("snapshot_file_size", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("aspect_ratio", sa.Float(), nullable=False),
        sa.Column("mime_type", sa.String(64), nullable=False),
        sa.Column("caption_id", sa.String(36), nullable=False),
        sa.Column("caption_revision", sa.Integer(), nullable=False),
        sa.Column("caption_sha256", sa.String(64), nullable=False),
        sa.Column("caption_text", sa.Text(), nullable=False),
        sa.Column("tag_count", sa.Integer(), nullable=False),
        sa.Column("trigger_word_count", sa.Integer(), nullable=False),
        sa.Column("quality_status", sa.String(24), nullable=False),
        sa.Column("exact_duplicate_status", sa.String(32), nullable=False),
        sa.Column("similarity_group_id", sa.String(36), nullable=True),
        sa.Column("is_similarity_representative", sa.Integer(), nullable=True),
        sa.Column("warnings_snapshot", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["dataset_snapshots.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint(
            "snapshot_id", "image_id", name="uq_dataset_snapshot_item_image"
        ),
        sa.UniqueConstraint(
            "snapshot_id", "sequence_number", name="uq_dataset_snapshot_item_seq"
        ),
    )
    op.create_index(
        "ix_dataset_snapshot_items_snapshot_id",
        "dataset_snapshot_items",
        ["snapshot_id"],
    )
    op.create_index(
        "ix_dataset_snapshot_items_image_id", "dataset_snapshot_items", ["image_id"]
    )

    op.create_table(
        "dataset_validation_issues",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("image_id", sa.String(36), nullable=True),
        sa.Column("issue_code", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("category", sa.String(24), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("measured_value", sa.Text(), nullable=True),
        sa.Column("threshold_value", sa.Text(), nullable=True),
        sa.Column("details", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["dataset_snapshots.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="SET NULL"),
    )
    op.create_index(
        "ix_dataset_validation_issues_snapshot_id",
        "dataset_validation_issues",
        ["snapshot_id"],
    )

    op.create_table(
        "snapshot_creation_jobs",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("snapshot_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("cancel_requested", sa.Integer(), nullable=False),
        sa.Column("current_step", sa.String(64), nullable=False),
        sa.Column("processed_count", sa.Integer(), nullable=False),
        sa.Column("total_count", sa.Integer(), nullable=False),
        sa.Column("current_image_id", sa.String(36), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"], ["dataset_snapshots.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_snapshot_creation_jobs_snapshot_id",
        "snapshot_creation_jobs",
        ["snapshot_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_snapshot_creation_jobs_snapshot_id", table_name="snapshot_creation_jobs"
    )
    op.drop_table("snapshot_creation_jobs")
    op.drop_index(
        "ix_dataset_validation_issues_snapshot_id",
        table_name="dataset_validation_issues",
    )
    op.drop_table("dataset_validation_issues")
    op.drop_index(
        "ix_dataset_snapshot_items_image_id", table_name="dataset_snapshot_items"
    )
    op.drop_index(
        "ix_dataset_snapshot_items_snapshot_id", table_name="dataset_snapshot_items"
    )
    op.drop_table("dataset_snapshot_items")
    op.drop_index("ix_dataset_snapshots_status", table_name="dataset_snapshots")
    op.drop_index("ix_dataset_snapshots_project_id", table_name="dataset_snapshots")
    op.drop_table("dataset_snapshots")
