"""add phase 6B progress metrics and artifacts"""

import sqlalchemy as sa
from alembic import op

revision = "0012_phase6b_progress_artifacts"
down_revision = "0011_remove_training_config_overrides"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_progress",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("training_job_id", sa.String(36), nullable=False),
        sa.Column("current_epoch", sa.Integer(), nullable=True),
        sa.Column("total_epochs", sa.Integer(), nullable=True),
        sa.Column("current_step", sa.Integer(), nullable=True),
        sa.Column("total_steps", sa.Integer(), nullable=True),
        sa.Column("progress_ratio", sa.Float(), nullable=True),
        sa.Column("latest_loss", sa.Float(), nullable=True),
        sa.Column("smoothed_loss", sa.Float(), nullable=True),
        sa.Column("learning_rate", sa.Float(), nullable=True),
        sa.Column("steps_per_second", sa.Float(), nullable=True),
        sa.Column("samples_per_second", sa.Float(), nullable=True),
        sa.Column("elapsed_seconds", sa.Float(), nullable=True),
        sa.Column("estimated_remaining_seconds", sa.Float(), nullable=True),
        sa.Column("latest_log_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stdout_offset", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stderr_offset", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("parser_version", sa.String(32), nullable=False),
        sa.Column("parse_status", sa.String(16), nullable=False),
        sa.Column("parse_warning", sa.Text(), nullable=True),
        sa.Column("progress_source", sa.String(16), nullable=False),
        sa.Column("parser_state", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["training_job_id"], ["training_jobs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("internal_id"),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint("training_job_id", name="uq_training_progress_job"),
    )
    op.create_index("ix_training_progress_id", "training_progress", ["id"], unique=True)
    op.create_table(
        "training_metric_points",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("training_job_id", sa.String(36), nullable=False),
        sa.Column("metric_name", sa.String(64), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=True),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("logged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["training_job_id"], ["training_jobs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("internal_id"),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint(
            "training_job_id", "metric_name", "step", name="uq_training_metric_step"
        ),
    )
    op.create_index(
        "ix_training_metric_points_id", "training_metric_points", ["id"], unique=True
    )
    op.create_index(
        "ix_training_metrics_job_name_step",
        "training_metric_points",
        ["training_job_id", "metric_name", "step"],
    )
    op.create_table(
        "training_artifacts",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("training_job_id", sa.String(36), nullable=False),
        sa.Column("artifact_type", sa.String(32), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=True),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("validation_status", sa.String(16), nullable=False),
        sa.Column("validation_code", sa.String(64), nullable=True),
        sa.Column("validation_message", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["training_job_id"], ["training_jobs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("internal_id"),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint(
            "training_job_id", "relative_path", name="uq_training_artifact_path"
        ),
    )
    op.create_index(
        "ix_training_artifacts_id", "training_artifacts", ["id"], unique=True
    )
    op.create_index(
        "ix_training_artifacts_job", "training_artifacts", ["training_job_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_training_artifacts_job", table_name="training_artifacts")
    op.drop_index("ix_training_artifacts_id", table_name="training_artifacts")
    op.drop_table("training_artifacts")
    op.drop_index(
        "ix_training_metrics_job_name_step", table_name="training_metric_points"
    )
    op.drop_index("ix_training_metric_points_id", table_name="training_metric_points")
    op.drop_table("training_metric_points")
    op.drop_index("ix_training_progress_id", table_name="training_progress")
    op.drop_table("training_progress")
