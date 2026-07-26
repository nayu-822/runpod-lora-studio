"""add phase 6A training configurations and jobs"""

import sqlalchemy as sa
from alembic import op

revision = "0010_phase6a_training_jobs"
down_revision = "0009_storage_transfer_progress"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_configs",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("managed_model_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("output_name", sa.String(length=255), nullable=False),
        sa.Column("output_directory", sa.Text(), nullable=False),
        sa.Column("sd_scripts_root", sa.Text(), nullable=False),
        sa.Column("trainer_script", sa.String(length=128), nullable=False),
        sa.Column("python_executable", sa.Text(), nullable=False),
        sa.Column("resolution", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("epochs", sa.Integer(), nullable=False),
        sa.Column("repeats", sa.Integer(), nullable=False),
        sa.Column("learning_rate", sa.Float(), nullable=False),
        sa.Column("optimizer", sa.String(length=128), nullable=False),
        sa.Column("scheduler", sa.String(length=128), nullable=False),
        sa.Column("network_module", sa.String(length=128), nullable=False),
        sa.Column("network_dim", sa.Integer(), nullable=False),
        sa.Column("network_alpha", sa.Integer(), nullable=False),
        sa.Column("mixed_precision", sa.String(length=16), nullable=False),
        sa.Column("save_every_n_epochs", sa.Integer(), nullable=False),
        sa.Column("cache_latents", sa.Integer(), nullable=False),
        sa.Column("gradient_checkpointing", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("extra_options", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["dataset_snapshot_id"], ["dataset_snapshots.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["managed_model_id"], ["managed_models.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("internal_id"),
        sa.UniqueConstraint("id"),
    )
    op.create_index("ix_training_configs_id", "training_configs", ["id"], unique=True)
    op.create_index(
        "ix_training_configs_project_id", "training_configs", ["project_id"]
    )
    op.create_index(
        "ix_training_configs_snapshot_id",
        "training_configs",
        ["dataset_snapshot_id"],
    )
    op.create_index(
        "ix_training_configs_model_id", "training_configs", ["managed_model_id"]
    )

    op.create_table(
        "training_jobs",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("training_config_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("managed_model_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("cancel_requested", sa.Integer(), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("worker_heartbeat", sa.DateTime(timezone=True), nullable=True),
        sa.Column("command_summary", sa.Text(), nullable=True),
        sa.Column("stdout_log_path", sa.Text(), nullable=True),
        sa.Column("stderr_log_path", sa.Text(), nullable=True),
        sa.Column("runtime_directory", sa.Text(), nullable=True),
        sa.Column("config_snapshot", sa.Text(), nullable=True),
        sa.Column("process_start_time", sa.Float(), nullable=True),
        sa.Column("process_group_id", sa.Integer(), nullable=True),
        sa.Column("process_identity", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("failure_code", sa.String(length=64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["training_config_id"], ["training_configs.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_snapshot_id"], ["dataset_snapshots.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["managed_model_id"], ["managed_models.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("internal_id"),
        sa.UniqueConstraint("id"),
    )
    op.create_index("ix_training_jobs_id", "training_jobs", ["id"], unique=True)
    op.create_index("ix_training_jobs_project_id", "training_jobs", ["project_id"])
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])
    op.create_index(
        "uq_training_jobs_active_project",
        "training_jobs",
        ["project_id"],
        unique=True,
        sqlite_where=sa.text(
            "status IN ('queued', 'starting', 'running', 'cancel_requested')"
        ),
    )


def downgrade() -> None:
    op.drop_index("uq_training_jobs_active_project", table_name="training_jobs")
    op.drop_index("ix_training_jobs_status", table_name="training_jobs")
    op.drop_index("ix_training_jobs_project_id", table_name="training_jobs")
    op.drop_index("ix_training_jobs_id", table_name="training_jobs")
    op.drop_table("training_jobs")
    op.drop_index("ix_training_configs_model_id", table_name="training_configs")
    op.drop_index("ix_training_configs_snapshot_id", table_name="training_configs")
    op.drop_index("ix_training_configs_project_id", table_name="training_configs")
    op.drop_index("ix_training_configs_id", table_name="training_configs")
    op.drop_table("training_configs")
