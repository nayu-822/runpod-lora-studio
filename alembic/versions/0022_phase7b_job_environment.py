"""capture execution-time GPU environment per training job"""

import sqlalchemy as sa
from alembic import op

revision = "0022_phase7b_job_environment"
down_revision = "0021_phase7b_calibration_compatibility"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_execution_summaries") as batch_op:
        batch_op.add_column(
            sa.Column("training_job_environment_snapshot_id", sa.String(36))
        )
        batch_op.add_column(
            sa.Column(
                "memory_warning_codes_json",
                sa.Text(),
                nullable=False,
                server_default="[]",
            )
        )
        batch_op.add_column(
            sa.Column(
                "memory_failure_codes_json",
                sa.Text(),
                nullable=False,
                server_default="[]",
            )
        )

    op.create_table(
        "training_job_environment_snapshots",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column(
            "training_job_id",
            sa.String(36),
            sa.ForeignKey("training_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("logical_gpu_index", sa.Integer()),
        sa.Column("physical_gpu_index", sa.Integer()),
        sa.Column("gpu_uuid_fingerprint", sa.String(64)),
        sa.Column("gpu_architecture", sa.String(128)),
        sa.Column("compute_capability", sa.String(32)),
        sa.Column("total_vram_bytes", sa.Integer()),
        sa.Column("cuda_available", sa.Boolean(), nullable=False),
        sa.Column("sd_scripts_version", sa.String(128)),
        sa.Column("xformers_available", sa.Boolean()),
        sa.Column("cuda_visible_devices", sa.String(512), nullable=False),
        sa.Column("visible_gpu_uuids_json", sa.Text(), nullable=False),
        sa.Column("detector_version", sa.String(64), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("warning_codes_json", sa.Text(), nullable=False),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint("training_job_id", name="uq_training_job_environment_job"),
    )
    op.create_index(
        "ix_training_job_environment_job",
        "training_job_environment_snapshots",
        ["training_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_job_environment_job",
        table_name="training_job_environment_snapshots",
    )
    op.drop_table("training_job_environment_snapshots")
    with op.batch_alter_table("training_execution_summaries") as batch_op:
        batch_op.drop_column("memory_failure_codes_json")
        batch_op.drop_column("memory_warning_codes_json")
        batch_op.drop_column("training_job_environment_snapshot_id")
