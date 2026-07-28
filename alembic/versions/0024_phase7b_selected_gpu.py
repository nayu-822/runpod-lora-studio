"""persist the GPU selected by the running training process"""

import sqlalchemy as sa
from alembic import op

revision = "0024_phase7b_selected_gpu"
down_revision = "0023_phase7b_memory_failure_codes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_execution_summaries") as batch_op:
        batch_op.add_column(sa.Column("physical_gpu_index", sa.Integer()))
        batch_op.add_column(sa.Column("compute_capability", sa.String(32)))
    with op.batch_alter_table("recommendation_calibration_snapshots") as batch_op:
        batch_op.add_column(sa.Column("compute_capability", sa.String(32)))
    op.create_table(
        "training_job_selected_gpus",
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
        sa.Column("gpu_uuid_fingerprint", sa.String(64), nullable=False),
        sa.Column("gpu_architecture", sa.String(128)),
        sa.Column("compute_capability", sa.String(32)),
        sa.Column("total_vram_bytes", sa.Integer()),
        sa.Column("selected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("selection_source", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("warning_codes_json", sa.Text(), nullable=False, server_default="[]"),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint("training_job_id", name="uq_training_job_selected_gpu_job"),
    )
    op.create_index(
        "ix_training_job_selected_gpu_job",
        "training_job_selected_gpus",
        ["training_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_job_selected_gpu_job", table_name="training_job_selected_gpus"
    )
    op.drop_table("training_job_selected_gpus")
    with op.batch_alter_table("training_execution_summaries") as batch_op:
        batch_op.drop_column("compute_capability")
        batch_op.drop_column("physical_gpu_index")
    with op.batch_alter_table("recommendation_calibration_snapshots") as batch_op:
        batch_op.drop_column("compute_capability")
