"""record GPU changes without mutating the first selected identity"""

import sqlalchemy as sa
from alembic import op

revision = "0025_phase7b_gpu_change_audit"
down_revision = "0024_phase7b_selected_gpu"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_job_selected_gpus") as batch_op:
        batch_op.add_column(
            sa.Column("last_observed_gpu_uuid_fingerprint", sa.String(64))
        )
        batch_op.add_column(
            sa.Column("gpu_change_detected_at", sa.DateTime(timezone=True))
        )
        batch_op.add_column(
            sa.Column(
                "gpu_change_count", sa.Integer(), nullable=False, server_default="0"
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("training_job_selected_gpus") as batch_op:
        batch_op.drop_column("gpu_change_count")
        batch_op.drop_column("gpu_change_detected_at")
        batch_op.drop_column("last_observed_gpu_uuid_fingerprint")
