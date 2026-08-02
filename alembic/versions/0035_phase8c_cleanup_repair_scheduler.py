"""add cleanup attempts and manifest repair state"""

import sqlalchemy as sa
from alembic import op

revision = "0035_phase8c_cleanup_repair_scheduler"
down_revision = "0034_phase8b_cleanup_retry_schedule"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_acquisition_job_items",
        sa.Column(
            "part_cleanup_attempt_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "image_acquisition_jobs",
        sa.Column("manifest_repair_state", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "image_acquisition_jobs",
        sa.Column(
            "manifest_repair_attempted_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("image_acquisition_jobs", "manifest_repair_attempted_at")
    op.drop_column("image_acquisition_jobs", "manifest_repair_state")
    op.drop_column("image_acquisition_job_items", "part_cleanup_attempt_count")
