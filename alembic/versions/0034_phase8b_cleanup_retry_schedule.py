"""add Phase 8B cleanup retry scheduling"""

import sqlalchemy as sa
from alembic import op

revision = "0034_phase8b_cleanup_retry_schedule"
down_revision = "0033_phase8b_part_cleanup_claims"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_acquisition_job_items",
        sa.Column(
            "part_cleanup_next_retry_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.create_index(
        "ix_image_acquisition_job_items_part_cleanup_schedule",
        "image_acquisition_job_items",
        [
            "part_cleanup_warning",
            "part_cleanup_next_retry_at",
            "part_cleanup_claimed_at",
        ],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_image_acquisition_job_items_part_cleanup_schedule",
        table_name="image_acquisition_job_items",
    )
    op.drop_column("image_acquisition_job_items", "part_cleanup_next_retry_at")
