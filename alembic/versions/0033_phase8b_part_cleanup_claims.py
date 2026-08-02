"""add Phase 8B part cleanup claims"""

import sqlalchemy as sa
from alembic import op

revision = "0033_phase8b_part_cleanup_claims"
down_revision = "0032_phase8b_part_cleanup_warnings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_acquisition_job_items",
        sa.Column("part_cleanup_claim_token", sa.String(64), nullable=True),
    )
    op.add_column(
        "image_acquisition_job_items",
        sa.Column("part_cleanup_claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_image_acquisition_job_items_part_cleanup",
        "image_acquisition_job_items",
        ["part_cleanup_warning", "part_cleanup_claimed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_image_acquisition_job_items_part_cleanup",
        table_name="image_acquisition_job_items",
    )
    op.drop_column("image_acquisition_job_items", "part_cleanup_claimed_at")
    op.drop_column("image_acquisition_job_items", "part_cleanup_claim_token")
