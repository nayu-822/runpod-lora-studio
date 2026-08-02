"""add Phase 8B part cleanup warning audit field"""

import sqlalchemy as sa
from alembic import op

revision = "0032_phase8b_part_cleanup_warnings"
down_revision = "0031_phase8b_integrity_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_acquisition_job_items",
        sa.Column("part_cleanup_warning", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("image_acquisition_job_items", "part_cleanup_warning")
