"""persist terminal manifest orphan scan progress"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0039_phase8c_manifest_orphan_scan"
down_revision = "0038_phase8c_legacy_manifest_recovery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_acquisition_jobs",
        sa.Column(
            "manifest_orphan_checked_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_image_acquisition_jobs_manifest_orphan_scan",
        "image_acquisition_jobs",
        ["manifest_orphan_checked_at", "updated_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_image_acquisition_jobs_manifest_orphan_scan",
        table_name="image_acquisition_jobs",
    )
    op.drop_column("image_acquisition_jobs", "manifest_orphan_checked_at")
