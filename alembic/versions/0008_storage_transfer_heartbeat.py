"""add storage transfer worker heartbeat and child pid"""

import sqlalchemy as sa
from alembic import op

revision = "0008_storage_transfer_heartbeat"
down_revision = "0007_phase5_storage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "storage_transfer_jobs",
        sa.Column("worker_id", sa.String(64), nullable=True),
    )
    op.add_column(
        "storage_transfer_jobs",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("storage_transfer_jobs", "heartbeat_at")
    op.drop_column("storage_transfer_jobs", "worker_id")
