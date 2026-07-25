"""add cumulative transfer progress fields"""

import sqlalchemy as sa
from alembic import op

revision = "0009_storage_transfer_progress"
down_revision = "0008_storage_transfer_heartbeat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "storage_transfer_jobs",
        sa.Column(
            "completed_transferred_bytes",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "storage_transfer_jobs",
        sa.Column(
            "current_file_transferred_bytes",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("storage_transfer_jobs", "current_file_transferred_bytes")
    op.drop_column("storage_transfer_jobs", "completed_transferred_bytes")
