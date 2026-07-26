"""remove unsafe training config overrides"""

import sqlalchemy as sa
from alembic import op

revision = "0011_remove_training_config_overrides"
down_revision = "0010_phase6a_training_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("training_configs", "python_executable")
    op.drop_column("training_configs", "repeats")


def downgrade() -> None:
    op.add_column(
        "training_configs",
        sa.Column("repeats", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "training_configs",
        sa.Column("python_executable", sa.Text(), nullable=False, server_default=""),
    )
