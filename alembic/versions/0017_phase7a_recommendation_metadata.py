"""record recommendation provenance on immutable training configurations"""

import sqlalchemy as sa
from alembic import op

revision = "0017_phase7a_recommendation_metadata"
down_revision = "0016_phase7a_recommendation_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_configs") as batch_op:
        batch_op.add_column(
            sa.Column("recommendation_id", sa.String(36), nullable=True)
        )
        batch_op.add_column(
            sa.Column("recommendation_engine_version", sa.String(32), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "recommendation_change_diff",
                sa.Text(),
                nullable=False,
                server_default="{}",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("training_configs") as batch_op:
        batch_op.drop_column("recommendation_change_diff")
        batch_op.drop_column("recommendation_engine_version")
        batch_op.drop_column("recommendation_id")
