"""persist deterministic GPU memory measurement warning and failure codes"""

import sqlalchemy as sa
from alembic import op

revision = "0023_phase7b_memory_failure_codes"
down_revision = "0022_phase7b_job_environment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_memory_aggregates") as batch_op:
        batch_op.add_column(
            sa.Column(
                "warning_codes_json",
                sa.Text(),
                nullable=False,
                server_default="[]",
            )
        )
        batch_op.add_column(
            sa.Column(
                "failure_codes_json",
                sa.Text(),
                nullable=False,
                server_default="[]",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("training_memory_aggregates") as batch_op:
        batch_op.drop_column("failure_codes_json")
        batch_op.drop_column("warning_codes_json")
