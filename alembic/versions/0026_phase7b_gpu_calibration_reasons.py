"""persist selected GPU classification in execution summaries"""

import sqlalchemy as sa
from alembic import op

revision = "0026_phase7b_gpu_calibration_reasons"
down_revision = "0025_phase7b_gpu_change_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_execution_summaries") as batch_op:
        batch_op.add_column(sa.Column("selected_gpu_status", sa.String(32)))
        batch_op.add_column(
            sa.Column(
                "selected_gpu_warning_codes_json",
                sa.Text(),
                nullable=False,
                server_default="[]",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("training_execution_summaries") as batch_op:
        batch_op.drop_column("selected_gpu_warning_codes_json")
        batch_op.drop_column("selected_gpu_status")
