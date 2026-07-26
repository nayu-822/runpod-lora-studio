"""persist recommendation input configuration for safe revalidation"""

import sqlalchemy as sa
from alembic import op

revision = "0018_phase7a_recommendation_input_config"
down_revision = "0017_phase7a_recommendation_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_recommendation_requests") as batch_op:
        batch_op.add_column(
            sa.Column(
                "current_config_json", sa.Text(), nullable=False, server_default="{}"
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("training_recommendation_requests") as batch_op:
        batch_op.drop_column("current_config_json")
