"""persist validated resume state position and provenance"""

import sqlalchemy as sa
from alembic import op

revision = "0015_phase6c_state_position_provenance"
down_revision = "0014_phase6c_resume_request_fingerprint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.add_column(sa.Column("initial_epoch_source", sa.String(64)))
        batch_op.add_column(sa.Column("initial_step_source", sa.String(64)))
    with op.batch_alter_table("training_resume_validations") as batch_op:
        batch_op.add_column(sa.Column("state_epoch", sa.Integer()))
        batch_op.add_column(sa.Column("state_step", sa.Integer()))
        batch_op.add_column(sa.Column("state_epoch_source", sa.String(64)))
        batch_op.add_column(sa.Column("state_step_source", sa.String(64)))


def downgrade() -> None:
    with op.batch_alter_table("training_resume_validations") as batch_op:
        batch_op.drop_column("state_step_source")
        batch_op.drop_column("state_epoch_source")
        batch_op.drop_column("state_step")
        batch_op.drop_column("state_epoch")
    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.drop_column("initial_step_source")
        batch_op.drop_column("initial_epoch_source")
