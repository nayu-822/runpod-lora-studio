"""add target-aware resume request fingerprint"""

import sqlalchemy as sa
from alembic import op

revision = "0014_phase6c_resume_request_fingerprint"
down_revision = "0013_phase6c_training_resume"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("resume_request_fingerprint", sa.String(64), nullable=True)
        )
    op.create_index(
        "uq_training_jobs_resume_request_fingerprint",
        "training_jobs",
        ["resume_request_fingerprint"],
        unique=True,
        sqlite_where=sa.text("resume_request_fingerprint IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_training_jobs_resume_request_fingerprint", table_name="training_jobs"
    )
    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.drop_column("resume_request_fingerprint")
