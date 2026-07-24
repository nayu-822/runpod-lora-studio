"""persist Phase 2A image inspection results"""

import sqlalchemy as sa
from alembic import op

revision = "0003_phase2a_image_inspection"
down_revision = "0002_phase1_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_inspection_results",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("image_id", sa.String(length=36), nullable=False),
        sa.Column("rule_code", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("reason_ja", sa.Text(), nullable=False),
        sa.Column("detector_version", sa.String(length=32), nullable=False),
        sa.Column("inspected_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "image_id",
            "rule_code",
            "detector_version",
            name="uq_image_inspection_rule_version",
        ),
    )
    op.create_index(
        "ix_image_inspection_results_image_id",
        "image_inspection_results",
        ["image_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_image_inspection_results_image_id",
        table_name="image_inspection_results",
    )
    op.drop_table("image_inspection_results")
