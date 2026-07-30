"""make Phase 8A worker claims and plan reservations durable"""

import sqlalchemy as sa
from alembic import op

revision = "0028_phase8a_claim_reservations"
down_revision = "0027_phase8a_image_acquisition"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_source_searches",
        sa.Column(
            "worker_generation", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "image_source_searches", sa.Column("claim_token", sa.String(64), nullable=True)
    )
    op.create_table(
        "image_acquisition_reservations",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "plan_id",
            sa.String(36),
            sa.ForeignKey("image_acquisition_plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("external_post_id", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "source_type",
            "external_post_id",
            name="uq_image_acquisition_reservation_source_post",
        ),
    )
    op.create_index(
        "ix_image_acquisition_reservations_id",
        "image_acquisition_reservations",
        ["id"],
    )
    op.create_index(
        "ix_image_acquisition_reservations_plan",
        "image_acquisition_reservations",
        ["plan_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_image_acquisition_reservations_plan",
        table_name="image_acquisition_reservations",
    )
    op.drop_index(
        "ix_image_acquisition_reservations_id",
        table_name="image_acquisition_reservations",
    )
    op.drop_table("image_acquisition_reservations")
    op.drop_column("image_source_searches", "claim_token")
    op.drop_column("image_source_searches", "worker_generation")
