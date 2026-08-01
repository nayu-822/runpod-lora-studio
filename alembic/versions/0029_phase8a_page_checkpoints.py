"""make Phase 8A page checkpoints restartable across worker generations"""

import sqlalchemy as sa
from alembic import op

revision = "0029_phase8a_page_checkpoints"
down_revision = "0028_phase8a_claim_reservations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_source_searches",
        sa.Column("request_cursor", sa.String(128), nullable=True),
    )
    op.add_column(
        "image_source_searches",
        sa.Column("completion_reason", sa.String(32), nullable=True),
    )
    op.create_table(
        "image_source_search_cursor_checkpoints",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "search_id",
            sa.String(36),
            sa.ForeignKey("image_source_searches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("request_cursor_fingerprint", sa.String(64), nullable=False),
        sa.Column("next_cursor_fingerprint", sa.String(64), nullable=True),
        sa.Column("worker_generation", sa.Integer(), nullable=False),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "search_id",
            "request_cursor_fingerprint",
            name="uq_image_search_cursor_checkpoint_request",
        ),
    )
    op.create_index(
        "ix_image_search_cursor_checkpoints_id",
        "image_source_search_cursor_checkpoints",
        ["id"],
    )
    op.create_index(
        "ix_image_search_cursor_checkpoints_search",
        "image_source_search_cursor_checkpoints",
        ["search_id", "committed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_image_search_cursor_checkpoints_search",
        table_name="image_source_search_cursor_checkpoints",
    )
    op.drop_index(
        "ix_image_search_cursor_checkpoints_id",
        table_name="image_source_search_cursor_checkpoints",
    )
    op.drop_table("image_source_search_cursor_checkpoints")
    op.drop_column("image_source_searches", "completion_reason")
    op.drop_column("image_source_searches", "request_cursor")
