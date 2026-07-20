"""add Phase 1 query indexes"""

from alembic import op

revision = "0002_phase1_indexes"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_projects_updated_at", "projects", ["updated_at"])
    op.drop_index("ix_image_assets_project_id", table_name="image_assets")
    op.create_index(
        "ix_image_assets_project_state",
        "image_assets",
        ["project_id", "selection_state"],
    )


def downgrade() -> None:
    op.drop_index("ix_image_assets_project_state", table_name="image_assets")
    op.create_index("ix_image_assets_project_id", "image_assets", ["project_id"])
    op.drop_index("ix_projects_updated_at", table_name="projects")
