"""create Phase 1 project and image tables"""

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("concept_type", sa.String(length=32), nullable=False),
        sa.Column("trigger_words", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("id"),
    )
    op.create_index("ix_projects_id", "projects", ["id"], unique=True)
    op.create_table(
        "image_assets",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("original_path", sa.Text(), nullable=False),
        sa.Column("thumbnail_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(length=64), nullable=False),
        sa.Column("selection_state", sa.String(length=32), nullable=False),
        sa.Column("exclusion_reasons", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("selection_source", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.UniqueConstraint("id"),
    )
    op.create_index("ix_image_assets_id", "image_assets", ["id"], unique=True)
    op.create_index(
        "ix_image_assets_project_id", "image_assets", ["project_id"], unique=False
    )
    op.create_index("ix_image_assets_sha256", "image_assets", ["sha256"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_image_assets_sha256", table_name="image_assets")
    op.drop_index("ix_image_assets_project_id", table_name="image_assets")
    op.drop_index("ix_image_assets_id", table_name="image_assets")
    op.drop_table("image_assets")
    op.drop_index("ix_projects_id", table_name="projects")
    op.drop_table("projects")
