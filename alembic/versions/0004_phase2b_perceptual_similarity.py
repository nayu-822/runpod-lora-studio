"""add Phase 2B perceptual hashes and similarity groups"""

import sqlalchemy as sa
from alembic import op

revision = "0004_phase2b_perceptual_similarity"
down_revision = "0003_phase2a_image_inspection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_perceptual_hashes",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("image_id", sa.String(36), nullable=False),
        sa.Column("algorithm", sa.String(32), nullable=False),
        sa.Column("hash_value", sa.String(256), nullable=True),
        sa.Column("hash_size", sa.Integer(), nullable=False),
        sa.Column("detector_version", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("calculated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "image_id",
            "algorithm",
            "hash_size",
            "detector_version",
            name="uq_image_phash_configuration",
        ),
    )
    op.create_index(
        "ix_image_perceptual_hashes_image_id",
        "image_perceptual_hashes",
        ["image_id"],
    )
    op.create_index(
        "ix_image_perceptual_hashes_lookup",
        "image_perceptual_hashes",
        ["algorithm", "hash_size", "detector_version", "status"],
    )

    op.create_table(
        "similarity_groups",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("group_type", sa.String(32), nullable=False),
        sa.Column("detector_version", sa.String(32), nullable=False),
        sa.Column("distance_threshold", sa.Integer(), nullable=False),
        sa.Column("representative_image_id", sa.String(36), nullable=True),
        sa.Column("representative_source", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["representative_image_id"], ["image_assets.id"], ondelete="SET NULL"
        ),
    )
    op.create_index(
        "ix_similarity_groups_project_id", "similarity_groups", ["project_id"]
    )

    op.create_table(
        "similarity_group_members",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("group_id", sa.String(36), nullable=False),
        sa.Column("image_id", sa.String(36), nullable=False),
        sa.Column("detector_version", sa.String(32), nullable=False),
        sa.Column("representative_candidate_score", sa.Float(), nullable=False),
        sa.Column("is_representative", sa.Integer(), nullable=False, default=0),
        sa.Column("representative_distance", sa.Integer(), nullable=True),
        sa.Column("minimum_distance", sa.Integer(), nullable=True),
        sa.Column("review_status", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"], ["similarity_groups.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("group_id", "image_id", name="uq_similarity_group_image"),
        sa.UniqueConstraint(
            "image_id", "detector_version", name="uq_similarity_image_version"
        ),
    )
    op.create_index(
        "ix_similarity_group_members_group_id",
        "similarity_group_members",
        ["group_id"],
    )
    op.create_index(
        "ix_similarity_group_members_image_id",
        "similarity_group_members",
        ["image_id"],
    )

    op.create_table(
        "similarity_pair_reviews",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("image_left_id", sa.String(36), nullable=False),
        sa.Column("image_right_id", sa.String(36), nullable=False),
        sa.Column("detector_version", sa.String(32), nullable=False),
        sa.Column("review_status", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["image_left_id"], ["image_assets.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["image_right_id"], ["image_assets.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "project_id",
            "image_left_id",
            "image_right_id",
            "detector_version",
            name="uq_similarity_pair_review",
        ),
    )
    op.create_index(
        "ix_similarity_pair_reviews_project_id",
        "similarity_pair_reviews",
        ["project_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_similarity_pair_reviews_project_id", table_name="similarity_pair_reviews"
    )
    op.drop_table("similarity_pair_reviews")
    op.drop_index(
        "ix_similarity_group_members_image_id", table_name="similarity_group_members"
    )
    op.drop_index(
        "ix_similarity_group_members_group_id", table_name="similarity_group_members"
    )
    op.drop_table("similarity_group_members")
    op.drop_index("ix_similarity_groups_project_id", table_name="similarity_groups")
    op.drop_table("similarity_groups")
    op.drop_index(
        "ix_image_perceptual_hashes_lookup", table_name="image_perceptual_hashes"
    )
    op.drop_index(
        "ix_image_perceptual_hashes_image_id", table_name="image_perceptual_hashes"
    )
    op.drop_table("image_perceptual_hashes")
