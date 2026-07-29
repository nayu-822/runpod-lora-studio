"""add Phase 8A image source search and acquisition plan tables"""

import sqlalchemy as sa
from alembic import op

revision = "0027_phase8a_image_acquisition"
down_revision = "0026_phase7b_gpu_calibration_reasons"
branch_labels = None
depends_on = None


def _timestamps(table: str) -> None:
    del table


def upgrade() -> None:
    op.create_table(
        "external_image_posts",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("external_post_id", sa.String(32), nullable=False),
        sa.Column("post_url", sa.Text(), nullable=False),
        sa.Column("file_url", sa.Text()),
        sa.Column("preview_url", sa.Text()),
        sa.Column("sample_url", sa.Text()),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("file_size", sa.Integer()),
        sa.Column("file_extension", sa.String(16)),
        sa.Column("rating", sa.String(8)),
        sa.Column("score", sa.Integer()),
        sa.Column("source_md5", sa.String(128)),
        sa.Column(
            "normalized_tags_json", sa.Text(), nullable=False, server_default="[]"
        ),
        sa.Column("metadata_fingerprint", sa.String(64), nullable=False),
        sa.Column(
            "source_metadata_json", sa.Text(), nullable=False, server_default="{}"
        ),
        sa.Column("is_deleted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_pending", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_flagged", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "source_type", "external_post_id", name="uq_external_image_source_post"
        ),
    )
    op.create_index("ix_external_image_posts_id", "external_image_posts", ["id"])
    op.create_index(
        "ix_external_image_posts_source_md5",
        "external_image_posts",
        ["source_type", "source_md5"],
    )
    op.create_table(
        "image_source_searches",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("normalized_query", sa.Text(), nullable=False),
        sa.Column("query_fingerprint", sa.String(64), nullable=False),
        sa.Column("query_version", sa.String(64), nullable=False),
        sa.Column("adapter_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_candidate_count", sa.Integer(), nullable=False),
        sa.Column(
            "returned_post_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "accepted_candidate_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "excluded_candidate_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "api_request_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rate_limit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_cursor", sa.String(128)),
        sa.Column(
            "cancellation_requested", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("worker_id", sa.String(128)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(64)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_image_source_searches_id", "image_source_searches", ["id"])
    op.create_index(
        "ix_image_source_searches_project", "image_source_searches", ["project_id"]
    )
    op.create_table(
        "image_source_search_results",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "search_id",
            sa.String(36),
            sa.ForeignKey("image_source_searches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_post_id", sa.String(32), nullable=False),
        sa.Column("result_order", sa.Integer(), nullable=False),
        sa.Column("candidate_status", sa.String(24), nullable=False),
        sa.Column(
            "exclusion_reasons_json", sa.Text(), nullable=False, server_default="[]"
        ),
        sa.Column("already_imported", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("already_planned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("selected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metadata_fingerprint_at_search", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "search_id", "external_post_id", name="uq_image_search_result_post"
        ),
    )
    op.create_index(
        "ix_image_source_search_results_id", "image_source_search_results", ["id"]
    )
    op.create_index(
        "ix_image_source_search_results_search",
        "image_source_search_results",
        ["search_id"],
    )
    op.create_table(
        "image_acquisition_plans",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column(
            "source_search_id",
            sa.String(36),
            sa.ForeignKey("image_source_searches.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("selected_count", sa.Integer(), nullable=False),
        sa.Column(
            "skipped_existing_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("blocked_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("plan_fingerprint", sa.String(64), nullable=False, unique=True),
        sa.Column("plan_version", sa.String(64), nullable=False),
        sa.Column("query_fingerprint", sa.String(64), nullable=False),
        sa.Column("adapter_version", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_image_acquisition_plans_id", "image_acquisition_plans", ["id"])
    op.create_index(
        "ix_image_acquisition_plans_project", "image_acquisition_plans", ["project_id"]
    )
    op.create_table(
        "image_acquisition_plan_items",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "plan_id",
            sa.String(36),
            sa.ForeignKey("image_acquisition_plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_post_id", sa.String(32), nullable=False),
        sa.Column(
            "search_result_id",
            sa.String(36),
            sa.ForeignKey("image_source_search_results.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.Column("planned_status", sa.String(24), nullable=False),
        sa.Column("expected_metadata_fingerprint", sa.String(64), nullable=False),
        sa.Column("expected_file_url_fingerprint", sa.String(64)),
        sa.Column("expected_md5", sa.String(128)),
        sa.Column("expected_width", sa.Integer()),
        sa.Column("expected_height", sa.Integer()),
        sa.Column("expected_extension", sa.String(16)),
        sa.Column("skip_reason", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "plan_id", "external_post_id", name="uq_image_plan_item_post"
        ),
    )
    op.create_index(
        "ix_image_acquisition_plan_items_id", "image_acquisition_plan_items", ["id"]
    )
    op.create_index(
        "ix_image_acquisition_plan_items_plan",
        "image_acquisition_plan_items",
        ["plan_id"],
    )
    op.create_table(
        "external_image_asset_links",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "image_asset_id",
            sa.String(36),
            sa.ForeignKey("image_assets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_post_id", sa.String(32), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "source_type",
            "external_post_id",
            name="uq_external_image_asset_source_post",
        ),
        sa.UniqueConstraint(
            "image_asset_id", "source_type", name="uq_image_asset_source"
        ),
    )
    op.create_index(
        "ix_external_image_asset_links_id", "external_image_asset_links", ["id"]
    )


def downgrade() -> None:
    op.drop_index(
        "ix_external_image_asset_links_id", table_name="external_image_asset_links"
    )
    op.drop_table("external_image_asset_links")
    op.drop_index(
        "ix_image_acquisition_plan_items_plan",
        table_name="image_acquisition_plan_items",
    )
    op.drop_index(
        "ix_image_acquisition_plan_items_id", table_name="image_acquisition_plan_items"
    )
    op.drop_table("image_acquisition_plan_items")
    op.drop_index(
        "ix_image_acquisition_plans_project", table_name="image_acquisition_plans"
    )
    op.drop_index("ix_image_acquisition_plans_id", table_name="image_acquisition_plans")
    op.drop_table("image_acquisition_plans")
    op.drop_index(
        "ix_image_source_search_results_search",
        table_name="image_source_search_results",
    )
    op.drop_index(
        "ix_image_source_search_results_id", table_name="image_source_search_results"
    )
    op.drop_table("image_source_search_results")
    op.drop_index(
        "ix_image_source_searches_project", table_name="image_source_searches"
    )
    op.drop_index("ix_image_source_searches_id", table_name="image_source_searches")
    op.drop_table("image_source_searches")
    op.drop_index(
        "ix_external_image_posts_source_md5", table_name="external_image_posts"
    )
    op.drop_index("ix_external_image_posts_id", table_name="external_image_posts")
    op.drop_table("external_image_posts")
