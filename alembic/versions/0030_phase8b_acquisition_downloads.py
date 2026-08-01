"""add Phase 8B acquisition jobs, resumable items, and provenance"""

import sqlalchemy as sa
from alembic import op

revision = "0030_phase8b_acquisition_downloads"
down_revision = "0029_phase8a_page_checkpoints"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_acquisition_plan_items",
        sa.Column("expected_file_size", sa.Integer(), nullable=True),
    )
    op.create_table(
        "image_acquisition_jobs",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("plan_id", sa.String(36), nullable=False),
        sa.Column("plan_fingerprint", sa.String(64), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("active_key", sa.String(36), nullable=True),
        sa.Column("worker_id", sa.String(128), nullable=True),
        sa.Column(
            "worker_generation", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("claim_token", sa.String(64), nullable=True),
        sa.Column(
            "cancellation_requested", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("requested_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pending_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "downloading_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("downloaded_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("validated_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("imported_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "linked_existing_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("skipped_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("received_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("expected_bytes", sa.Integer(), nullable=True),
        sa.Column("current_item_id", sa.String(36), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("manifest_relative_path", sa.Text(), nullable=True),
        sa.Column("manifest_warning", sa.Text(), nullable=True),
        sa.Column("downloader_version", sa.String(64), nullable=False),
        sa.Column("validator_version", sa.String(64), nullable=False),
        sa.Column("importer_version", sa.String(64), nullable=False),
        sa.Column("job_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["plan_id"], ["image_acquisition_plans.id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint("active_key", name="uq_image_acquisition_job_active_key"),
    )
    op.create_index("ix_image_acquisition_jobs_id", "image_acquisition_jobs", ["id"])
    op.create_index(
        "ix_image_acquisition_jobs_project",
        "image_acquisition_jobs",
        ["project_id", "created_at"],
    )
    op.create_index(
        "ix_image_acquisition_jobs_plan",
        "image_acquisition_jobs",
        ["plan_id", "created_at"],
    )
    op.create_index(
        "ix_image_acquisition_jobs_status", "image_acquisition_jobs", ["status"]
    )

    op.create_table(
        "image_acquisition_job_items",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("plan_item_id", sa.String(36), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("external_post_id", sa.String(32), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("expected_metadata_fingerprint", sa.String(64), nullable=False),
        sa.Column("expected_file_url_fingerprint", sa.String(64), nullable=True),
        sa.Column("expected_md5", sa.String(128), nullable=True),
        sa.Column("expected_width", sa.Integer(), nullable=True),
        sa.Column("expected_height", sa.Integer(), nullable=True),
        sa.Column("expected_extension", sa.String(16), nullable=True),
        sa.Column("expected_file_size", sa.Integer(), nullable=True),
        sa.Column("expected_file_url", sa.Text(), nullable=True),
        sa.Column("received_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("part_relative_path", sa.Text(), nullable=False),
        sa.Column("etag", sa.String(256), nullable=True),
        sa.Column("last_modified", sa.String(256), nullable=True),
        sa.Column("accept_ranges", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("range_start", sa.Integer(), nullable=True),
        sa.Column("calculated_md5", sa.String(32), nullable=True),
        sa.Column("calculated_sha256", sa.String(64), nullable=True),
        sa.Column("detected_format", sa.String(16), nullable=True),
        sa.Column("detected_mime_type", sa.String(64), nullable=True),
        sa.Column("detected_width", sa.Integer(), nullable=True),
        sa.Column("detected_height", sa.Integer(), nullable=True),
        sa.Column("detected_file_size", sa.Integer(), nullable=True),
        sa.Column("image_asset_id", sa.String(36), nullable=True),
        sa.Column("failure_code", sa.String(64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("retryable", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["image_acquisition_jobs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["plan_item_id"], ["image_acquisition_plan_items.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["image_asset_id"], ["image_assets.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(
            "job_id", "plan_item_id", name="uq_image_acquisition_job_item_plan"
        ),
    )
    op.create_index(
        "ix_image_acquisition_job_items_id", "image_acquisition_job_items", ["id"]
    )
    op.create_index(
        "ix_image_acquisition_job_items_job",
        "image_acquisition_job_items",
        ["job_id", "display_order"],
    )
    op.create_index(
        "ix_image_acquisition_job_items_status",
        "image_acquisition_job_items",
        ["job_id", "status"],
    )
    op.create_index(
        "ix_image_acquisition_job_items_status_only",
        "image_acquisition_job_items",
        ["status"],
    )

    op.create_table(
        "image_acquisition_attempts",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("job_item_id", sa.String(36), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("failure_code", sa.String(64), nullable=True),
        sa.Column("retryable", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("received_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_item_id"], ["image_acquisition_job_items.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "job_item_id", "attempt_number", name="uq_image_acquisition_attempt_number"
        ),
    )
    op.create_index(
        "ix_image_acquisition_attempts_id", "image_acquisition_attempts", ["id"]
    )
    op.create_index(
        "ix_image_acquisition_attempts_item",
        "image_acquisition_attempts",
        ["job_item_id", "attempt_number"],
    )

    for column in (
        sa.Column("project_id", sa.String(36), nullable=True),
        sa.Column("source_md5", sa.String(128), nullable=True),
        sa.Column("source_metadata_fingerprint", sa.String(64), nullable=True),
        sa.Column("acquisition_plan_id", sa.String(36), nullable=True),
        sa.Column("acquisition_job_id", sa.String(36), nullable=True),
        sa.Column("acquisition_job_item_id", sa.String(36), nullable=True),
        sa.Column("linked_at", sa.DateTime(timezone=True), nullable=True),
    ):
        op.add_column("external_image_asset_links", column)
    op.execute(
        sa.text(
            "UPDATE external_image_asset_links "
            "SET project_id = (SELECT project_id FROM image_assets "
            "WHERE image_assets.id = external_image_asset_links.image_asset_id), "
            "linked_at = created_at "
            "WHERE project_id IS NULL"
        )
    )
    op.create_index(
        "ix_external_image_asset_links_project",
        "external_image_asset_links",
        ["project_id"],
    )
    with op.batch_alter_table("external_image_asset_links", recreate="always") as batch:
        batch.drop_constraint("uq_image_asset_source", type_="unique")
        batch.alter_column("project_id", existing_type=sa.String(36), nullable=False)
        batch.alter_column(
            "linked_at", existing_type=sa.DateTime(timezone=True), nullable=False
        )
        batch.create_foreign_key(
            "fk_external_link_project",
            "projects",
            ["project_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.create_foreign_key(
            "fk_external_link_plan",
            "image_acquisition_plans",
            ["acquisition_plan_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_foreign_key(
            "fk_external_link_job",
            "image_acquisition_jobs",
            ["acquisition_job_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_foreign_key(
            "fk_external_link_job_item",
            "image_acquisition_job_items",
            ["acquisition_job_item_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    op.drop_column("image_acquisition_plan_items", "expected_file_size")
    with op.batch_alter_table("external_image_asset_links", recreate="always") as batch:
        batch.drop_constraint("fk_external_link_job_item", type_="foreignkey")
        batch.drop_constraint("fk_external_link_job", type_="foreignkey")
        batch.drop_constraint("fk_external_link_plan", type_="foreignkey")
        batch.drop_constraint("fk_external_link_project", type_="foreignkey")
        batch.create_unique_constraint(
            "uq_image_asset_source", ["image_asset_id", "source_type"]
        )
    op.drop_index(
        "ix_external_image_asset_links_project", table_name="external_image_asset_links"
    )
    for name in (
        "linked_at",
        "acquisition_job_item_id",
        "acquisition_job_id",
        "acquisition_plan_id",
        "source_metadata_fingerprint",
        "source_md5",
        "project_id",
    ):
        op.drop_column("external_image_asset_links", name)
    op.drop_index(
        "ix_image_acquisition_attempts_item", table_name="image_acquisition_attempts"
    )
    op.drop_index(
        "ix_image_acquisition_attempts_id", table_name="image_acquisition_attempts"
    )
    op.drop_table("image_acquisition_attempts")
    op.drop_index(
        "ix_image_acquisition_job_items_status_only",
        table_name="image_acquisition_job_items",
    )
    op.drop_index(
        "ix_image_acquisition_job_items_status",
        table_name="image_acquisition_job_items",
    )
    op.drop_index(
        "ix_image_acquisition_job_items_job", table_name="image_acquisition_job_items"
    )
    op.drop_index(
        "ix_image_acquisition_job_items_id", table_name="image_acquisition_job_items"
    )
    op.drop_table("image_acquisition_job_items")
    op.drop_index(
        "ix_image_acquisition_jobs_status", table_name="image_acquisition_jobs"
    )
    op.drop_index("ix_image_acquisition_jobs_plan", table_name="image_acquisition_jobs")
    op.drop_index(
        "ix_image_acquisition_jobs_project", table_name="image_acquisition_jobs"
    )
    op.drop_index("ix_image_acquisition_jobs_id", table_name="image_acquisition_jobs")
    op.drop_table("image_acquisition_jobs")
