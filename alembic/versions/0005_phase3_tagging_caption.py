"""add Phase 3 tagging and caption editing tables"""

import sqlalchemy as sa
from alembic import op

revision = "0005_phase3_tagging_caption"
down_revision = "0004_phase2b_perceptual_similarity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tagger_runs",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("adapter_name", sa.String(64), nullable=False),
        sa.Column("model_identifier", sa.String(256), nullable=False),
        sa.Column("model_revision", sa.String(128), nullable=False),
        sa.Column("model_path", sa.Text(), nullable=False),
        sa.Column("device", sa.String(16), nullable=False),
        sa.Column("general_threshold", sa.Float(), nullable=False),
        sa.Column("character_threshold", sa.Float(), nullable=False),
        sa.Column("save_rating", sa.Integer(), nullable=False),
        sa.Column("save_character", sa.Integer(), nullable=False),
        sa.Column("save_general", sa.Integer(), nullable=False),
        sa.Column("underscore_to_space", sa.Integer(), nullable=False),
        sa.Column("escape_mode", sa.String(16), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("max_workers", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("target_image_count", sa.Integer(), nullable=False),
        sa.Column("processed_image_count", sa.Integer(), nullable=False),
        sa.Column("succeeded_image_count", sa.Integer(), nullable=False),
        sa.Column("failed_image_count", sa.Integer(), nullable=False),
        sa.Column("skipped_image_count", sa.Integer(), nullable=False),
        sa.Column("current_image_id", sa.String(36), nullable=True),
        sa.Column("cancel_requested", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("settings_snapshot", sa.Text(), nullable=False),
        sa.Column("implementation_version", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_tagger_runs_project_id", "tagger_runs", ["project_id"])
    op.create_index("ix_tagger_runs_status", "tagger_runs", ["status"])

    op.create_table(
        "image_tagging_results",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("image_id", sa.String(36), nullable=False),
        sa.Column("tagger_run_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("tagged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_output", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["tagger_run_id"], ["tagger_runs.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "image_id", "tagger_run_id", name="uq_image_tagging_result_run"
        ),
    )
    op.create_index(
        "ix_image_tagging_results_image_id", "image_tagging_results", ["image_id"]
    )
    op.create_index(
        "ix_image_tagging_results_tagger_run_id",
        "image_tagging_results",
        ["tagger_run_id"],
    )

    op.create_table(
        "detected_tags",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("image_tagging_result_id", sa.Integer(), nullable=False),
        sa.Column("tag_name_raw", sa.String(512), nullable=False),
        sa.Column("tag_name_normalized", sa.String(512), nullable=False),
        sa.Column("category", sa.String(24), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("original_order", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("model_identifier", sa.String(256), nullable=False),
        sa.Column("model_revision", sa.String(128), nullable=False),
        sa.Column("tagger_run_id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["image_tagging_result_id"],
            ["image_tagging_results.internal_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "image_tagging_result_id",
            "original_order",
            name="uq_detected_tag_order",
        ),
    )
    op.create_index(
        "ix_detected_tags_result_id", "detected_tags", ["image_tagging_result_id"]
    )

    op.create_table(
        "image_captions",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("image_id", sa.String(36), nullable=False),
        sa.Column("source_tagger_run_id", sa.String(36), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("caption_text", sa.Text(), nullable=False),
        sa.Column("caption_format_version", sa.String(32), nullable=False),
        sa.Column("is_current", sa.Integer(), nullable=False),
        sa.Column("edit_source", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_tagger_run_id"], ["tagger_runs.id"], ondelete="SET NULL"
        ),
    )
    op.create_index("ix_image_captions_image_id", "image_captions", ["image_id"])
    op.create_index(
        "uq_image_caption_current",
        "image_captions",
        ["image_id"],
        unique=True,
        sqlite_where=sa.text("is_current = 1"),
    )

    op.create_table(
        "caption_tags",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("image_caption_id", sa.Integer(), nullable=False),
        sa.Column("tag_name", sa.String(512), nullable=False),
        sa.Column("normalized_name", sa.String(512), nullable=False),
        sa.Column("category", sa.String(24), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("manually_added", sa.Integer(), nullable=False),
        sa.Column("manually_removed", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["image_caption_id"], ["image_captions.internal_id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "image_caption_id", "position", name="uq_caption_tag_position"
        ),
    )
    op.create_index("ix_caption_tags_caption_id", "caption_tags", ["image_caption_id"])

    op.create_table(
        "project_tag_rules",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("normalized_tag_name", sa.String(512), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("category", sa.String(24), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_by", sa.String(24), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "project_id", "normalized_tag_name", name="uq_project_tag_rule_name"
        ),
    )
    op.create_index(
        "ix_project_tag_rules_project_id", "project_tag_rules", ["project_id"]
    )

    op.create_table(
        "caption_edit_history",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("image_id", sa.String(36), nullable=False),
        sa.Column("image_caption_id", sa.Integer(), nullable=True),
        sa.Column("previous_revision", sa.Integer(), nullable=True),
        sa.Column("new_revision", sa.Integer(), nullable=False),
        sa.Column("before_text", sa.Text(), nullable=False),
        sa.Column("after_text", sa.Text(), nullable=False),
        sa.Column("diff_snapshot", sa.Text(), nullable=False),
        sa.Column("edit_source", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["image_caption_id"], ["image_captions.internal_id"], ondelete="SET NULL"
        ),
    )
    op.create_index(
        "ix_caption_edit_history_image_id", "caption_edit_history", ["image_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_caption_edit_history_image_id", table_name="caption_edit_history")
    op.drop_table("caption_edit_history")
    op.drop_index("ix_project_tag_rules_project_id", table_name="project_tag_rules")
    op.drop_table("project_tag_rules")
    op.drop_index("ix_caption_tags_caption_id", table_name="caption_tags")
    op.drop_table("caption_tags")
    op.drop_index("uq_image_caption_current", table_name="image_captions")
    op.drop_index("ix_image_captions_image_id", table_name="image_captions")
    op.drop_table("image_captions")
    op.drop_index("ix_detected_tags_result_id", table_name="detected_tags")
    op.drop_table("detected_tags")
    op.drop_index(
        "ix_image_tagging_results_tagger_run_id", table_name="image_tagging_results"
    )
    op.drop_index(
        "ix_image_tagging_results_image_id", table_name="image_tagging_results"
    )
    op.drop_table("image_tagging_results")
    op.drop_index("ix_tagger_runs_status", table_name="tagger_runs")
    op.drop_index("ix_tagger_runs_project_id", table_name="tagger_runs")
    op.drop_table("tagger_runs")
