"""store training performance and recommendation calibration snapshots"""

import sqlalchemy as sa
from alembic import op

revision = "0019_phase7b_training_performance"
down_revision = "0018_phase7a_recommendation_input_config"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_recommendations") as batch_op:
        batch_op.add_column(sa.Column("calibration_snapshot_id", sa.String(36)))
        batch_op.add_column(
            sa.Column(
                "calibration_applied",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.add_column(sa.Column("calibration_confidence", sa.String(16)))
        batch_op.add_column(
            sa.Column(
                "calibration_reason_codes_json",
                sa.Text(),
                nullable=False,
                server_default="[]",
            )
        )
        batch_op.add_column(sa.Column("baseline_duration_seconds", sa.Float()))
        batch_op.add_column(sa.Column("calibrated_duration_seconds", sa.Float()))
        batch_op.add_column(sa.Column("baseline_vram_bytes", sa.Integer()))
        batch_op.add_column(sa.Column("calibrated_vram_bytes", sa.Integer()))
        batch_op.add_column(sa.Column("baseline_batch_size", sa.Integer()))
        batch_op.add_column(sa.Column("calibrated_batch_size", sa.Integer()))
        batch_op.add_column(sa.Column("calibration_fingerprint", sa.String(64)))

    op.create_table(
        "training_execution_summaries",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column(
            "training_job_id",
            sa.String(36),
            sa.ForeignKey("training_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "training_config_id",
            sa.String(36),
            sa.ForeignKey("training_configs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("recommendation_id", sa.String(36)),
        sa.Column("environment_snapshot_id", sa.String(36)),
        sa.Column(
            "dataset_snapshot_id",
            sa.String(36),
            sa.ForeignKey("dataset_snapshots.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "managed_model_id",
            sa.String(36),
            sa.ForeignKey("managed_models.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("job_result_status", sa.String(32), nullable=False),
        sa.Column("gpu_identity_fingerprint", sa.String(128)),
        sa.Column("gpu_total_vram_bytes", sa.Integer()),
        sa.Column("dataset_scale_fingerprint", sa.String(64)),
        sa.Column("settings_fingerprint", sa.String(64)),
        sa.Column("resolution", sa.Integer()),
        sa.Column("batch_size", sa.Integer()),
        sa.Column("gradient_accumulation_steps", sa.Integer()),
        sa.Column("effective_batch_size", sa.Integer()),
        sa.Column("network_module", sa.String(128)),
        sa.Column("network_dim", sa.Integer()),
        sa.Column("network_alpha", sa.Integer()),
        sa.Column("optimizer", sa.String(128)),
        sa.Column("scheduler", sa.String(128)),
        sa.Column("mixed_precision", sa.String(16)),
        sa.Column("cache_latents", sa.Boolean()),
        sa.Column("gradient_checkpointing", sa.Boolean()),
        sa.Column("total_epochs", sa.Integer()),
        sa.Column("planned_total_steps", sa.Integer()),
        sa.Column("completed_steps", sa.Integer()),
        sa.Column("resume_initial_step", sa.Integer()),
        sa.Column("resume_step_mode", sa.String(32)),
        sa.Column("elapsed_seconds", sa.Float()),
        sa.Column("measured_steps_per_second", sa.Float()),
        sa.Column("measured_images_per_second", sa.Float()),
        sa.Column("peak_allocated_vram_bytes", sa.Integer()),
        sa.Column("peak_reserved_vram_bytes", sa.Integer()),
        sa.Column("free_vram_before_bytes", sa.Integer()),
        sa.Column("free_vram_after_bytes", sa.Integer()),
        sa.Column(
            "memory_sample_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "memory_confidence", sa.String(16), nullable=False, server_default="none"
        ),
        sa.Column("exit_code", sa.Integer()),
        sa.Column(
            "oom_detected", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "failure_category", sa.String(32), nullable=False, server_default="none"
        ),
        sa.Column(
            "failure_evidence_codes_json",
            sa.Text(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "usable_for_speed_calibration",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "usable_for_memory_calibration",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "exclusion_reasons_json", sa.Text(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "calibration_included",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column("manual_exclusion_reason", sa.Text()),
        sa.Column("collector_version", sa.String(32), nullable=False),
        sa.Column("summary_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint("training_job_id", name="uq_execution_summary_job"),
        sa.UniqueConstraint(
            "summary_fingerprint", name="uq_execution_summary_fingerprint"
        ),
    )
    op.create_index(
        "ix_execution_summaries_project", "training_execution_summaries", ["project_id"]
    )
    op.create_index(
        "ix_execution_summaries_gpu_settings",
        "training_execution_summaries",
        ["gpu_identity_fingerprint", "settings_fingerprint"],
    )

    op.create_table(
        "recommendation_calibration_snapshots",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column(
            "scope_project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
        ),
        sa.Column("gpu_identity_fingerprint", sa.String(128), nullable=False),
        sa.Column("gpu_total_vram_class", sa.String(32)),
        sa.Column("resolution", sa.Integer()),
        sa.Column("optimizer", sa.String(128)),
        sa.Column("mixed_precision", sa.String(16)),
        sa.Column("cache_latents", sa.Boolean()),
        sa.Column("gradient_checkpointing", sa.Boolean()),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("successful_sample_count", sa.Integer(), nullable=False),
        sa.Column("oom_sample_count", sa.Integer(), nullable=False),
        sa.Column("median_steps_per_second", sa.Float()),
        sa.Column("lower_percentile_steps_per_second", sa.Float()),
        sa.Column("median_peak_vram_bytes", sa.Integer()),
        sa.Column("upper_percentile_peak_vram_bytes", sa.Integer()),
        sa.Column("confidence", sa.String(16), nullable=False),
        sa.Column("calibration_fingerprint", sa.String(64), nullable=False),
        sa.Column("calibration_version", sa.String(32), nullable=False),
        sa.Column("source_summary_fingerprint", sa.String(64), nullable=False),
        sa.Column("reason_codes_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("stale", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint(
            "calibration_fingerprint", name="uq_calibration_fingerprint"
        ),
    )
    op.create_index(
        "ix_calibration_snapshots_scope",
        "recommendation_calibration_snapshots",
        ["scope_project_id"],
    )
    op.create_index(
        "ix_calibration_snapshots_match",
        "recommendation_calibration_snapshots",
        ["gpu_identity_fingerprint", "resolution"],
    )

    op.create_table(
        "recommendation_calibration_sources",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column(
            "calibration_id",
            sa.String(36),
            sa.ForeignKey(
                "recommendation_calibration_snapshots.id", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column(
            "summary_id",
            sa.String(36),
            sa.ForeignKey("training_execution_summaries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "calibration_id", "summary_id", name="uq_calibration_source"
        ),
    )
    op.create_index(
        "ix_calibration_sources_summary",
        "recommendation_calibration_sources",
        ["summary_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_calibration_sources_summary",
        table_name="recommendation_calibration_sources",
    )
    op.drop_table("recommendation_calibration_sources")
    op.drop_index(
        "ix_calibration_snapshots_match",
        table_name="recommendation_calibration_snapshots",
    )
    op.drop_index(
        "ix_calibration_snapshots_scope",
        table_name="recommendation_calibration_snapshots",
    )
    op.drop_table("recommendation_calibration_snapshots")
    op.drop_index(
        "ix_execution_summaries_gpu_settings", table_name="training_execution_summaries"
    )
    op.drop_index(
        "ix_execution_summaries_project", table_name="training_execution_summaries"
    )
    op.drop_table("training_execution_summaries")
    with op.batch_alter_table("training_recommendations") as batch_op:
        for column in (
            "calibration_fingerprint",
            "calibrated_batch_size",
            "baseline_batch_size",
            "calibrated_vram_bytes",
            "baseline_vram_bytes",
            "calibrated_duration_seconds",
            "baseline_duration_seconds",
            "calibration_reason_codes_json",
            "calibration_confidence",
            "calibration_applied",
            "calibration_snapshot_id",
        ):
            batch_op.drop_column(column)
