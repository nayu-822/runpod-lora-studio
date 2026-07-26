"""persist job-scoped training memory measurements and fingerprints"""

import sqlalchemy as sa
from alembic import op

revision = "0020_phase7b_memory_measurements"
down_revision = "0019_phase7b_training_performance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_execution_summaries") as batch_op:
        for column in (
            sa.Column("gpu_architecture", sa.String(128)),
            sa.Column("gpu_index", sa.Integer()),
            sa.Column("world_size", sa.Integer()),
            sa.Column("sd_scripts_version", sa.String(128)),
            sa.Column("xformers_available", sa.Boolean()),
            sa.Column("minimum_free_vram_bytes", sa.Integer()),
            sa.Column("whole_gpu_peak_used_vram_bytes", sa.Integer()),
            sa.Column("other_process_peak_vram_bytes", sa.Integer()),
            sa.Column(
                "memory_failed_sample_count",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.Column("memory_first_sampled_at", sa.DateTime(timezone=True)),
            sa.Column("memory_last_sampled_at", sa.DateTime(timezone=True)),
            sa.Column("memory_coverage_seconds", sa.Float()),
            sa.Column(
                "process_identity_verified",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            sa.Column(
                "gpu_identity_verified",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            sa.Column(
                "measurement_version",
                sa.String(32),
                nullable=False,
                server_default="phase7b-memory-v1",
            ),
            sa.Column(
                "classifier_version",
                sa.String(32),
                nullable=False,
                server_default="phase7b-failure-v1",
            ),
            sa.Column(
                "summary_content_fingerprint",
                sa.String(64),
                nullable=False,
                server_default="",
            ),
            sa.Column(
                "calibration_state_fingerprint",
                sa.String(64),
                nullable=False,
                server_default="",
            ),
        ):
            batch_op.add_column(column)

    op.create_table(
        "training_memory_aggregates",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column(
            "training_job_id",
            sa.String(36),
            sa.ForeignKey("training_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("gpu_index", sa.Integer()),
        sa.Column("gpu_uuid_fingerprint", sa.String(64)),
        sa.Column("gpu_total_vram_bytes", sa.Integer()),
        sa.Column("free_vram_before_bytes", sa.Integer()),
        sa.Column("minimum_free_vram_bytes", sa.Integer()),
        sa.Column("free_vram_after_bytes", sa.Integer()),
        sa.Column("target_process_peak_used_bytes", sa.Integer()),
        sa.Column("whole_gpu_peak_used_bytes", sa.Integer()),
        sa.Column("other_process_peak_used_bytes", sa.Integer()),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "failed_sample_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("first_sampled_at", sa.DateTime(timezone=True)),
        sa.Column("last_sampled_at", sa.DateTime(timezone=True)),
        sa.Column(
            "process_identity_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "gpu_identity_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("confidence", sa.String(16), nullable=False, server_default="none"),
        sa.Column("last_sample_fingerprint", sa.String(64)),
        sa.Column(
            "measurement_version",
            sa.String(32),
            nullable=False,
            server_default="phase7b-memory-v1",
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint("training_job_id", name="uq_training_memory_aggregate_job"),
    )
    op.create_index(
        "ix_training_memory_aggregates_job",
        "training_memory_aggregates",
        ["training_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_memory_aggregates_job", table_name="training_memory_aggregates"
    )
    op.drop_table("training_memory_aggregates")
    with op.batch_alter_table("training_execution_summaries") as batch_op:
        for name in (
            "calibration_state_fingerprint",
            "summary_content_fingerprint",
            "classifier_version",
            "measurement_version",
            "gpu_identity_verified",
            "process_identity_verified",
            "memory_coverage_seconds",
            "memory_last_sampled_at",
            "memory_first_sampled_at",
            "memory_failed_sample_count",
            "other_process_peak_vram_bytes",
            "whole_gpu_peak_used_vram_bytes",
            "minimum_free_vram_bytes",
            "xformers_available",
            "sd_scripts_version",
            "world_size",
            "gpu_index",
            "gpu_architecture",
        ):
            batch_op.drop_column(name)
