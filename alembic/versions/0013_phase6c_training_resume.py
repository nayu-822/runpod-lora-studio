"""add safe training state resume metadata"""

import sqlalchemy as sa
from alembic import op

revision = "0013_phase6c_training_resume"
down_revision = "0012_phase6b_progress_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.add_column(sa.Column("parent_job_id", sa.String(36), nullable=True))
        batch_op.add_column(
            sa.Column("resume_artifact_id", sa.String(36), nullable=True)
        )
        batch_op.add_column(sa.Column("resume_mode", sa.String(32), nullable=True))
        batch_op.add_column(
            sa.Column("resume_requested_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("resume_validation_status", sa.String(32), nullable=True)
        )
        batch_op.add_column(
            sa.Column("resume_validation_code", sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("resume_validation_message", sa.Text(), nullable=True)
        )
        batch_op.add_column(sa.Column("initial_epoch", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("initial_step", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("progress_step_offset", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("progress_epoch_offset", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_training_jobs_parent_job",
            "training_jobs",
            ["parent_job_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_training_jobs_resume_artifact",
            "training_artifacts",
            ["resume_artifact_id"],
            ["id"],
            ondelete="RESTRICT",
        )
    op.create_index(
        "ix_training_jobs_parent_job_id", "training_jobs", ["parent_job_id"]
    )
    op.create_table(
        "training_resume_validations",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("source_job_id", sa.String(36), nullable=False),
        sa.Column("source_artifact_id", sa.String(36), nullable=False),
        sa.Column("target_training_config_id", sa.String(36), nullable=False),
        sa.Column("source_state_relative_path", sa.Text(), nullable=False),
        sa.Column("source_state_fingerprint", sa.String(64), nullable=True),
        sa.Column("source_job_config_fingerprint", sa.String(64), nullable=True),
        sa.Column("target_config_fingerprint", sa.String(64), nullable=True),
        sa.Column("compatibility_status", sa.String(32), nullable=False),
        sa.Column("compatibility_issues", sa.Text(), nullable=False),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validator_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_job_id"], ["training_jobs.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["source_artifact_id"], ["training_artifacts.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["target_training_config_id"], ["training_configs.id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint("id"),
    )
    op.create_index(
        "ix_training_resume_validations_id",
        "training_resume_validations",
        ["id"],
        unique=True,
    )
    op.create_index(
        "ix_training_resume_validations_source_job",
        "training_resume_validations",
        ["source_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_resume_validations_source_job",
        table_name="training_resume_validations",
    )
    op.drop_index(
        "ix_training_resume_validations_id", table_name="training_resume_validations"
    )
    op.drop_table("training_resume_validations")
    op.drop_index("ix_training_jobs_parent_job_id", table_name="training_jobs")
    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.drop_constraint("fk_training_jobs_resume_artifact", type_="foreignkey")
        batch_op.drop_constraint("fk_training_jobs_parent_job", type_="foreignkey")
        for name in (
            "progress_epoch_offset",
            "progress_step_offset",
            "initial_step",
            "initial_epoch",
            "resume_validation_message",
            "resume_validation_code",
            "resume_validation_status",
            "resume_requested_at",
            "resume_mode",
            "resume_artifact_id",
            "parent_job_id",
        ):
            batch_op.drop_column(name)
