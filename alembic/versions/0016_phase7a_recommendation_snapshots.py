"""persist Phase 7A environment snapshots and recommendations"""

import sqlalchemy as sa
from alembic import op

revision = "0016_phase7a_recommendation_snapshots"
down_revision = "0015_phase6c_state_position_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "compute_environment_snapshots",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("warning_json", sa.Text(), nullable=False),
        sa.Column("error_json", sa.Text(), nullable=False),
        sa.Column("detector_version", sa.String(32), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_compute_environment_snapshots_project",
        "compute_environment_snapshots",
        ["project_id"],
    )
    op.create_table(
        "training_environment_snapshots",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
        ),
        sa.Column(
            "compute_snapshot_id",
            sa.String(36),
            sa.ForeignKey("compute_environment_snapshots.id", ondelete="SET NULL"),
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("warning_json", sa.Text(), nullable=False),
        sa.Column("error_json", sa.Text(), nullable=False),
        sa.Column("detector_version", sa.String(32), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_training_environment_snapshots_project",
        "training_environment_snapshots",
        ["project_id"],
    )
    op.create_table(
        "training_recommendation_requests",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
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
        sa.Column(
            "environment_snapshot_id",
            sa.String(36),
            sa.ForeignKey("compute_environment_snapshots.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("concept_type", sa.String(32), nullable=False),
        sa.Column("quality_profile", sa.String(32), nullable=False),
        sa.Column("speed_profile", sa.String(32), nullable=False),
        sa.Column("user_constraints_json", sa.Text(), nullable=False),
        sa.Column("input_fingerprint", sa.String(64), nullable=False),
        sa.Column("engine_version", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("warning_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_training_recommendation_requests_project",
        "training_recommendation_requests",
        ["project_id"],
    )
    op.create_index(
        "ix_training_recommendation_requests_snapshot",
        "training_recommendation_requests",
        ["dataset_snapshot_id"],
    )
    op.create_table(
        "training_recommendations",
        sa.Column("internal_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column(
            "request_id",
            sa.String(36),
            sa.ForeignKey("training_recommendation_requests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("profile_name", sa.String(64), nullable=False),
        sa.Column("settings_json", sa.Text(), nullable=False),
        sa.Column("reasons_json", sa.Text(), nullable=False),
        sa.Column("warnings_json", sa.Text(), nullable=False),
        sa.Column("settings_fingerprint", sa.String(64), nullable=False),
        sa.Column("engine_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "request_id", "rank", name="uq_training_recommendation_rank"
        ),
    )
    op.create_index(
        "ix_training_recommendations_request",
        "training_recommendations",
        ["request_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_recommendations_request", table_name="training_recommendations"
    )
    op.drop_table("training_recommendations")
    op.drop_index(
        "ix_training_recommendation_requests_snapshot",
        table_name="training_recommendation_requests",
    )
    op.drop_index(
        "ix_training_recommendation_requests_project",
        table_name="training_recommendation_requests",
    )
    op.drop_table("training_recommendation_requests")
    op.drop_index(
        "ix_training_environment_snapshots_project",
        table_name="training_environment_snapshots",
    )
    op.drop_table("training_environment_snapshots")
    op.drop_index(
        "ix_compute_environment_snapshots_project",
        table_name="compute_environment_snapshots",
    )
    op.drop_table("compute_environment_snapshots")
