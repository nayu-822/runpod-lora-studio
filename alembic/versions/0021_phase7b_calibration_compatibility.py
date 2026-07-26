"""persist strict calibration compatibility dimensions"""

import sqlalchemy as sa
from alembic import op

revision = "0021_phase7b_calibration_compatibility"
down_revision = "0020_phase7b_memory_measurements"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("recommendation_calibration_snapshots") as batch_op:
        for column in (
            sa.Column("gpu_architecture", sa.String(128)),
            sa.Column("batch_size", sa.Integer()),
            sa.Column("gradient_accumulation_steps", sa.Integer()),
            sa.Column("effective_batch_size", sa.Integer()),
            sa.Column("network_module", sa.String(128)),
            sa.Column("network_dim", sa.Integer()),
            sa.Column("network_alpha", sa.Integer()),
            sa.Column("world_size", sa.Integer()),
            sa.Column("sd_scripts_version", sa.String(128)),
            sa.Column("xformers_available", sa.Boolean()),
        ):
            batch_op.add_column(column)


def downgrade() -> None:
    with op.batch_alter_table("recommendation_calibration_snapshots") as batch_op:
        for name in (
            "xformers_available",
            "sd_scripts_version",
            "world_size",
            "network_alpha",
            "network_dim",
            "network_module",
            "effective_batch_size",
            "gradient_accumulation_steps",
            "batch_size",
            "gpu_architecture",
        ):
            batch_op.drop_column(name)
