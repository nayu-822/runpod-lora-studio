from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

from runpod_lora_studio.config.settings import get_settings
from runpod_lora_studio.persistence.database import enable_sqlite_foreign_keys
from runpod_lora_studio.persistence.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def database_url() -> str:
    settings = get_settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{Path(settings.database_path).as_posix()}"


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(), target_metadata=target_metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = database_url()
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    enable_sqlite_foreign_keys(connectable)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
