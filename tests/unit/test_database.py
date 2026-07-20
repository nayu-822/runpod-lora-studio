from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from runpod_lora_studio.config.settings import (
    AppSettings,
    ensure_runtime_directories,
    get_settings,
)
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import ImageAssetRecord, ProjectRecord


def migrate(test_workspace: Path, revision: str = "head") -> AppSettings:
    settings = AppSettings(
        workspace_root=test_workspace / "runtime",
        projects_dir=test_workspace / "runtime" / "projects",
        models_dir=test_workspace / "runtime" / "models",
        outputs_dir=test_workspace / "runtime" / "outputs",
        logs_dir=test_workspace / "runtime" / "logs",
        temp_dir=test_workspace / "runtime" / "tmp",
        database_path=test_workspace / "runtime" / "database" / "studio.sqlite3",
    )
    ensure_runtime_directories(settings)
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.upgrade(config, revision)
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    return settings


def test_empty_database_and_existing_0001_upgrade_to_head(test_workspace: Path) -> None:
    settings = migrate(test_workspace)
    engine = create_engine_for_settings(settings)
    indexes = {item["name"] for item in inspect(engine).get_indexes("projects")}
    image_indexes = {
        item["name"] for item in inspect(engine).get_indexes("image_assets")
    }
    assert "ix_projects_updated_at" in indexes
    assert "ix_image_assets_project_state" in image_indexes
    assert "ix_image_assets_project_id" not in image_indexes

    config = Config(str(Path("alembic.ini").resolve()))
    command.upgrade(config, "head")


def test_existing_0001_database_upgrades_to_head(test_workspace: Path) -> None:
    settings = migrate(test_workspace, "0001_initial")
    migrate(test_workspace, "head")
    engine = create_engine_for_settings(settings)
    indexes = {item["name"] for item in inspect(engine).get_indexes("projects")}
    assert "ix_projects_updated_at" in indexes


def test_foreign_keys_are_enabled_and_reject_orphans(test_workspace: Path) -> None:
    settings = migrate(test_workspace)
    engine = create_engine_for_settings(settings)
    with engine.connect() as connection:
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1

    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.add(
            ImageAssetRecord(
                id=str(uuid4()),
                project_id=str(uuid4()),
                original_filename="image.png",
                stored_filename="image.png",
                original_path="originals/image.png",
                thumbnail_path="thumbnails/image.png",
                sha256="0" * 64,
                width=1,
                height=1,
                file_size=1,
                mime_type="image/png",
                selection_state="pending",
                exclusion_reasons="[]",
                source_type="upload",
                selection_source="manual",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_project_and_image_registration_succeeds(test_workspace: Path) -> None:
    settings = migrate(test_workspace)
    engine = create_engine_for_settings(settings)
    from sqlalchemy.orm import Session

    project_id = str(uuid4())
    with Session(engine) as session:
        session.add(
            ProjectRecord(
                id=project_id,
                name="project",
                description="",
                concept_type="other",
                trigger_words="[]",
                status="draft",
                schema_version=1,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        session.flush()
        session.add(
            ImageAssetRecord(
                id=str(uuid4()),
                project_id=project_id,
                original_filename="image.png",
                stored_filename="image.png",
                original_path="originals/image.png",
                thumbnail_path="thumbnails/image.png",
                sha256="1" * 64,
                width=1,
                height=1,
                file_size=1,
                mime_type="image/png",
                selection_state="pending",
                exclusion_reasons="[]",
                source_type="upload",
                selection_source="manual",
                created_at=__import__("datetime").datetime.now(
                    __import__("datetime").UTC
                ),
                updated_at=__import__("datetime").datetime.now(
                    __import__("datetime").UTC
                ),
            )
        )
        session.commit()
