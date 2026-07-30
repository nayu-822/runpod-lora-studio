from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from PIL import Image
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from runpod_lora_studio.config.settings import (
    AppSettings,
    ensure_runtime_directories,
    get_settings,
)
from runpod_lora_studio.domain.models import SelectionState
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import ImageAssetRecord, ProjectRecord
from runpod_lora_studio.services.caption_service import CaptionEditingService
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.project_service import ProjectInput, ProjectService


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

    migrate(test_workspace, "head")
    with engine.connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0028_phase8a_claim_reservations"
        )


def test_phase6b_tables_have_job_scoped_constraints(test_workspace: Path) -> None:
    settings = migrate(test_workspace)
    engine = create_engine_for_settings(settings)
    inspector = inspect(engine)
    assert {
        "training_progress",
        "training_metric_points",
        "training_artifacts",
    }.issubset(inspector.get_table_names())
    progress_columns = {
        column["name"] for column in inspector.get_columns("training_progress")
    }
    assert {
        "current_step",
        "total_steps",
        "progress_ratio",
        "stdout_offset",
        "stderr_offset",
        "parse_warning",
    }.issubset(progress_columns)
    metric_unique = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("training_metric_points")
    }
    artifact_unique = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("training_artifacts")
    }
    assert "uq_training_metric_step" in metric_unique
    assert "uq_training_artifact_path" in artifact_unique


def test_phase7b_memory_and_compatibility_migrations_are_present(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    inspector = inspect(create_engine_for_settings(settings))
    assert "training_memory_aggregates" in inspector.get_table_names()
    memory_columns = {
        column["name"] for column in inspector.get_columns("training_memory_aggregates")
    }
    assert {
        "target_process_peak_used_bytes",
        "whole_gpu_peak_used_bytes",
        "other_process_peak_used_bytes",
        "failed_sample_count",
        "measurement_version",
    }.issubset(memory_columns)
    summary_columns = {
        column["name"]
        for column in inspector.get_columns("training_execution_summaries")
    }
    assert {
        "summary_content_fingerprint",
        "calibration_state_fingerprint",
        "process_identity_verified",
        "gpu_identity_verified",
        "training_job_environment_snapshot_id",
        "memory_warning_codes_json",
        "memory_failure_codes_json",
    }.issubset(summary_columns)
    assert "training_job_environment_snapshots" in inspector.get_table_names()
    job_environment_columns = {
        column["name"]
        for column in inspector.get_columns("training_job_environment_snapshots")
    }
    assert {
        "logical_gpu_index",
        "physical_gpu_index",
        "gpu_uuid_fingerprint",
        "total_vram_bytes",
        "cuda_visible_devices",
        "visible_gpu_uuids_json",
        "detector_version",
    }.issubset(job_environment_columns)
    assert "training_job_selected_gpus" in inspector.get_table_names()
    selected_gpu_columns = {
        column["name"] for column in inspector.get_columns("training_job_selected_gpus")
    }
    assert {
        "gpu_uuid_fingerprint",
        "physical_gpu_index",
        "gpu_architecture",
        "compute_capability",
        "total_vram_bytes",
        "selection_source",
        "last_observed_gpu_uuid_fingerprint",
        "gpu_change_detected_at",
        "gpu_change_count",
    }.issubset(selected_gpu_columns)
    summary_columns = {
        column["name"]
        for column in inspector.get_columns("training_execution_summaries")
    }
    assert {
        "selected_gpu_status",
        "selected_gpu_warning_codes_json",
    }.issubset(summary_columns)
    summary_columns = {
        column["name"]
        for column in inspector.get_columns("training_execution_summaries")
    }
    assert {"physical_gpu_index", "compute_capability"}.issubset(summary_columns)
    assert {"warning_codes_json", "failure_codes_json"}.issubset(memory_columns)
    calibration_columns = {
        column["name"]
        for column in inspector.get_columns("recommendation_calibration_snapshots")
    }
    assert {
        "batch_size",
        "network_module",
        "network_dim",
        "sd_scripts_version",
    }.issubset(calibration_columns)


def test_phase8a_acquisition_tables_have_source_and_plan_constraints(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    inspector = inspect(create_engine_for_settings(settings))
    tables = set(inspector.get_table_names())
    assert {
        "external_image_posts",
        "image_source_searches",
        "image_source_search_results",
        "image_acquisition_plans",
        "image_acquisition_plan_items",
        "external_image_asset_links",
        "image_acquisition_reservations",
    }.issubset(tables)
    post_columns = {
        item["name"] for item in inspector.get_columns("external_image_posts")
    }
    assert {
        "source_type",
        "external_post_id",
        "source_md5",
        "metadata_fingerprint",
    }.issubset(post_columns)
    post_unique = {
        item["name"]
        for item in inspector.get_unique_constraints("external_image_posts")
    }
    result_unique = {
        item["name"]
        for item in inspector.get_unique_constraints("image_source_search_results")
    }
    plan_unique = {
        item["name"]
        for item in inspector.get_unique_constraints("image_acquisition_plans")
    }
    assert "uq_external_image_source_post" in post_unique
    assert "uq_image_search_result_post" in result_unique
    assert "plan_fingerprint" in {
        column["name"] for column in inspector.get_columns("image_acquisition_plans")
    }
    assert "uq_image_plan_item_post" in {
        item["name"]
        for item in inspector.get_unique_constraints("image_acquisition_plan_items")
    }
    search_columns = {
        column["name"] for column in inspector.get_columns("image_source_searches")
    }
    assert {"worker_generation", "claim_token"}.issubset(search_columns)
    reservation_unique = {
        item["name"]
        for item in inspector.get_unique_constraints("image_acquisition_reservations")
    }
    assert "uq_image_acquisition_reservation_source_post" in reservation_unique
    assert plan_unique


def test_existing_0001_database_upgrades_to_head(test_workspace: Path) -> None:
    settings = migrate(test_workspace, "0001_initial")
    migrate(test_workspace, "head")
    engine = create_engine_for_settings(settings)
    indexes = {item["name"] for item in inspect(engine).get_indexes("projects")}
    assert "ix_projects_updated_at" in indexes


def test_phase2a_migration_downgrade_removes_only_inspection_table(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0002_phase1_indexes")
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    engine = create_engine_for_settings(settings)
    assert "image_inspection_results" not in inspect(engine).get_table_names()
    assert "image_assets" in inspect(engine).get_table_names()


def test_phase3_downgrade_and_reupgrade_preserves_phase2_tables(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0004_phase2b_perceptual_similarity")
        tables_after_downgrade = set(
            inspect(create_engine_for_settings(settings)).get_table_names()
        )
        assert "tagger_runs" not in tables_after_downgrade
        assert "similarity_groups" in tables_after_downgrade
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    with create_engine_for_settings(settings).connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0028_phase8a_claim_reservations"
        )


def test_phase4_downgrade_and_reupgrade_preserves_phase3_tables(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0005_phase3_tagging_caption")
        tables_after_downgrade = set(
            inspect(create_engine_for_settings(settings)).get_table_names()
        )
        assert "dataset_snapshots" not in tables_after_downgrade
        assert "tagger_runs" in tables_after_downgrade
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    with create_engine_for_settings(settings).connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0028_phase8a_claim_reservations"
        )


def test_phase5_upgrades_existing_0006_database_to_head(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace, "0006_phase4_dataset_snapshots")
    migrate(test_workspace, "head")
    with create_engine_for_settings(settings).connect() as connection:
        tables = set(inspect(connection).get_table_names())
        assert "managed_models" in tables
        assert "storage_transfer_jobs" in tables
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0028_phase8a_claim_reservations"
        )


def test_phase5_heartbeat_migration_upgrades_existing_0007_database(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace, "0007_phase5_storage")
    migrate(test_workspace, "head")
    with create_engine_for_settings(settings).connect() as connection:
        columns = {
            column["name"]
            for column in inspect(connection).get_columns("storage_transfer_jobs")
        }
        assert {
            "worker_id",
            "heartbeat_at",
            "completed_transferred_bytes",
            "current_file_transferred_bytes",
        }.issubset(columns)
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0028_phase8a_claim_reservations"
        )


def test_phase5_progress_migration_upgrades_existing_0008_database(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace, "0008_storage_transfer_heartbeat")
    job_id = str(uuid4())
    with create_engine_for_settings(settings).begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO storage_transfer_jobs
                (id, project_id, snapshot_id, training_run_id, transfer_type,
                 source_kind, destination_kind, status, current_step, item_count,
                 processed_item_count, succeeded_item_count, failed_item_count,
                 skipped_item_count, total_bytes, transferred_bytes, cancel_requested,
                 pid, worker_id, heartbeat_at, started_at, completed_at,
                 error_summary, manifest_path, created_at, updated_at)
                VALUES
                (:id, NULL, NULL, NULL, 'model_download', 'remote', 'local',
                 'running', 'transferring', 1, 0, 0, 0, 0, 10, 0, 0,
                 NULL, 'legacy-worker', :now, :now, NULL, NULL, NULL, :now, :now)
                """
            ),
            {"id": job_id, "now": datetime.now(UTC)},
        )
    migrate(test_workspace, "head")
    with create_engine_for_settings(settings).connect() as connection:
        columns = {
            column["name"]
            for column in inspect(connection).get_columns("storage_transfer_jobs")
        }
        assert {
            "completed_transferred_bytes",
            "current_file_transferred_bytes",
        }.issubset(columns)
        row = connection.execute(
            text(
                "SELECT status, completed_transferred_bytes, "
                "current_file_transferred_bytes FROM storage_transfer_jobs "
                "WHERE id = :id"
            ),
            {"id": job_id},
        ).one()
        assert row == ("running", 0, 0)
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0028_phase8a_claim_reservations"
        )


def test_phase5_progress_downgrade_and_reupgrade(test_workspace: Path) -> None:
    settings = migrate(test_workspace, "head")
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0008_storage_transfer_heartbeat")
        with create_engine_for_settings(settings).connect() as connection:
            columns = {
                column["name"]
                for column in inspect(connection).get_columns("storage_transfer_jobs")
            }
            assert "completed_transferred_bytes" not in columns
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    with create_engine_for_settings(settings).connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0028_phase8a_claim_reservations"
        )


def test_phase6a_downgrade_removes_training_tables_only(test_workspace: Path) -> None:
    settings = migrate(test_workspace, "head")
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0009_storage_transfer_progress")
        tables = set(inspect(create_engine_for_settings(settings)).get_table_names())
        assert "training_configs" not in tables
        assert "training_jobs" not in tables
        assert "managed_models" in tables
        command.upgrade(config, "head")
        columns = {
            column["name"]
            for column in inspect(create_engine_for_settings(settings)).get_columns(
                "training_configs"
            )
        }
        assert "python_executable" not in columns
        assert "repeats" not in columns
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path


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


def test_phase1_services_work_on_alembic_database(test_workspace: Path) -> None:
    settings = migrate(test_workspace)
    source = test_workspace / "service.png"
    Image.new("RGB", (16, 16), "purple").save(source)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("Service integration"))
    images = ImageService(settings, projects)

    result = images.register_uploads(project.id, [source])
    assert len(result.successes) == 1
    assert (
        images.change_state(
            project.id, [result.successes[0].id], SelectionState.ACCEPTED
        )
        == 1
    )

    restored = ImageService(settings, ProjectService(settings))
    listed, total = restored.list_images(project.id, state=SelectionState.ACCEPTED)
    assert total == 1
    assert listed[0].selection_state is SelectionState.ACCEPTED
    assert "ix_image_assets_project_state" in {
        item["name"]
        for item in inspect(create_engine_for_settings(settings)).get_indexes(
            "image_assets"
        )
    }


def test_0006_database_supports_snapshot_creation_and_revalidation(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    projects = ProjectService(settings)
    project = projects.create(ProjectInput("phase4-db"))
    source = test_workspace / "phase4.png"
    Image.new("RGB", (96, 96), "blue").save(source)
    uploads = ImageService(settings, projects).register_uploads(project.id, [source])
    assert len(uploads.successes) == 1
    image = uploads.successes[0]
    ImageService(settings, projects).change_state(
        project.id, [image.id], SelectionState.ACCEPTED
    )
    CaptionEditingService(settings, projects).save_image_caption(
        project.id, image.id, "phase4"
    )
    service = DatasetSnapshotService(settings, projects)
    snapshot = service.create_snapshot_sync(service.preview(project.id), name="db-head")
    assert service.revalidate(snapshot.id).value == "completed"
