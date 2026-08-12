from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from PIL import Image
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from runpod_lora_studio.config.settings import (
    AppSettings,
    ensure_runtime_directories,
    get_settings,
)
from runpod_lora_studio.domain.acquisition_download_models import (
    ImageAcquisitionJobStatus,
)
from runpod_lora_studio.domain.models import SelectionState
from runpod_lora_studio.persistence.database import create_engine_for_settings
from runpod_lora_studio.persistence.models import (
    ImageAcquisitionJobRecord,
    ImageAssetRecord,
    ProjectRecord,
)
from runpod_lora_studio.services.acquisition_download_service import (
    ImageAcquisitionDownloadService,
)
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
            "0040_phase9a_training_completion_export"
        )


def test_phase9a_completion_export_schema_survives_downgrade_and_reupgrade(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    engine = create_engine_for_settings(settings)
    inspector = inspect(engine)
    assert "training_completion_exports" in inspector.get_table_names()
    columns = {
        column["name"]
        for column in inspector.get_columns("training_completion_exports")
    }
    assert {
        "training_job_id",
        "worker_generation",
        "heartbeat_at",
        "cancel_requested",
        "source_fingerprint",
        "preview_fingerprint",
        "export_manifest_sha256",
        "completion_manifest_sha256",
    }.issubset(columns)
    unique_constraints = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints(
            "training_completion_exports"
        )
    }
    assert "uq_training_completion_exports_training_job" in unique_constraints

    project = ProjectService(settings).create(ProjectInput("phase9a-preserved"))
    with engine.connect() as connection:
        before = connection.scalar(
            text("SELECT COUNT(*) FROM projects WHERE id = :project_id"),
            {"project_id": str(project.id)},
        )
    assert before == 1

    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0039_phase8c_manifest_orphan_scan")
        assert "training_completion_exports" not in inspect(engine).get_table_names()
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    with engine.connect() as connection:
        after = connection.scalar(
            text("SELECT COUNT(*) FROM projects WHERE id = :project_id"),
            {"project_id": str(project.id)},
        )
    assert after == 1


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
        "image_source_search_cursor_checkpoints",
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
    assert {
        "worker_generation",
        "claim_token",
        "request_cursor",
        "completion_reason",
    }.issubset(search_columns)
    checkpoint_columns = {
        column["name"]
        for column in inspector.get_columns("image_source_search_cursor_checkpoints")
    }
    assert {
        "request_cursor_fingerprint",
        "next_cursor_fingerprint",
        "worker_generation",
        "committed_at",
    }.issubset(checkpoint_columns)
    reservation_unique = {
        item["name"]
        for item in inspector.get_unique_constraints("image_acquisition_reservations")
    }
    assert "uq_image_acquisition_reservation_source_post" in reservation_unique
    assert plan_unique


def test_phase8b_acquisition_download_schema_is_migrated(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    inspector = inspect(create_engine_for_settings(settings))
    tables = set(inspector.get_table_names())
    assert {
        "image_acquisition_jobs",
        "image_acquisition_job_items",
        "image_acquisition_attempts",
    }.issubset(tables)
    job_columns = {
        column["name"] for column in inspector.get_columns("image_acquisition_jobs")
    }
    assert {
        "worker_generation",
        "claim_token",
        "heartbeat_at",
        "current_item_id",
        "manifest_relative_path",
        "downloader_version",
    }.issubset(job_columns)
    item_columns = {
        column["name"]
        for column in inspector.get_columns("image_acquisition_job_items")
    }
    assert {
        "part_relative_path",
        "expected_file_url",
        "range_start",
        "last_attempted_at",
        "calculated_sha256",
        "detected_format",
        "failure_code",
        "part_cleanup_warning",
        "part_cleanup_claim_token",
        "part_cleanup_claimed_at",
        "part_cleanup_next_retry_at",
    }.issubset(item_columns)
    link_columns = {
        column["name"] for column in inspector.get_columns("external_image_asset_links")
    }
    assert {
        "project_id",
        "source_md5",
        "source_metadata_fingerprint",
        "acquisition_job_id",
        "acquisition_job_item_id",
        "linked_at",
    }.issubset(link_columns)
    assert "expected_file_size" in {
        column["name"]
        for column in inspector.get_columns("image_acquisition_plan_items")
    }
    assert "uq_image_asset_source" not in {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("external_image_asset_links")
    }
    attempt_columns = {
        column["name"] for column in inspector.get_columns("image_acquisition_attempts")
    }
    assert {
        "http_status",
        "requested_range_start",
        "retry_after_seconds",
        "response_etag_fingerprint",
        "response_last_modified_fingerprint",
        "worker_generation",
    }.issubset(attempt_columns)


def test_phase8b_part_cleanup_claims_downgrade_and_reupgrade(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0032_phase8b_part_cleanup_warnings")
        inspector = inspect(create_engine_for_settings(settings))
        item_columns = {
            column["name"]
            for column in inspector.get_columns("image_acquisition_job_items")
        }
        assert "part_cleanup_claim_token" not in item_columns
        assert "part_cleanup_claimed_at" not in item_columns
        assert "ix_image_acquisition_job_items_part_cleanup" not in {
            index["name"]
            for index in inspector.get_indexes("image_acquisition_job_items")
        }
        command.upgrade(config, "head")
        upgraded_inspector = inspect(create_engine_for_settings(settings))
        upgraded_item_columns = {
            column["name"]
            for column in upgraded_inspector.get_columns("image_acquisition_job_items")
        }
        upgraded_job_columns = {
            column["name"]
            for column in upgraded_inspector.get_columns("image_acquisition_jobs")
        }
        assert "part_cleanup_attempt_count" in upgraded_item_columns
        assert {
            "manifest_repair_state",
            "manifest_repair_attempted_at",
            "manifest_target_status",
            "manifest_target_error_code",
        }.issubset(upgraded_job_columns)
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    with create_engine_for_settings(settings).connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0040_phase9a_training_completion_export"
        )


def test_phase8c_manifest_orphan_scan_migration_downgrade_and_reupgrade(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace, "0038_phase8c_legacy_manifest_recovery")
    engine = create_engine_for_settings(settings)
    before_inspector = inspect(engine)
    before_columns = {
        column["name"]
        for column in before_inspector.get_columns("image_acquisition_jobs")
    }
    before_indexes = {
        index["name"]
        for index in before_inspector.get_indexes("image_acquisition_jobs")
    }
    assert "manifest_orphan_checked_at" not in before_columns
    assert "ix_image_acquisition_jobs_manifest_orphan_scan" not in before_indexes

    migrate(test_workspace, "head")
    upgraded_inspector = inspect(engine)
    assert "manifest_orphan_checked_at" in {
        column["name"]
        for column in upgraded_inspector.get_columns("image_acquisition_jobs")
    }
    assert "ix_image_acquisition_jobs_manifest_orphan_scan" in {
        index["name"]
        for index in upgraded_inspector.get_indexes("image_acquisition_jobs")
    }

    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0038_phase8c_legacy_manifest_recovery")
        downgraded_inspector = inspect(engine)
        assert "manifest_orphan_checked_at" not in {
            column["name"]
            for column in downgraded_inspector.get_columns("image_acquisition_jobs")
        }
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path

    with engine.connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0040_phase9a_training_completion_export"
        )


def test_phase8c_manifest_repair_intent_backfills_existing_0035_rows(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace, "0035_phase8c_cleanup_repair_scheduler")
    engine = create_engine_for_settings(settings)
    connection = engine.connect()
    connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
    connection.commit()
    transaction = connection.begin()
    try:
        connection.execute(
            text(
                """
                INSERT INTO image_acquisition_jobs (
                    id, project_id, plan_id, plan_fingerprint, source_type, status,
                    worker_generation, cancellation_requested, manifest_warning,
                    manifest_repair_state, error_code, downloader_version,
                    validator_version, importer_version, job_fingerprint,
                    created_at, updated_at
                ) VALUES (
                    :id, :project_id, :plan_id, :plan_fingerprint, :source_type,
                    :status, 0, :cancellation_requested, :manifest_warning,
                    :manifest_repair_state, :error_code, :downloader_version,
                    :validator_version, :importer_version, :job_fingerprint,
                    :created_at, :updated_at
                )
                """
            ),
            {
                "id": "legacy-repair-job",
                "project_id": "missing-project",
                "plan_id": "legacy-plan",
                "plan_fingerprint": "legacy-fingerprint",
                "source_type": "danbooru",
                "status": "stale",
                "cancellation_requested": 1,
                "manifest_warning": "MANIFEST_REPAIR_PENDING:running",
                "manifest_repair_state": "pending",
                "error_code": "UNKNOWN_DOWNLOAD_ERROR",
                "downloader_version": "legacy",
                "validator_version": "legacy",
                "importer_version": "legacy",
                "job_fingerprint": "legacy-job-fingerprint",
                "created_at": "2026-08-05T00:00:00+00:00",
                "updated_at": "2026-08-05T00:00:00+00:00",
            },
        )
        transaction.commit()
    finally:
        connection.close()

    migrate(test_workspace)
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT manifest_target_status, manifest_target_error_code,
                       manifest_warning
                FROM image_acquisition_jobs
                WHERE id = 'legacy-repair-job'
                """
            )
        ).one()
    assert row == ("canceled", "CANCELED", "MANIFEST_WRITE_FAILED")


def test_phase8c_manifest_repair_backfill_reclassifies_legacy_item_states(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace, "0035_phase8c_cleanup_repair_scheduler")
    engine = create_engine_for_settings(settings)
    project = ProjectService(settings).create(ProjectInput(name="legacy-repair"))
    legacy_job_ids = {
        label: str(uuid4())
        for label in (
            "completed",
            "mixed",
            "failed",
            "incomplete",
            "canceled",
        )
    }
    legacy_jobs = {
        legacy_job_ids["completed"]: (False, None, ["imported", "linked_existing"]),
        legacy_job_ids["mixed"]: (
            False,
            "SOURCE_POST_NOT_FOUND",
            ["imported", "failed"],
        ),
        legacy_job_ids["failed"]: (False, None, ["failed", "canceled"]),
        legacy_job_ids["incomplete"]: (False, None, ["importing"]),
        legacy_job_ids["canceled"]: (True, "UNKNOWN_DOWNLOAD_ERROR", ["importing"]),
    }
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.commit()
        transaction = connection.begin()
        try:
            for job_id, (canceled, error_code, item_statuses) in legacy_jobs.items():
                connection.execute(
                    text(
                        """
                        INSERT INTO image_acquisition_jobs (
                            id, project_id, plan_id, plan_fingerprint, source_type,
                            status, worker_generation, cancellation_requested,
                            manifest_warning, manifest_repair_state, error_code,
                            downloader_version, validator_version, importer_version,
                            job_fingerprint, created_at, updated_at
                        ) VALUES (
                            :id, :project_id, :plan_id, :plan_fingerprint,
                            :source_type, 'stale', 0, :canceled,
                            'MANIFEST_WRITE_FAILED', 'pending', :error_code,
                            'legacy', 'legacy', 'legacy', 'legacy-job-fingerprint',
                            '2026-08-05T00:00:00+00:00',
                            '2026-08-05T00:00:00+00:00'
                        )
                        """
                    ),
                    {
                        "id": job_id,
                        "project_id": str(project.id),
                        "plan_id": str(uuid4()),
                        "plan_fingerprint": f"fingerprint-{job_id}",
                        "source_type": "danbooru",
                        "canceled": int(canceled),
                        "error_code": error_code,
                    },
                )
                for index, item_status in enumerate(item_statuses):
                    connection.execute(
                        text(
                            """
                            INSERT INTO image_acquisition_job_items (
                                id, job_id, plan_item_id, source_type,
                                external_post_id, display_order, status,
                                expected_metadata_fingerprint, part_relative_path,
                                created_at, updated_at
                            ) VALUES (
                                :id, :job_id, :plan_item_id, 'danbooru',
                                :external_post_id, :display_order, :status,
                                'legacy-item-fingerprint', :part_relative_path,
                                '2026-08-05T00:00:00+00:00',
                                '2026-08-05T00:00:00+00:00'
                            )
                            """
                        ),
                        {
                            "id": f"{job_id}-item-{index}",
                            "job_id": job_id,
                            "plan_item_id": f"{job_id}-plan-item-{index}",
                            "external_post_id": str(index),
                            "display_order": index,
                            "status": item_status,
                            "part_relative_path": (
                                f"acquisition/jobs/{job_id}/parts/item-{index}.part"
                            ),
                        },
                    )
            transaction.commit()
        finally:
            if transaction.is_active:
                transaction.rollback()

    migrate(test_workspace, "head")
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT id, manifest_target_status, manifest_target_error_code
                FROM image_acquisition_jobs
                ORDER BY id
                """
            )
        ).all()
    assert {row[0]: (row[1], row[2]) for row in rows} == {
        legacy_job_ids["canceled"]: ("canceled", "CANCELED"),
        legacy_job_ids["completed"]: ("completed", None),
        legacy_job_ids["failed"]: ("failed", None),
        legacy_job_ids["incomplete"]: ("failed", "INCOMPLETE_ITEM_STATE"),
        legacy_job_ids["mixed"]: ("partially_completed", "SOURCE_POST_NOT_FOUND"),
    }

    migrate(test_workspace, "0036_phase8c_manifest_repair_intent")
    migrate(test_workspace, "head")
    with engine.connect() as connection:
        completed_target = connection.execute(
            text(
                """
                SELECT manifest_target_status, manifest_target_error_code
                FROM image_acquisition_jobs
                WHERE id = :job_id
                """
            ),
            {"job_id": legacy_job_ids["completed"]},
        ).one()
    assert completed_target == ("completed", None)

    service = ImageAcquisitionDownloadService(settings, auto_start=False)
    repair_worker = "legacy-repair-worker"
    repair_token = "legacy-repair-token"
    repair_generation = service._claim_manifest_repair_job(
        legacy_job_ids["completed"], repair_worker, repair_token
    )
    assert repair_generation is not None
    with service.session_factory() as session:
        job = session.scalar(
            select(ImageAcquisitionJobRecord).where(
                ImageAcquisitionJobRecord.id == legacy_job_ids["completed"]
            )
        )
        assert job is not None
        assert job.status == ImageAcquisitionJobStatus.RUNNING.value
        session.commit()
    counts = service._recompute_counts(
        UUID(legacy_job_ids["completed"]),
        repair_worker,
        repair_token,
        repair_generation,
    )
    assert (
        service._write_manifest(
            UUID(legacy_job_ids["completed"]),
            repair_worker,
            repair_token,
            repair_generation,
            ImageAcquisitionJobStatus.COMPLETED,
            counts=counts,
        ).value
        == "success"
    )
    assert service._finish_job(
        UUID(legacy_job_ids["completed"]),
        repair_worker,
        repair_token,
        repair_generation,
        ImageAcquisitionJobStatus.COMPLETED,
        None,
    )
    repaired = service.get_job(UUID(legacy_job_ids["completed"]))
    assert repaired is not None
    assert repaired.status is ImageAcquisitionJobStatus.COMPLETED


def test_phase8c_legacy_terminal_manifest_recovery_repairs_only_proven_jobs(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace, "0035_phase8c_cleanup_repair_scheduler")
    engine = create_engine_for_settings(settings)
    project = ProjectService(settings).create(ProjectInput(name="legacy-terminal"))
    job_ids = {label: str(uuid4()) for label in ("completed", "mixed", "all-failed")}
    item_statuses = {
        "completed": ["imported", "linked_existing", "skipped"],
        "mixed": ["imported", "failed", "canceled"],
        "all-failed": ["failed", "canceled"],
    }
    manifest_names = {
        "completed": "manifest-g1-aaaaaaaaaaaa.json",
        "mixed": "manifest-g1-bbbbbbbbbbbb.json",
        "all-failed": "manifest-g1-cccccccccccc.json",
    }
    relative_paths = {
        label: f"acquisition/jobs/{job_id}/manifests/{manifest_names[label]}"
        for label, job_id in job_ids.items()
    }

    # The 0036 app had already written a FAILED manifest and then committed the
    # terminal job row, clearing its repair intent. Migration 0038 must use DB
    # state only and must not inspect or mutate these files.
    for _label, relative_path in relative_paths.items():
        manifest_path = settings.projects_dir / str(project.id) / relative_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"status":"failed"}', encoding="utf-8")

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.commit()
        transaction = connection.begin()
        try:
            for label, job_id in job_ids.items():
                connection.execute(
                    text(
                        """
                        INSERT INTO image_acquisition_jobs (
                            id, project_id, plan_id, plan_fingerprint, source_type,
                            status, active_key, worker_id, worker_generation,
                            claim_token, cancellation_requested, completed_at,
                            error_code, error_summary, manifest_relative_path,
                            manifest_warning, manifest_repair_state,
                            manifest_repair_attempted_at, downloader_version,
                            validator_version, importer_version, job_fingerprint,
                            created_at, updated_at
                        ) VALUES (
                            :id, :project_id, :plan_id, 'legacy-plan-fingerprint',
                            'danbooru', 'stale', NULL, NULL, 1, NULL, 0, NULL,
                            'INCOMPLETE_ITEM_STATE', 'INCOMPLETE_ITEM_STATE',
                            :manifest_relative_path,
                            'MANIFEST_REPAIR_PENDING:failed', 'pending', NULL,
                            'legacy', 'legacy', 'legacy', 'legacy-job-fingerprint',
                            '2026-08-05T00:00:00+00:00',
                            '2026-08-05T00:00:00+00:00'
                        )
                        """
                    ),
                    {
                        "id": job_id,
                        "project_id": str(project.id),
                        "plan_id": str(uuid4()),
                        "manifest_relative_path": relative_paths[label],
                    },
                )
                for display_order, status in enumerate(item_statuses[label]):
                    connection.execute(
                        text(
                            """
                            INSERT INTO image_acquisition_job_items (
                                id, job_id, plan_item_id, source_type,
                                external_post_id, display_order, status,
                                expected_metadata_fingerprint, part_relative_path,
                                created_at, updated_at
                            ) VALUES (
                                :id, :job_id, :plan_item_id, 'danbooru',
                                :external_post_id, :display_order, :status,
                                'legacy-item-fingerprint', :part_relative_path,
                                '2026-08-05T00:00:00+00:00',
                                '2026-08-05T00:00:00+00:00'
                            )
                            """
                        ),
                        {
                            "id": str(uuid4()),
                            "job_id": job_id,
                            "plan_item_id": str(uuid4()),
                            "external_post_id": str(display_order),
                            "display_order": display_order,
                            "status": status,
                            "part_relative_path": (
                                f"acquisition/jobs/{job_id}/parts/{display_order}.part"
                            ),
                        },
                    )
            transaction.commit()
        finally:
            if transaction.is_active:
                transaction.rollback()

    migrate(test_workspace, "0036_phase8c_manifest_repair_intent")
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE image_acquisition_jobs
                SET status = 'failed', completed_at = '2026-08-05T00:01:00+00:00',
                    manifest_warning = NULL, manifest_repair_state = NULL,
                    manifest_repair_attempted_at = NULL,
                    manifest_target_status = NULL,
                    manifest_target_error_code = NULL,
                    active_key = NULL, worker_id = NULL, claim_token = NULL,
                    current_item_id = NULL, heartbeat_at = NULL,
                    error_code = 'INCOMPLETE_ITEM_STATE',
                    error_summary = 'INCOMPLETE_ITEM_STATE'
                WHERE id IN (:completed_id, :mixed_id, :all_failed_id)
                """
            ),
            {
                "completed_id": job_ids["completed"],
                "mixed_id": job_ids["mixed"],
                "all_failed_id": job_ids["all-failed"],
            },
        )

    before_migration = {
        label: settings.projects_dir / str(project.id) / relative_path
        for label, relative_path in relative_paths.items()
    }
    assert all(path.exists() for path in before_migration.values())

    migrate(test_workspace, "head")
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT id, status, completed_at, manifest_repair_state,
                       manifest_target_status, manifest_target_error_code,
                       worker_id, claim_token, current_item_id
                FROM image_acquisition_jobs
                ORDER BY id
                """
            )
        ).all()
    migrated = {row[0]: row[1:] for row in rows}
    assert migrated[job_ids["completed"]][0:2] == ("stale", None)
    assert migrated[job_ids["completed"]][2:5] == (
        "pending",
        "completed",
        None,
    )
    assert migrated[job_ids["mixed"]][0:2] == ("stale", None)
    assert migrated[job_ids["mixed"]][2:5] == (
        "pending",
        "partially_completed",
        "INCOMPLETE_ITEM_STATE",
    )
    assert migrated[job_ids["all-failed"]][0] == "failed"
    assert migrated[job_ids["all-failed"]][1] is not None
    assert migrated[job_ids["all-failed"]][2:5] == (None, None, None)
    assert all(value is None for value in migrated[job_ids["completed"]][5:8])
    assert all(value is None for value in migrated[job_ids["mixed"]][5:8])
    assert all(path.exists() for path in before_migration.values())

    service = ImageAcquisitionDownloadService(settings, auto_start=False)
    assert service.reconcile_manifest_repairs(limit=3) == 2
    for label, job_id in job_ids.items():
        job = service.get_job(UUID(job_id))
        assert job is not None
        if label == "completed":
            assert job.status is ImageAcquisitionJobStatus.COMPLETED
        elif label == "mixed":
            assert job.status is ImageAcquisitionJobStatus.PARTIALLY_COMPLETED
        else:
            assert job.status is ImageAcquisitionJobStatus.FAILED
        if label != "all-failed":
            assert not before_migration[label].exists()
            with engine.connect() as connection:
                repaired_relative_path = connection.scalar(
                    text(
                        """
                        SELECT manifest_relative_path
                        FROM image_acquisition_jobs
                        WHERE id = :job_id
                        """
                    ),
                    {"job_id": job_id},
                )
            assert repaired_relative_path is not None
            repaired_path = (
                settings.projects_dir / str(project.id) / repaired_relative_path
            )
            assert repaired_path.exists()
            assert (
                json.loads(repaired_path.read_text(encoding="utf-8"))["status"]
                == job.status.value
            )
        else:
            assert before_migration[label].exists()


def test_phase8b_cleanup_retry_schedule_downgrade_and_reupgrade(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0033_phase8b_part_cleanup_claims")
        inspector = inspect(create_engine_for_settings(settings))
        item_columns = {
            column["name"]
            for column in inspector.get_columns("image_acquisition_job_items")
        }
        assert "part_cleanup_next_retry_at" not in item_columns
        assert "ix_image_acquisition_job_items_part_cleanup_schedule" not in {
            index["name"]
            for index in inspector.get_indexes("image_acquisition_job_items")
        }
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    with create_engine_for_settings(settings).connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0040_phase9a_training_completion_export"
        )


def test_phase8a_page_checkpoint_migration_downgrade_and_reupgrade(
    test_workspace: Path,
) -> None:
    settings = migrate(test_workspace)
    config = Config(str(Path("alembic.ini").resolve()))
    old_path = os.environ.get("RUNPOD_LORA_STUDIO_DATABASE_PATH")
    os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = str(settings.database_path)
    get_settings.cache_clear()
    try:
        command.downgrade(config, "0028_phase8a_claim_reservations")
        inspector = inspect(create_engine_for_settings(settings))
        assert (
            "image_source_search_cursor_checkpoints" not in inspector.get_table_names()
        )
        search_columns = {
            column["name"] for column in inspector.get_columns("image_source_searches")
        }
        assert "request_cursor" not in search_columns
        assert "completion_reason" not in search_columns
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
        if old_path is None:
            os.environ.pop("RUNPOD_LORA_STUDIO_DATABASE_PATH", None)
        else:
            os.environ["RUNPOD_LORA_STUDIO_DATABASE_PATH"] = old_path
    with create_engine_for_settings(settings).connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == (
            "0040_phase9a_training_completion_export"
        )


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
            "0040_phase9a_training_completion_export"
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
            "0040_phase9a_training_completion_export"
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
            "0040_phase9a_training_completion_export"
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
            "0040_phase9a_training_completion_export"
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
            "0040_phase9a_training_completion_export"
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
            "0040_phase9a_training_completion_export"
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
