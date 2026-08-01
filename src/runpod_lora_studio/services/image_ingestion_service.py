from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image, ImageFile, ImageOps
from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.acquisition_download_models import (
    ImageSourceProvenance,
    VerifiedImageFile,
)
from runpod_lora_studio.domain.models import ImageAsset, SelectionState
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    ExternalImageAssetLinkRecord,
    ImageAssetRecord,
)
from runpod_lora_studio.persistence.repositories import (
    ImageRepository,
    ProjectRepository,
    image_from_record,
)
from runpod_lora_studio.services.project_service import ProjectService, UserFacingError

logger = logging.getLogger("runpod_lora_studio.image_ingestion")
ImageFile.LOAD_TRUNCATED_IMAGES = False
VALIDATOR_VERSION = "phase8b-image-validator-v1"
IMPORTER_VERSION = "phase8b-image-importer-v1"
_MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")


class ImageVerificationError(UserFacingError):
    def __init__(self, message: str, *, code: str = "IMAGE_CORRUPTED") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class VerifiedImportResult:
    image: ImageAsset
    linked_existing: bool
    skipped_download: bool


class VerifiedImageIngestionService:
    """Validate an image and atomically register it with source provenance."""

    version = IMPORTER_VERSION

    def __init__(
        self, settings: AppSettings, projects: ProjectService | None = None
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)

    def inspect_image(
        self,
        path: Path,
        *,
        expected_md5: str | None = None,
        expected_file_size: int | None = None,
        expected_width: int | None = None,
        expected_height: int | None = None,
        expected_extension: str | None = None,
    ) -> VerifiedImageFile:
        if not path.is_file() or path.is_symlink():
            raise ImageVerificationError(
                "staging file is not a regular file", code="STAGING_PATH_INVALID"
            )
        file_size = path.stat().st_size
        if file_size <= 0:
            raise ImageVerificationError("image file is empty", code="EMPTY_FILE")
        if file_size > self.settings.max_upload_file_size_bytes:
            raise ImageVerificationError(
                "image file is too large", code="FILE_TOO_LARGE"
            )
        if expected_file_size is not None and file_size != expected_file_size:
            raise ImageVerificationError(
                "received size does not match source metadata",
                code="RECEIVED_SIZE_MISMATCH",
            )
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    md5.update(chunk)
                    sha256.update(chunk)
        except OSError as exc:
            raise ImageVerificationError(
                "image hash calculation failed", code="SHA256_FAILED"
            ) from exc
        calculated_md5 = md5.hexdigest()
        calculated_sha256 = sha256.hexdigest()
        if expected_md5 is not None:
            if not _MD5_RE.fullmatch(expected_md5):
                raise ImageVerificationError(
                    "source MD5 metadata is invalid", code="SOURCE_MD5_INVALID"
                )
            if calculated_md5.lower() != expected_md5.lower():
                raise ImageVerificationError(
                    "source MD5 does not match", code="SOURCE_MD5_MISMATCH"
                )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image_format = (image.format or "").upper()
                    width, height = image.size
                    if width * height > self.settings.max_image_pixels:
                        raise ImageVerificationError(
                            "image pixel count is too large",
                            code="IMAGE_PIXEL_LIMIT_EXCEEDED",
                        )
                    image.load()
        except ImageVerificationError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ImageVerificationError(
                "image pixel count is too large", code="IMAGE_PIXEL_LIMIT_EXCEEDED"
            ) from exc
        except Exception as exc:
            raise ImageVerificationError(
                "image is corrupt or truncated", code="IMAGE_CORRUPTED"
            ) from exc
        details = {
            "JPEG": (".jpg", "image/jpeg"),
            "PNG": (".png", "image/png"),
            "WEBP": (".webp", "image/webp"),
        }
        if image_format not in details:
            raise ImageVerificationError(
                "image format is not supported", code="UNSUPPORTED_IMAGE_TYPE"
            )
        extension, mime_type = details[image_format]
        if expected_extension and not _extension_matches(
            expected_extension, image_format
        ):
            raise ImageVerificationError(
                "image format does not match source metadata",
                code="IMAGE_FORMAT_MISMATCH",
            )
        if (
            expected_width is not None
            and expected_height is not None
            and (width, height) != (expected_width, expected_height)
        ):
            raise ImageVerificationError(
                "image dimensions do not match source metadata",
                code="IMAGE_DIMENSION_MISMATCH",
            )
        return VerifiedImageFile(
            path=path,
            sha256=calculated_sha256,
            md5=calculated_md5,
            file_size=file_size,
            width=width,
            height=height,
            detected_format=image_format,
            mime_type=mime_type,
            extension=extension,
        )

    def import_verified(
        self,
        project_id: UUID,
        verified: VerifiedImageFile,
        provenance: ImageSourceProvenance,
    ) -> VerifiedImportResult:
        staging_path = self._safe_staging_path(project_id, verified.path)
        project_root = self._safe_project_root(project_id)
        original_dir = project_root / "originals"
        thumbnail_dir = project_root / "thumbnails"
        original_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        asset_id = uuid4()
        final_original = original_dir / f"{asset_id}{verified.extension}"
        final_thumbnail = thumbnail_dir / f"{asset_id}.png"
        temporary_original = original_dir / f".{asset_id}.original.tmp"
        temporary_thumbnail = self.settings.temp_dir / f"{asset_id}.thumbnail.tmp"
        source_type = provenance.source_type.value
        now = datetime.now(UTC)
        try:
            with self.session_factory() as session:
                existing_link = session.scalar(
                    select(ExternalImageAssetLinkRecord).where(
                        ExternalImageAssetLinkRecord.source_type == source_type,
                        ExternalImageAssetLinkRecord.external_post_id
                        == provenance.external_post_id,
                    )
                )
                if existing_link is not None:
                    existing = session.scalar(
                        select(ImageAssetRecord).where(
                            ImageAssetRecord.id == existing_link.image_asset_id
                        )
                    )
                    if existing is None:
                        raise ImageVerificationError(
                            "source link points to a missing image",
                            code="DUPLICATE_SOURCE_CONFLICT",
                        )
                    session.commit()
                    return VerifiedImportResult(image_from_record(existing), True, True)
                existing = session.scalar(
                    select(ImageAssetRecord).where(
                        ImageAssetRecord.project_id == str(project_id),
                        ImageAssetRecord.sha256 == verified.sha256,
                    )
                )
                if existing is not None:
                    session.add(
                        self._link_record(existing.id, project_id, provenance, now)
                    )
                    session.commit()
                    return VerifiedImportResult(
                        image_from_record(existing), True, False
                    )
                self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staging_path, temporary_original)
                _fsync_file(temporary_original)
                self.create_thumbnail(temporary_original, temporary_thumbnail)
                image = ImageAsset(
                    id=asset_id,
                    project_id=project_id,
                    original_filename=f"{provenance.external_post_id}{verified.extension}",
                    stored_filename=final_original.name,
                    original_path=final_original,
                    thumbnail_path=final_thumbnail,
                    sha256=verified.sha256,
                    width=verified.width,
                    height=verified.height,
                    file_size=verified.file_size,
                    mime_type=verified.mime_type,
                    selection_state=SelectionState.PENDING,
                    exclusion_reasons=(),
                    source_type=source_type,
                    created_at=now,
                    updated_at=now,
                    selection_source="acquisition",
                )
                ImageRepository(session).add(image)
                session.flush()
                session.add(
                    self._link_record(str(asset_id), project_id, provenance, now)
                )
                ProjectRepository(session).touch(project_id)
                session.flush()
                os.replace(temporary_original, final_original)
                os.replace(temporary_thumbnail, final_thumbnail)
                session.commit()
                return VerifiedImportResult(image, False, False)
        except Exception:
            _cleanup_paths(
                (
                    temporary_original,
                    temporary_thumbnail,
                    final_original,
                    final_thumbnail,
                )
            )
            raise

    def create_thumbnail(self, source: Path, destination: Path) -> None:
        temporary = self.settings.temp_dir / f"{uuid4()}.thumbnail"
        try:
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image)
                image.thumbnail(
                    (self.settings.thumbnail_size, self.settings.thumbnail_size)
                )
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
                image.save(temporary, format="PNG")
            os.replace(temporary, destination)
        except Exception as exc:
            _cleanup_paths((temporary, destination))
            raise ImageVerificationError(
                "thumbnail generation failed", code="THUMBNAIL_FAILED"
            ) from exc

    def _safe_project_root(self, project_id: UUID) -> Path:
        projects_root = self.settings.projects_dir.resolve()
        root = self.projects.project_root(project_id)
        if root.is_symlink():
            raise ImageVerificationError(
                "project path must not be a symlink", code="STAGING_PATH_INVALID"
            )
        resolved = root.resolve()
        if not resolved.is_relative_to(projects_root):
            raise ImageVerificationError(
                "project path is outside the projects directory",
                code="STAGING_PATH_INVALID",
            )
        if resolved.exists() and resolved.is_symlink():
            raise ImageVerificationError(
                "project path must not be a symlink", code="STAGING_PATH_INVALID"
            )
        if not resolved.exists():
            raise ImageVerificationError(
                "project directory does not exist", code="STAGING_PATH_INVALID"
            )
        return resolved

    def _safe_staging_path(self, project_id: UUID, path: Path) -> Path:
        root = self._safe_project_root(project_id)
        resolved = path.resolve()
        staging_root = (root / "acquisition").resolve()
        if (
            not resolved.is_relative_to(staging_root)
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ImageVerificationError(
                "staging path is invalid", code="STAGING_PATH_INVALID"
            )
        return resolved

    @staticmethod
    def _link_record(
        image_asset_id: str,
        project_id: UUID,
        provenance: ImageSourceProvenance,
        now: datetime,
    ) -> ExternalImageAssetLinkRecord:
        return ExternalImageAssetLinkRecord(
            id=str(uuid4()),
            image_asset_id=image_asset_id,
            project_id=str(project_id),
            external_post_id=provenance.external_post_id,
            source_type=provenance.source_type.value,
            source_md5=provenance.source_md5,
            source_metadata_fingerprint=provenance.source_metadata_fingerprint,
            acquisition_plan_id=str(provenance.acquisition_plan_id),
            acquisition_job_id=str(provenance.acquisition_job_id),
            acquisition_job_item_id=str(provenance.acquisition_job_item_id),
            linked_at=now,
            created_at=now,
        )


def _extension_matches(extension: str, image_format: str) -> bool:
    normalized = extension.lower()
    return (
        image_format == "JPEG" and normalized in {".jpg", ".jpeg"}
    ) or normalized == {
        "PNG": ".png",
        "WEBP": ".webp",
    }.get(image_format, "")


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("image_ingestion_cleanup_failed filename=%s", path.name)
