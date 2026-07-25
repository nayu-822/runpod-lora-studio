from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import UUID

from PIL import Image, ImageOps

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    ImageAsset,
    PerceptualHash,
    PerceptualHashStatus,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.repositories import (
    ImageRepository,
    PerceptualHashRepository,
)
from runpod_lora_studio.services.project_service import ProjectService, UserFacingError

logger = logging.getLogger("runpod_lora_studio.perceptual_hash")


class PerceptualHashService:
    """Calculate and persist normalized imagehash pHash values."""

    algorithm = "phash"
    detector_version = "phase2b-v1"

    def __init__(
        self, settings: AppSettings, projects: ProjectService | None = None
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)

    @property
    def hash_size(self) -> int:
        return self.settings.phash_hash_size

    def calculate(self, image: ImageAsset) -> PerceptualHash:
        now = datetime.now(UTC)
        try:
            if not image.original_path.is_file():
                raise FileNotFoundError("image file is missing")
            with Image.open(image.original_path) as source:
                source.load()
                normalized = self._normalize(source)
                try:
                    import imagehash
                except ImportError as exc:
                    raise RuntimeError("ImageHash dependency is not installed") from exc
                value = str(imagehash.phash(normalized, hash_size=self.hash_size))
            self.validate_hash(value, self.hash_size)
            return PerceptualHash(
                image_id=image.id,
                algorithm=self.algorithm,
                hash_value=value,
                hash_size=self.hash_size,
                detector_version=self.detector_version,
                status=PerceptualHashStatus.CALCULATED,
                calculated_at=now,
            )
        except Exception as exc:
            logger.warning("phash_calculation_failed image_id=%s", image.id)
            return PerceptualHash(
                image_id=image.id,
                algorithm=self.algorithm,
                hash_value="",
                hash_size=self.hash_size,
                detector_version=self.detector_version,
                status=PerceptualHashStatus.FAILED,
                calculated_at=now,
                error_summary=str(exc)[:500],
            )

    def calculate_and_save(self, image: ImageAsset) -> PerceptualHash:
        result = self.calculate(image)
        with self.session_factory() as session:
            PerceptualHashRepository(session).upsert(result)
            session.commit()
        return result

    def calculate_project(
        self, project_id: UUID, image_ids: Iterable[UUID] | None = None
    ) -> tuple[int, int, int]:
        """Process image files in bounded batches."""
        if self.projects.get(project_id) is None:
            raise UserFacingError("プロジェクトを選択してください。")
        selected = {str(item) for item in image_ids} if image_ids is not None else None
        calculated = failed = skipped = 0
        with self.session_factory() as session:
            repository = ImageRepository(session)
            for batch in repository.iter_batches_for_project(
                project_id, self.settings.phash_batch_size
            ):
                for image in batch:
                    if selected is not None and str(image.id) not in selected:
                        skipped += 1
                        continue
                    result = self.calculate(image)
                    try:
                        PerceptualHashRepository(session).upsert(result)
                        session.commit()
                    except Exception:
                        session.rollback()
                        logger.exception(
                            "phash_persistence_failed image_id=%s", image.id
                        )
                        failed += 1
                        continue
                    if result.status is PerceptualHashStatus.CALCULATED:
                        calculated += 1
                    else:
                        failed += 1
        return calculated, failed, skipped

    def get_project_hashes(self, project_id: UUID) -> list[PerceptualHash]:
        with self.session_factory() as session:
            return PerceptualHashRepository(session).list_for_project(
                project_id,
                self.algorithm,
                self.hash_size,
                self.detector_version,
                calculated_only=True,
            )

    def get_status_counts(self, project_id: UUID) -> dict[str, int]:
        with self.session_factory() as session:
            return PerceptualHashRepository(session).status_counts(
                project_id, self.algorithm, self.hash_size, self.detector_version
            )

    @staticmethod
    def _normalize(source: Image.Image) -> Image.Image:
        """Apply EXIF orientation and composite transparency on an opaque background."""
        oriented = ImageOps.exif_transpose(source)
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        normalized = background.convert("RGB")
        normalized.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return normalized

    @staticmethod
    def validate_hash(value: str, hash_size: int) -> None:
        if hash_size < 1 or hash_size > 64:
            raise ValueError("invalid hash_size")
        expected_length = (hash_size * hash_size + 3) // 4
        if (
            len(value) != expected_length
            or re.fullmatch(r"[0-9a-fA-F]+", value) is None
        ):
            raise ValueError("invalid perceptual hash")

    @classmethod
    def hamming_distance(
        cls, left: str, left_hash_size: int, right: str, right_hash_size: int
    ) -> int:
        cls.validate_hash(left, left_hash_size)
        cls.validate_hash(right, right_hash_size)
        if left_hash_size != right_hash_size:
            raise ValueError("hash_size mismatch")
        return (int(left, 16) ^ int(right, 16)).bit_count()


PHashService = PerceptualHashService
