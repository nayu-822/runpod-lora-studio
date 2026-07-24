from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import UUID

from PIL import Image, ImageOps

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    ImageAsset,
    ImageInspectionResult,
    InspectionRule,
    InspectionRunResult,
    InspectionStatus,
    InspectionSummary,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.repositories import (
    ImageInspectionRepository,
    ImageRepository,
)
from runpod_lora_studio.services.project_service import ProjectService, UserFacingError

logger = logging.getLogger("runpod_lora_studio.image_inspection")


class ImageInspectionService:
    """Run lightweight, repeatable image checks without changing source files."""

    detector_version = "phase2a-v1"

    def __init__(
        self, settings: AppSettings, projects: ProjectService | None = None
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)

    def inspect_image(
        self, project_id: UUID, image_id: UUID
    ) -> tuple[ImageInspectionResult, ...]:
        images = self._project_images(project_id)
        image = next((item for item in images if item.id == image_id), None)
        if image is None:
            raise UserFacingError("指定された画像が見つかりません。")
        results = self._inspect_one(image, self._duplicate_map(images))
        self._save_results(image.id, results)
        return results

    def inspect_project(
        self, project_id: UUID, image_ids: Iterable[UUID] | None = None
    ) -> InspectionRunResult:
        images = self._project_images(project_id)
        selected = set(image_ids) if image_ids is not None else None
        duplicate_map = self._duplicate_map(images)
        inspected = 0
        failed = 0
        for image in images:
            if selected is not None and image.id not in selected:
                continue
            try:
                results = self._inspect_one(image, duplicate_map)
            except Exception:
                logger.exception(
                    "image_inspection_failed project_id=%s image_id=%s",
                    project_id,
                    image.id,
                )
                results = self._failed_results(
                    image.id,
                    datetime.now(UTC),
                    "画像検査処理に失敗しました。",
                )
            try:
                self._save_results(image.id, results)
            except Exception:
                logger.exception(
                    "image_inspection_persistence_failed project_id=%s image_id=%s",
                    project_id,
                    image.id,
                )
                # A persistence failure counts once, even if computed results
                # also contained a failed rule.
                failed += 1
                continue
            inspected += 1
            failed += int(
                any(result.status is InspectionStatus.FAILED for result in results)
            )
        summary = self.get_summary(project_id)
        return InspectionRunResult(summary, inspected, failed)

    def reinspect_images(
        self, project_id: UUID, image_ids: Iterable[UUID]
    ) -> InspectionRunResult:
        return self.inspect_project(project_id, image_ids)

    def update_duplicate_groups(self, project_id: UUID) -> InspectionRunResult:
        return self.inspect_project(project_id)

    def get_results(self, image_id: UUID) -> list[ImageInspectionResult]:
        with self.session_factory() as session:
            return ImageInspectionRepository(session).list_for_image(
                image_id, self.detector_version
            )

    def get_project_results(
        self, project_id: UUID
    ) -> dict[UUID, list[ImageInspectionResult]]:
        with self.session_factory() as session:
            return ImageInspectionRepository(session).list_for_project(
                project_id, self.detector_version
            )

    def get_summary(self, project_id: UUID) -> InspectionSummary:
        with self.session_factory() as session:
            return ImageInspectionRepository(session).summary(
                project_id, self.detector_version
            )

    def _project_images(self, project_id: UUID) -> list[ImageAsset]:
        if self.projects.get(project_id) is None:
            raise UserFacingError("指定されたプロジェクトが見つかりません。")
        with self.session_factory() as session:
            return ImageRepository(session).list_all_for_project(project_id)

    def _save_results(
        self, image_id: UUID, results: tuple[ImageInspectionResult, ...]
    ) -> None:
        with self.session_factory() as session:
            ImageInspectionRepository(session).replace_for_image(
                image_id, results, self.detector_version
            )
            session.commit()

    def _inspect_one(
        self, image: ImageAsset, duplicate_map: dict[UUID, bool]
    ) -> tuple[ImageInspectionResult, ...]:
        now = datetime.now(UTC)
        if not image.original_path.is_file():
            return self._failed_results(
                image.id, now, "原画像ファイルが見つかりません。"
            )
        try:
            with Image.open(image.original_path) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                width, height = oriented.size
                gray = oriented.convert("L")
                sample = self._sample(gray)
                low_information_score = self._standard_deviation(sample)
                blur_score = self._laplacian_variance(sample)
        except Exception:
            return self._failed_results(
                image.id, now, "画像を読み込めないため検査できません。"
            )

        minimum_side = float(min(width, height))
        resolution_failed = (
            width < self.settings.inspection_min_width
            or height < self.settings.inspection_min_height
        )
        missing_dimensions: list[str] = []
        if width < self.settings.inspection_min_width:
            missing_dimensions.append(f"幅 {self.settings.inspection_min_width}px未満")
        if height < self.settings.inspection_min_height:
            missing_dimensions.append(
                f"高さ {self.settings.inspection_min_height}px未満"
            )
        aspect_ratio = max(width / height, height / width)
        aspect_failed = aspect_ratio > self.settings.inspection_max_aspect_ratio
        low_information = (
            low_information_score
            <= self.settings.inspection_low_information_stddev_threshold
        )
        blur = blur_score < self.settings.inspection_blur_score_threshold
        return (
            self._result(
                image.id,
                InspectionRule.EXACT_DUPLICATE,
                InspectionStatus.WARNING
                if duplicate_map[image.id]
                else InspectionStatus.PASS,
                1.0 if duplicate_map[image.id] else 0.0,
                0.0,
                "同一SHA-256の画像です。代表画像以外のため確認してください。"
                if duplicate_map[image.id]
                else "同一SHA-256の重複画像はありません。",
                now,
            ),
            self._result(
                image.id,
                InspectionRule.RESOLUTION_TOO_SMALL,
                InspectionStatus.WARNING
                if resolution_failed
                else InspectionStatus.PASS,
                minimum_side,
                float(
                    min(
                        self.settings.inspection_min_width,
                        self.settings.inspection_min_height,
                    )
                ),
                "解像度が不足しています: " + "、".join(missing_dimensions)
                if resolution_failed
                else "幅と高さは最低解像度を満たしています。",
                now,
            ),
            self._result(
                image.id,
                InspectionRule.ASPECT_RATIO_EXTREME,
                InspectionStatus.WARNING if aspect_failed else InspectionStatus.PASS,
                aspect_ratio,
                self.settings.inspection_max_aspect_ratio,
                f"縦横比が極端です（{aspect_ratio:.2f}）。"
                if aspect_failed
                else f"縦横比は許容範囲内です（{aspect_ratio:.2f}）。",
                now,
            ),
            self._result(
                image.id,
                InspectionRule.LOW_INFORMATION,
                InspectionStatus.WARNING if low_information else InspectionStatus.PASS,
                low_information_score,
                self.settings.inspection_low_information_stddev_threshold,
                (
                    "低情報量候補です（グレースケール標準偏差 "
                    f"{low_information_score:.2f}）。"
                )
                if low_information
                else (
                    "十分な画素情報があります（標準偏差 "
                    f"{low_information_score:.2f}）。"
                ),
                now,
            ),
            self._result(
                image.id,
                InspectionRule.BLUR_SCORE,
                InspectionStatus.WARNING if blur else InspectionStatus.PASS,
                blur_score,
                self.settings.inspection_blur_score_threshold,
                f"ぼけ候補です（鮮明度スコア {blur_score:.2f}）。"
                if blur
                else f"鮮明度スコアは許容範囲内です（{blur_score:.2f}）。",
                now,
            ),
        )

    def _failed_results(
        self, image_id: UUID, inspected_at: datetime, reason: str
    ) -> tuple[ImageInspectionResult, ...]:
        return tuple(
            self._result(
                image_id,
                rule,
                InspectionStatus.FAILED,
                None,
                None,
                reason,
                inspected_at,
            )
            for rule in InspectionRule
        )

    def _result(
        self,
        image_id: UUID,
        rule: InspectionRule,
        status: InspectionStatus,
        score: float | None,
        threshold: float | None,
        reason: str,
        inspected_at: datetime,
    ) -> ImageInspectionResult:
        return ImageInspectionResult(
            image_id=image_id,
            rule=rule,
            status=status,
            score=self._finite(score),
            threshold=self._finite(threshold),
            reason=reason,
            detector_version=self.detector_version,
            inspected_at=inspected_at,
        )

    @staticmethod
    def _finite(value: float | None) -> float | None:
        return value if value is None or math.isfinite(value) else None

    @staticmethod
    def _duplicate_map(images: list[ImageAsset]) -> dict[UUID, bool]:
        groups: dict[str, list[ImageAsset]] = {}
        for image in images:
            groups.setdefault(image.sha256, []).append(image)
        duplicate: dict[UUID, bool] = {image.id: False for image in images}
        for group in groups.values():
            if len(group) > 1:
                ordered = sorted(
                    group, key=lambda item: (item.created_at, str(item.id))
                )
                for image in ordered[1:]:
                    duplicate[image.id] = True
        return duplicate

    @staticmethod
    def _sample(image: Image.Image) -> Image.Image:
        sample = image.copy()
        sample.thumbnail((256, 256), Image.Resampling.BILINEAR)
        return sample

    @staticmethod
    def _standard_deviation(pixels: list[int] | Image.Image) -> float:
        if isinstance(pixels, Image.Image):
            pixels = list(pixels.getdata())
        if not pixels:
            return 0.0
        mean = math.fsum(pixels) / len(pixels)
        variance = math.fsum((pixel - mean) ** 2 for pixel in pixels) / len(pixels)
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def _laplacian_variance(image: Image.Image) -> float:
        if not isinstance(image, Image.Image):
            return 0.0
        width, height = image.size
        if width < 3 or height < 3:
            return 0.0
        pixels = image.load()
        values: list[float] = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                center = float(pixels[x, y])
                laplacian = (
                    4 * center
                    - float(pixels[x - 1, y])
                    - float(pixels[x + 1, y])
                    - float(pixels[x, y - 1])
                    - float(pixels[x, y + 1])
                )
                values.append(laplacian)
        return (
            ImageInspectionService._standard_deviation([int(value) for value in values])
            ** 2
        )


# Public alias that reads naturally at call sites and keeps Phase 2B extensible.
QualityInspectionService = ImageInspectionService
