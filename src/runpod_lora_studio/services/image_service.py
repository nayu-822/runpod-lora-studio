from __future__ import annotations

import hashlib
import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image, ImageFile, ImageOps

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import ImageAsset, SelectionState
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.repositories import (
    ImageRepository,
    ProjectRepository,
)
from runpod_lora_studio.services.project_service import ProjectService, UserFacingError

logger = logging.getLogger("runpod_lora_studio.images")
ImageFile.LOAD_TRUNCATED_IMAGES = False


@dataclass(frozen=True, slots=True)
class UploadFailure:
    filename: str
    reason: str


@dataclass(frozen=True, slots=True)
class UploadResult:
    successes: tuple[ImageAsset, ...]
    failures: tuple[UploadFailure, ...]
    duplicate_warning_count: int = 0


class ImageService:
    def __init__(
        self, settings: AppSettings, projects: ProjectService | None = None
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.session_factory = create_session_factory(settings)

    def register_uploads(
        self, project_id: UUID, uploads: Iterable[str | Path]
    ) -> UploadResult:
        if self.projects.get(project_id) is None:
            raise UserFacingError("指定されたプロジェクトが見つかりません。")
        paths = list(uploads)
        if len(paths) > self.settings.max_upload_files:
            raise UserFacingError(
                f"一度に登録できる画像は{self.settings.max_upload_files}枚までです。"
            )
        successes: list[ImageAsset] = []
        failures: list[UploadFailure] = []
        duplicate_warnings = 0
        for upload in paths:
            try:
                image, is_duplicate = self._register_one(project_id, Path(upload))
                successes.append(image)
                duplicate_warnings += int(is_duplicate)
            except (OSError, UserFacingError, ValueError) as exc:
                failures.append(UploadFailure(Path(upload).name, str(exc)))
            except Exception:
                logger.exception("image_registration_failed project_id=%s", project_id)
                failures.append(
                    UploadFailure(
                        Path(upload).name, "画像登録中に内部エラーが発生しました。"
                    )
                )
        return UploadResult(tuple(successes), tuple(failures), duplicate_warnings)

    def _register_one(self, project_id: UUID, source: Path) -> tuple[ImageAsset, bool]:
        if not source.is_file():
            raise UserFacingError("アップロードファイルが見つかりません。")
        if source.stat().st_size > self.settings.max_upload_file_size_bytes:
            raise UserFacingError("ファイルサイズ上限を超えています。")
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        asset_id = uuid4()
        temporary = self.settings.temp_dir / f"{asset_id}.upload"
        project_root = self.projects.project_root(project_id)
        original_dir = project_root / "originals"
        thumbnail_dir = project_root / "thumbnails"
        original_path: Path | None = None
        thumbnail_path: Path | None = None
        try:
            shutil.copyfile(source, temporary)
            digest = hashlib.sha256()
            with temporary.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            image_format, width, height = self._inspect_image(temporary)
            extension, mime_type = self._format_details(image_format)
            stored_filename = f"{asset_id}{extension}"
            original_path = original_dir / stored_filename
            thumbnail_path = thumbnail_dir / f"{asset_id}.png"
            original_dir.mkdir(parents=True, exist_ok=True)
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, original_path)
            self._create_thumbnail(original_path, thumbnail_path)
            now = datetime.now(UTC)
            image = ImageAsset(
                id=asset_id,
                project_id=project_id,
                original_filename=source.name,
                stored_filename=stored_filename,
                original_path=original_path,
                thumbnail_path=thumbnail_path,
                sha256=digest.hexdigest(),
                width=width,
                height=height,
                file_size=original_path.stat().st_size,
                mime_type=mime_type,
                selection_state=SelectionState.PENDING,
                exclusion_reasons=(),
                source_type="upload",
                created_at=now,
                updated_at=now,
            )
            with self.session_factory() as session:
                repository = ImageRepository(session)
                duplicate = repository.sha_exists(project_id, image.sha256)
                repository.add(image)
                ProjectRepository(session).touch(project_id)
                session.commit()
            logger.info(
                "image_registered project_id=%s image_id=%s", project_id, asset_id
            )
            return image, duplicate
        except Exception:
            for path in (temporary, original_path, thumbnail_path):
                if path is not None:
                    path.unlink(missing_ok=True)
            raise

    def _inspect_image(self, path: Path) -> tuple[str, int, int]:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_format = (image.format or "").upper()
                width, height = image.size
                if width * height > self.settings.max_image_pixels:
                    raise UserFacingError("画像のピクセル数上限を超えています。")
                image.load()
        except UserFacingError:
            raise
        except Exception as exc:
            raise UserFacingError("画像を読み込めないか、破損しています。") from exc
        if image_format not in {"JPEG", "PNG", "WEBP"}:
            raise UserFacingError("対応形式はJPEG、PNG、WebPです。")
        return image_format, width, height

    @staticmethod
    def _format_details(image_format: str) -> tuple[str, str]:
        values = {
            "JPEG": (".jpg", "image/jpeg"),
            "PNG": (".png", "image/png"),
            "WEBP": (".webp", "image/webp"),
        }
        return values[image_format]

    def _create_thumbnail(self, source: Path, destination: Path) -> None:
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
            temporary.unlink(missing_ok=True)
            raise UserFacingError("サムネイル生成に失敗しました。") from exc

    def list_images(
        self,
        project_id: UUID,
        *,
        state: SelectionState | None = None,
        search: str = "",
        page: int = 1,
        page_size: int = 30,
    ) -> tuple[list[ImageAsset], int]:
        with self.session_factory() as session:
            return ImageRepository(session).list_for_project(
                project_id, state=state, search=search, page=page, page_size=page_size
            )

    def change_state(
        self, project_id: UUID, image_ids: Iterable[UUID], state: SelectionState
    ) -> int:
        with self.session_factory() as session:
            repository = ImageRepository(session)
            records = [repository.get(image_id) for image_id in image_ids]
            if any(record is None for record in records):
                raise UserFacingError("指定された画像が見つかりません。")
            if any(
                record.project_id != str(project_id) for record in records if record
            ):
                raise UserFacingError("別のプロジェクトの画像は変更できません。")
            count = repository.update_state(
                project_id, [UUID(record.id) for record in records if record], state
            )
            ProjectRepository(session).touch(project_id)
            session.commit()
            return count
