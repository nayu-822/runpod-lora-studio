from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import DatasetSnapshotStatus
from runpod_lora_studio.domain.recommendation_models import TrainingDatasetStatistics
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.models import (
    DatasetSnapshotItemRecord,
    DatasetSnapshotRecord,
)


class DatasetStatisticsService:
    """Calculate immutable training inputs from a completed dataset snapshot."""

    analyzer_version = "phase7a-dataset-statistics-v1"

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.session_factory = create_session_factory(settings)

    def calculate(
        self, snapshot_id: UUID, *, trigger_words: tuple[str, ...] = ()
    ) -> TrainingDatasetStatistics:
        with self.session_factory() as session:
            snapshot = session.scalar(
                select(DatasetSnapshotRecord).where(
                    DatasetSnapshotRecord.id == str(snapshot_id)
                )
            )
            if snapshot is None:
                raise ValueError("dataset snapshot not found")
            if snapshot.status != DatasetSnapshotStatus.COMPLETED.value:
                raise ValueError("dataset snapshot must be completed")
            items = list(
                session.scalars(
                    select(DatasetSnapshotItemRecord)
                    .where(DatasetSnapshotItemRecord.snapshot_id == str(snapshot_id))
                    .order_by(DatasetSnapshotItemRecord.sequence_number)
                ).all()
            )
            try:
                document = tomllib.loads(
                    Path(snapshot.dataset_toml_path).read_text(encoding="utf-8")
                )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ValueError("dataset TOML could not be read") from exc

        subset_counts = _subset_image_counts(document, items)
        repeats = _subset_repeats(document, len(subset_counts))
        effective_count = sum(
            count * repeat for count, repeat in zip(subset_counts, repeats, strict=True)
        )
        widths = [item.width for item in items if item.width > 0]
        heights = [item.height for item in items if item.height > 0]
        ratios = [item.aspect_ratio for item in items if item.aspect_ratio > 0]
        trigger_coverage = _trigger_coverage(items, trigger_words, document)
        duplicate_count = sum(
            item.exact_duplicate_status.lower()
            in {"duplicate", "non_representative", "exact_duplicate"}
            for item in items
        )
        groups = {
            item.similarity_group_id for item in items if item.similarity_group_id
        }
        unreviewed = {
            item.similarity_group_id
            for item in items
            if item.similarity_group_id
            and item.is_similarity_representative is not True
        }
        bucket_step = _positive_int(
            document.get("general", {}).get("bucket_reso_steps"), 64
        )
        buckets = {
            (
                max(1, round(item.width / bucket_step)),
                max(1, round(item.height / bucket_step)),
            )
            for item in items
            if item.width > 0 and item.height > 0
        }
        warnings: list[str] = []
        if not items:
            warnings.append("DATASET_TOO_SMALL")
        if any(not item.caption_text.strip() for item in items):
            warnings.append("EMPTY_CAPTION_FOUND")
        return TrainingDatasetStatistics(
            snapshot_id=snapshot_id,
            image_count=len(items),
            effective_image_count=effective_count,
            subset_count=len(subset_counts),
            subset_image_counts=tuple(subset_counts),
            repeats=tuple(repeats),
            caption_count=sum(bool(item.caption_text.strip()) for item in items),
            empty_caption_count=sum(not item.caption_text.strip() for item in items),
            trigger_word_coverage=trigger_coverage,
            duplicate_ratio=duplicate_count / len(items) if items else None,
            similarity_group_count=len(groups),
            unreviewed_similarity_group_count=len(unreviewed),
            min_width=min(widths) if widths else None,
            max_width=max(widths) if widths else None,
            min_height=min(heights) if heights else None,
            max_height=max(heights) if heights else None,
            mean_aspect_ratio=sum(ratios) / len(ratios) if ratios else None,
            min_aspect_ratio=min(ratios) if ratios else None,
            max_aspect_ratio=max(ratios) if ratios else None,
            bucket_count=len(buckets) if items else None,
            content_sha256=snapshot.content_sha256,
            dataset_toml_sha256=(
                snapshot.dataset_toml_sha256
                or _sha256(Path(snapshot.dataset_toml_path))
            ),
            analyzer_version=self.analyzer_version,
            warnings=tuple(warnings),
        )


def _subset_image_counts(
    document: dict[str, object], items: list[DatasetSnapshotItemRecord]
) -> list[int]:
    datasets = document.get("datasets")
    if not isinstance(datasets, list):
        return [len(items)] if items else []
    counts: list[int] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        subsets = dataset.get("subsets")
        if not isinstance(subsets, list):
            continue
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            image_dir = subset.get("image_dir")
            if not isinstance(image_dir, str):
                continue
            prefix = image_dir.rstrip("/") + "/"
            counts.append(
                sum(
                    item.snapshot_image_relative_path == image_dir
                    or item.snapshot_image_relative_path.startswith(prefix)
                    for item in items
                )
            )
    return counts or ([len(items)] if items else [])


def _subset_repeats(document: dict[str, object], count: int) -> list[int]:
    values: list[int] = []
    datasets = document.get("datasets")
    if isinstance(datasets, list):
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            subsets = dataset.get("subsets")
            if not isinstance(subsets, list):
                continue
            for subset in subsets:
                if isinstance(subset, dict):
                    value = _positive_int(subset.get("num_repeats"), 1)
                    values.append(value)
    if not values:
        values = [1] * count
    if len(values) < count:
        values.extend([1] * (count - len(values)))
    return values[:count]


def _trigger_coverage(
    items: list[DatasetSnapshotItemRecord],
    trigger_words: tuple[str, ...],
    document: dict[str, object],
) -> float | None:
    words = tuple(word.strip() for word in trigger_words if word.strip())
    if not words:
        datasets = document.get("datasets")
        if isinstance(datasets, list):
            for dataset in datasets:
                if not isinstance(dataset, dict):
                    continue
                subsets = dataset.get("subsets")
                if not isinstance(subsets, list):
                    continue
                for subset in subsets:
                    if isinstance(subset, dict) and isinstance(
                        subset.get("class_tokens"), str
                    ):
                        words += tuple(
                            value.strip()
                            for value in str(subset["class_tokens"]).split(",")
                            if value.strip()
                        )
    if not words:
        return (
            sum(item.trigger_word_count > 0 for item in items) / len(items)
            if items
            else None
        )
    return (
        sum(any(word in item.caption_text for word in words) for item in items)
        / len(items)
        if items
        else 0.0
    )


def _positive_int(value: object, default: int) -> int:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
