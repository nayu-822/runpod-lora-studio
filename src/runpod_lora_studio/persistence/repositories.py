from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from runpod_lora_studio.domain.models import (
    ConceptType,
    ImageAsset,
    ImageInspectionResult,
    InspectionRule,
    InspectionStatus,
    InspectionSummary,
    PerceptualHash,
    PerceptualHashStatus,
    Project,
    ProjectStatus,
    RepresentativeSource,
    SelectionState,
    SimilarityGroup,
    SimilarityGroupMember,
    SimilarityReviewStatus,
    SimilaritySummary,
)
from runpod_lora_studio.persistence.models import (
    ImageAssetRecord,
    ImageInspectionResultRecord,
    ImagePerceptualHashRecord,
    ProjectRecord,
    SimilarityGroupMemberRecord,
    SimilarityGroupRecord,
    SimilarityPairReviewRecord,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def project_from_record(
    record: ProjectRecord, counts: Counter[str] | None = None
) -> Project:
    count_map = Counter(counts or {})
    return Project(
        id=UUID(record.id),
        name=record.name,
        description=record.description,
        concept_type=ConceptType(record.concept_type),
        trigger_words=tuple(json.loads(record.trigger_words)),
        status=ProjectStatus(record.status),
        schema_version=record.schema_version,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
        image_counts={state: count_map[state.value] for state in SelectionState},
    )


def image_from_record(record: ImageAssetRecord) -> ImageAsset:
    return ImageAsset(
        id=UUID(record.id),
        project_id=UUID(record.project_id),
        original_filename=record.original_filename,
        stored_filename=record.stored_filename,
        original_path=Path(record.original_path),
        thumbnail_path=Path(record.thumbnail_path),
        sha256=record.sha256,
        width=record.width,
        height=record.height,
        file_size=record.file_size,
        mime_type=record.mime_type,
        selection_state=SelectionState(record.selection_state),
        exclusion_reasons=tuple(json.loads(record.exclusion_reasons)),
        source_type=record.source_type,
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
        selection_source=record.selection_source,
    )


class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, project: Project) -> ProjectRecord:
        record = ProjectRecord(
            id=str(project.id),
            name=project.name,
            description=project.description,
            concept_type=project.concept_type.value,
            trigger_words=json.dumps(project.trigger_words, ensure_ascii=False),
            status=project.status.value,
            schema_version=project.schema_version,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )
        self.session.add(record)
        return record

    def get(self, project_id: UUID) -> ProjectRecord | None:
        return cast(
            ProjectRecord | None,
            self.session.scalar(
                select(ProjectRecord).where(ProjectRecord.id == str(project_id))
            ),
        )

    def list_for_record(self, record: ProjectRecord) -> Project:
        rows = self.session.execute(
            select(ImageAssetRecord.selection_state, func.count())
            .where(ImageAssetRecord.project_id == record.id)
            .group_by(ImageAssetRecord.selection_state)
        ).all()
        counts = Counter({state: count for state, count in rows})
        return project_from_record(record, counts)

    def list(self) -> list[Project]:
        records = self.session.scalars(
            select(ProjectRecord).order_by(ProjectRecord.updated_at.desc())
        ).all()
        if not records:
            return []
        project_ids = [record.id for record in records]
        rows = self.session.execute(
            select(
                ImageAssetRecord.project_id,
                ImageAssetRecord.selection_state,
                func.count(),
            )
            .where(ImageAssetRecord.project_id.in_(project_ids))
            .group_by(ImageAssetRecord.project_id, ImageAssetRecord.selection_state)
        ).all()
        counts: dict[str, Counter[str]] = {}
        for project_id, state, count in rows:
            counts.setdefault(project_id, Counter())[state] = count
        return [
            project_from_record(record, counts.get(record.id)) for record in records
        ]

    def update(
        self,
        record: ProjectRecord,
        *,
        name: str,
        description: str,
        concept_type: ConceptType,
        trigger_words: tuple[str, ...],
    ) -> ProjectRecord:
        record.name = name
        record.description = description
        record.concept_type = concept_type.value
        record.trigger_words = json.dumps(trigger_words, ensure_ascii=False)
        record.updated_at = utc_now()
        return record

    def touch(self, project_id: UUID) -> None:
        record = self.get(project_id)
        if record is not None:
            record.updated_at = utc_now()


class ImageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, image: ImageAsset) -> ImageAssetRecord:
        record = ImageAssetRecord(
            id=str(image.id),
            project_id=str(image.project_id),
            original_filename=image.original_filename,
            stored_filename=image.stored_filename,
            original_path=str(image.original_path),
            thumbnail_path=str(image.thumbnail_path),
            sha256=image.sha256,
            width=image.width,
            height=image.height,
            file_size=image.file_size,
            mime_type=image.mime_type,
            selection_state=image.selection_state.value,
            exclusion_reasons=json.dumps(image.exclusion_reasons, ensure_ascii=False),
            source_type=image.source_type,
            selection_source=image.selection_source,
            created_at=image.created_at,
            updated_at=image.updated_at,
        )
        self.session.add(record)
        return record

    def get(self, image_id: UUID) -> ImageAssetRecord | None:
        return cast(
            ImageAssetRecord | None,
            self.session.scalar(
                select(ImageAssetRecord).where(ImageAssetRecord.id == str(image_id))
            ),
        )

    def sha_exists(self, project_id: UUID, sha256: str) -> bool:
        return (
            self.session.scalar(
                select(ImageAssetRecord.internal_id).where(
                    ImageAssetRecord.project_id == str(project_id),
                    ImageAssetRecord.sha256 == sha256,
                )
            )
            is not None
        )

    def list_for_project(
        self,
        project_id: UUID,
        *,
        state: SelectionState | None = None,
        search: str = "",
        page: int = 1,
        page_size: int = 30,
    ) -> tuple[list[ImageAsset], int]:
        page = max(page, 1)
        page_size = max(page_size, 1)
        query = select(ImageAssetRecord).where(
            ImageAssetRecord.project_id == str(project_id)
        )
        count_query = (
            select(func.count())
            .select_from(ImageAssetRecord)
            .where(ImageAssetRecord.project_id == str(project_id))
        )
        if state is not None:
            query = query.where(ImageAssetRecord.selection_state == state.value)
            count_query = count_query.where(
                ImageAssetRecord.selection_state == state.value
            )
        if search.strip():
            pattern = f"%{search.strip()}%"
            query = query.where(ImageAssetRecord.original_filename.ilike(pattern))
            count_query = count_query.where(
                ImageAssetRecord.original_filename.ilike(pattern)
            )
        total = int(self.session.scalar(count_query) or 0)
        records = self.session.scalars(
            query.order_by(ImageAssetRecord.created_at.desc())
            .offset(max(page - 1, 0) * page_size)
            .limit(page_size)
        ).all()
        return [image_from_record(record) for record in records], total

    def list_all_for_project(self, project_id: UUID) -> list[ImageAsset]:
        records = self.session.scalars(
            select(ImageAssetRecord)
            .where(ImageAssetRecord.project_id == str(project_id))
            .order_by(ImageAssetRecord.created_at, ImageAssetRecord.id)
        ).all()
        return [image_from_record(record) for record in records]

    def iter_batches_for_project(
        self, project_id: UUID, batch_size: int
    ) -> Iterable[list[ImageAsset]]:
        """Yield deterministic image batches without materializing the project."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        last_internal_id = 0
        while True:
            records = self.session.scalars(
                select(ImageAssetRecord)
                .where(
                    ImageAssetRecord.project_id == str(project_id),
                    ImageAssetRecord.internal_id > last_internal_id,
                )
                .order_by(ImageAssetRecord.internal_id)
                .limit(batch_size)
            ).all()
            if not records:
                return
            yield [image_from_record(record) for record in records]
            last_internal_id = records[-1].internal_id

    def update_state(
        self,
        project_id: UUID,
        image_ids: Iterable[UUID],
        state: SelectionState,
    ) -> int:
        ids = [str(image_id) for image_id in image_ids]
        if not ids:
            return 0
        records = self.session.scalars(
            select(ImageAssetRecord).where(
                ImageAssetRecord.project_id == str(project_id),
                ImageAssetRecord.id.in_(ids),
            )
        ).all()
        now = utc_now()
        for record in records:
            record.selection_state = state.value
            record.selection_source = "manual"
            record.updated_at = now
        return len(records)


def inspection_from_record(
    record: ImageInspectionResultRecord,
) -> ImageInspectionResult:
    return ImageInspectionResult(
        image_id=UUID(record.image_id),
        rule=InspectionRule(record.rule_code),
        status=InspectionStatus(record.status),
        score=record.score,
        threshold=record.threshold,
        reason=record.reason_ja,
        detector_version=record.detector_version,
        inspected_at=_utc(record.inspected_at),
    )


class ImageInspectionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def replace_for_image(
        self,
        image_id: UUID,
        results: Iterable[ImageInspectionResult],
        detector_version: str,
    ) -> None:
        materialized = tuple(results)
        result_versions = {result.detector_version for result in materialized}
        if not materialized or result_versions != {detector_version}:
            raise ValueError(
                "All inspection results must use the requested detector_version."
            )
        image_key = str(image_id)
        self.session.query(ImageInspectionResultRecord).filter(
            ImageInspectionResultRecord.image_id == image_key,
            ImageInspectionResultRecord.detector_version == detector_version,
        ).delete(synchronize_session=False)
        for result in materialized:
            self.session.add(
                ImageInspectionResultRecord(
                    image_id=image_key,
                    rule_code=result.rule.value,
                    status=result.status.value,
                    score=result.score,
                    threshold=result.threshold,
                    reason_ja=result.reason,
                    detector_version=result.detector_version,
                    inspected_at=result.inspected_at,
                )
            )

    def list_for_image(
        self, image_id: UUID, detector_version: str | None = None
    ) -> list[ImageInspectionResult]:
        query = select(ImageInspectionResultRecord).where(
            ImageInspectionResultRecord.image_id == str(image_id)
        )
        if detector_version is not None:
            query = query.where(
                ImageInspectionResultRecord.detector_version == detector_version
            )
        records = self.session.scalars(
            query.order_by(ImageInspectionResultRecord.rule_code)
        ).all()
        return [inspection_from_record(record) for record in records]

    def list_for_project(
        self, project_id: UUID, detector_version: str | None = None
    ) -> dict[UUID, list[ImageInspectionResult]]:
        query = (
            select(ImageInspectionResultRecord)
            .join(ImageAssetRecord)
            .where(ImageAssetRecord.project_id == str(project_id))
        )
        if detector_version is not None:
            query = query.where(
                ImageInspectionResultRecord.detector_version == detector_version
            )
        rows = self.session.scalars(
            query.order_by(
                ImageInspectionResultRecord.image_id,
                ImageInspectionResultRecord.rule_code,
            )
        ).all()
        result: dict[UUID, list[ImageInspectionResult]] = {}
        for row in rows:
            result.setdefault(UUID(row.image_id), []).append(
                inspection_from_record(row)
            )
        return result

    def summary(
        self, project_id: UUID, detector_version: str | None = None
    ) -> InspectionSummary:
        image_query = (
            select(func.count())
            .select_from(ImageAssetRecord)
            .where(ImageAssetRecord.project_id == str(project_id))
        )
        total = int(self.session.scalar(image_query) or 0)
        query = (
            select(
                ImageInspectionResultRecord.image_id,
                ImageInspectionResultRecord.rule_code,
                ImageInspectionResultRecord.status,
            )
            .join(ImageAssetRecord)
            .where(ImageAssetRecord.project_id == str(project_id))
        )
        if detector_version is not None:
            query = query.where(
                ImageInspectionResultRecord.detector_version == detector_version
            )
        rows = self.session.execute(query).all()
        statuses_by_image: dict[str, set[str]] = {}
        rules_by_image: dict[str, set[str]] = {}
        warning_rules: dict[str, set[str]] = {
            rule.value: set() for rule in InspectionRule
        }
        for image_id, rule, status in rows:
            statuses_by_image.setdefault(image_id, set()).add(status)
            rules_by_image.setdefault(image_id, set()).add(rule)
            if status == InspectionStatus.WARNING.value and rule in warning_rules:
                warning_rules[rule].add(image_id)

        # Each image receives exactly one aggregate state. Unknown statuses and
        # incomplete rows are treated as failed so the summary remains safe and
        # its counts remain exhaustive.
        counts = {"pass": 0, "warning": 0, "failed": 0}
        expected_rules = {rule.value for rule in InspectionRule}
        for image_id, statuses in statuses_by_image.items():
            image_rules = rules_by_image.get(image_id, set())
            if image_rules != expected_rules:
                aggregate = InspectionStatus.FAILED.value
            elif InspectionStatus.FAILED.value in statuses:
                aggregate = InspectionStatus.FAILED.value
            elif InspectionStatus.WARNING.value in statuses:
                aggregate = InspectionStatus.WARNING.value
            elif (
                statuses == {InspectionStatus.PASS.value}
                and rules_by_image.get(image_id, set()) == expected_rules
            ):
                aggregate = InspectionStatus.PASS.value
            else:
                aggregate = InspectionStatus.FAILED.value
            counts[aggregate] += 1
        return InspectionSummary(
            project_id=project_id,
            total_images=total,
            inspected_images=len(statuses_by_image),
            pass_count=counts[InspectionStatus.PASS.value],
            warning_count=counts[InspectionStatus.WARNING.value],
            failed_count=counts[InspectionStatus.FAILED.value],
            exact_duplicate_count=len(
                warning_rules[InspectionRule.EXACT_DUPLICATE.value]
            ),
            resolution_too_small_count=len(
                warning_rules[InspectionRule.RESOLUTION_TOO_SMALL.value]
            ),
            aspect_ratio_extreme_count=len(
                warning_rules[InspectionRule.ASPECT_RATIO_EXTREME.value]
            ),
            low_information_count=len(
                warning_rules[InspectionRule.LOW_INFORMATION.value]
            ),
            blur_score_count=len(warning_rules[InspectionRule.BLUR_SCORE.value]),
        )


def perceptual_hash_from_record(record: ImagePerceptualHashRecord) -> PerceptualHash:
    return PerceptualHash(
        image_id=UUID(record.image_id),
        algorithm=record.algorithm,
        hash_value=record.hash_value or "",
        hash_size=record.hash_size,
        detector_version=record.detector_version,
        status=PerceptualHashStatus(record.status),
        calculated_at=_utc(record.calculated_at),
        error_summary=record.error_summary,
    )


class PerceptualHashRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, value: PerceptualHash) -> None:
        record = self.session.scalar(
            select(ImagePerceptualHashRecord).where(
                ImagePerceptualHashRecord.image_id == str(value.image_id),
                ImagePerceptualHashRecord.algorithm == value.algorithm,
                ImagePerceptualHashRecord.hash_size == value.hash_size,
                ImagePerceptualHashRecord.detector_version == value.detector_version,
            )
        )
        if record is None:
            record = ImagePerceptualHashRecord(
                image_id=str(value.image_id),
                algorithm=value.algorithm,
                hash_value=value.hash_value or None,
                hash_size=value.hash_size,
                detector_version=value.detector_version,
                status=value.status.value,
                calculated_at=value.calculated_at,
                error_summary=value.error_summary,
            )
            self.session.add(record)
            return
        record.hash_value = value.hash_value or None
        record.status = value.status.value
        record.calculated_at = value.calculated_at
        record.error_summary = value.error_summary

    def list_for_project(
        self,
        project_id: UUID,
        algorithm: str,
        hash_size: int,
        detector_version: str,
        *,
        calculated_only: bool = False,
    ) -> list[PerceptualHash]:
        query = (
            select(ImagePerceptualHashRecord)
            .join(ImageAssetRecord)
            .where(
                ImageAssetRecord.project_id == str(project_id),
                ImagePerceptualHashRecord.algorithm == algorithm,
                ImagePerceptualHashRecord.hash_size == hash_size,
                ImagePerceptualHashRecord.detector_version == detector_version,
            )
            .order_by(ImageAssetRecord.internal_id)
        )
        if calculated_only:
            query = query.where(
                ImagePerceptualHashRecord.status
                == PerceptualHashStatus.CALCULATED.value
            )
        return [
            perceptual_hash_from_record(record)
            for record in self.session.scalars(query).all()
        ]

    def status_counts(
        self, project_id: UUID, algorithm: str, hash_size: int, detector_version: str
    ) -> dict[str, int]:
        query = (
            select(ImagePerceptualHashRecord.status, func.count())
            .join(ImageAssetRecord)
            .where(
                ImageAssetRecord.project_id == str(project_id),
                ImagePerceptualHashRecord.algorithm == algorithm,
                ImagePerceptualHashRecord.hash_size == hash_size,
                ImagePerceptualHashRecord.detector_version == detector_version,
            )
            .group_by(ImagePerceptualHashRecord.status)
        )
        return {status: int(count) for status, count in self.session.execute(query)}


# Both spellings are kept for callers that use the acronym form from the spec.
PHashRepository = PerceptualHashRepository


def _group_from_record(record: SimilarityGroupRecord) -> SimilarityGroup:
    members = tuple(
        SimilarityGroupMember(
            group_id=UUID(member.group_id),
            image_id=UUID(member.image_id),
            representative_candidate_score=member.representative_candidate_score,
            is_representative=bool(member.is_representative),
            representative_distance=member.representative_distance,
            minimum_distance=member.minimum_distance,
            review_status=SimilarityReviewStatus(member.review_status),
            image=image_from_record(member.image) if member.image else None,
        )
        for member in sorted(record.members, key=lambda item: item.image_id)
    )
    return SimilarityGroup(
        id=UUID(record.id),
        project_id=UUID(record.project_id),
        group_type=record.group_type,
        detector_version=record.detector_version,
        distance_threshold=record.distance_threshold,
        representative_image_id=(
            UUID(record.representative_image_id)
            if record.representative_image_id
            else None
        ),
        representative_source=RepresentativeSource(record.representative_source),
        created_at=_utc(record.created_at),
        updated_at=_utc(record.updated_at),
        members=members,
    )


class SimilarityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def current_groups(
        self, project_id: UUID, detector_version: str
    ) -> list[SimilarityGroupRecord]:
        return list(
            self.session.scalars(
                select(SimilarityGroupRecord)
                .where(
                    SimilarityGroupRecord.project_id == str(project_id),
                    SimilarityGroupRecord.detector_version == detector_version,
                )
                .order_by(SimilarityGroupRecord.id)
            ).all()
        )

    def list_groups(
        self, project_id: UUID, detector_version: str, page: int, page_size: int
    ) -> tuple[list[SimilarityGroup], int]:
        count = int(
            self.session.scalar(
                select(func.count())
                .select_from(SimilarityGroupRecord)
                .where(
                    SimilarityGroupRecord.project_id == str(project_id),
                    SimilarityGroupRecord.detector_version == detector_version,
                )
            )
            or 0
        )
        rows = self.session.scalars(
            select(SimilarityGroupRecord)
            .where(
                SimilarityGroupRecord.project_id == str(project_id),
                SimilarityGroupRecord.detector_version == detector_version,
            )
            .order_by(SimilarityGroupRecord.created_at, SimilarityGroupRecord.id)
            .offset(max(page - 1, 0) * max(page_size, 1))
            .limit(max(page_size, 1))
        ).all()
        return [_group_from_record(row) for row in rows], count

    def get_group(self, group_id: UUID) -> SimilarityGroup | None:
        record = self.session.scalar(
            select(SimilarityGroupRecord).where(
                SimilarityGroupRecord.id == str(group_id)
            )
        )
        return _group_from_record(record) if record else None

    def replace_groups(
        self,
        project_id: UUID,
        detector_version: str,
        threshold: int,
        groups: Iterable[SimilarityGroup],
    ) -> None:
        old = self.current_groups(project_id, detector_version)
        manual_representatives = {
            member.image_id
            for group in old
            if group.representative_source == RepresentativeSource.MANUAL.value
            for member in group.members
            if member.is_representative
        }
        old_review: dict[str, str] = {
            member.image_id: member.review_status
            for group in old
            for member in group.members
        }
        for record in old:
            self.session.delete(record)
        self.session.flush()
        now = utc_now()
        for group in groups:
            manual_image = next(
                (
                    str(image_id)
                    for image_id in manual_representatives
                    if image_id in {str(member.image_id) for member in group.members}
                ),
                None,
            )
            representative_id = manual_image or (
                str(group.representative_image_id)
                if group.representative_image_id
                else None
            )
            source = (
                RepresentativeSource.MANUAL.value
                if manual_image
                else group.representative_source.value
            )
            record = SimilarityGroupRecord(
                id=str(group.id),
                project_id=str(project_id),
                group_type=group.group_type,
                detector_version=detector_version,
                distance_threshold=threshold,
                representative_image_id=representative_id,
                representative_source=source,
                created_at=now,
                updated_at=now,
            )
            self.session.add(record)
            for member in group.members:
                image_key = str(member.image_id)
                is_rep = image_key == representative_id
                review = old_review.get(image_key, member.review_status.value)
                record.members.append(
                    SimilarityGroupMemberRecord(
                        group_id=str(group.id),
                        image_id=image_key,
                        detector_version=detector_version,
                        representative_candidate_score=member.representative_candidate_score,
                        is_representative=is_rep,
                        representative_distance=member.representative_distance,
                        minimum_distance=member.minimum_distance,
                        review_status=review,
                        created_at=now,
                        updated_at=now,
                    )
                )

    def set_representative(self, group_id: UUID, image_id: UUID) -> None:
        group = self.session.scalar(
            select(SimilarityGroupRecord).where(
                SimilarityGroupRecord.id == str(group_id)
            )
        )
        if group is None:
            raise ValueError("similarity group not found")
        member = self.session.scalar(
            select(SimilarityGroupMemberRecord).where(
                SimilarityGroupMemberRecord.group_id == str(group_id),
                SimilarityGroupMemberRecord.image_id == str(image_id),
            )
        )
        if member is None:
            raise ValueError("image is not a member of the similarity group")
        for item in group.members:
            item.is_representative = item.image_id == str(image_id)
            item.updated_at = utc_now()
        group.representative_image_id = str(image_id)
        group.representative_source = RepresentativeSource.MANUAL.value
        group.updated_at = utc_now()

    def set_group_review(self, group_id: UUID, status: SimilarityReviewStatus) -> None:
        group = self.session.scalar(
            select(SimilarityGroupRecord).where(
                SimilarityGroupRecord.id == str(group_id)
            )
        )
        if group is None:
            raise ValueError("similarity group not found")
        now = utc_now()
        member_ids = sorted(member.image_id for member in group.members)
        for member in group.members:
            member.review_status = status.value
            member.updated_at = now
        for index, left in enumerate(member_ids):
            for right in member_ids[index + 1 :]:
                self._upsert_pair_review(
                    group.project_id, left, right, group.detector_version, status
                )

    def _upsert_pair_review(
        self,
        project_id: str,
        left: str,
        right: str,
        detector_version: str,
        status: SimilarityReviewStatus,
    ) -> None:
        first, second = sorted((left, right))
        record = self.session.scalar(
            select(SimilarityPairReviewRecord).where(
                SimilarityPairReviewRecord.project_id == project_id,
                SimilarityPairReviewRecord.image_left_id == first,
                SimilarityPairReviewRecord.image_right_id == second,
                SimilarityPairReviewRecord.detector_version == detector_version,
            )
        )
        now = utc_now()
        if record is None:
            self.session.add(
                SimilarityPairReviewRecord(
                    project_id=project_id,
                    image_left_id=first,
                    image_right_id=second,
                    detector_version=detector_version,
                    review_status=status.value,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            record.review_status = status.value
            record.updated_at = now

    def rejected_pairs(
        self, project_id: UUID, detector_version: str
    ) -> set[tuple[str, str]]:
        rows = self.session.scalars(
            select(SimilarityPairReviewRecord).where(
                SimilarityPairReviewRecord.project_id == str(project_id),
                SimilarityPairReviewRecord.detector_version == detector_version,
                SimilarityPairReviewRecord.review_status
                == SimilarityReviewStatus.REJECTED_SIMILARITY.value,
            )
        ).all()
        return {(row.image_left_id, row.image_right_id) for row in rows}

    def summary(
        self, project_id: UUID, algorithm: str, hash_size: int, detector_version: str
    ) -> SimilaritySummary:
        total = int(
            self.session.scalar(
                select(func.count())
                .select_from(ImageAssetRecord)
                .where(ImageAssetRecord.project_id == str(project_id))
            )
            or 0
        )
        counts = PerceptualHashRepository(self.session).status_counts(
            project_id, algorithm, hash_size, detector_version
        )
        groups, _ = self.list_groups(project_id, detector_version, 1, 2**31 - 1)
        unreviewed = sum(
            any(
                member.review_status is SimilarityReviewStatus.UNREVIEWED
                for member in group.members
            )
            for group in groups
        )
        return SimilaritySummary(
            project_id=project_id,
            calculated_count=counts.get(PerceptualHashStatus.CALCULATED.value, 0),
            uncalculated_count=max(total - sum(counts.values()), 0),
            failed_count=counts.get(PerceptualHashStatus.FAILED.value, 0),
            group_count=len(groups),
            candidate_image_count=sum(len(group.members) for group in groups),
            exact_only_group_count=sum(
                group.group_type == "exact_only" for group in groups
            ),
            unreviewed_group_count=unreviewed,
        )
