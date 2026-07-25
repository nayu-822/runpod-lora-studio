from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    ImageAsset,
    InspectionStatus,
    PerceptualHash,
    PerceptualHashStatus,
    RepresentativeCandidate,
    RepresentativeSource,
    SelectionState,
    SimilarityGroup,
    SimilarityGroupMember,
    SimilarityReviewStatus,
    SimilarityRunResult,
    SimilaritySummary,
)
from runpod_lora_studio.persistence.database import create_session_factory
from runpod_lora_studio.persistence.repositories import (
    ImageInspectionRepository,
    ImageRepository,
    PerceptualHashRepository,
    SimilarityRepository,
)
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.perceptual_hash_service import PerceptualHashService
from runpod_lora_studio.services.project_service import ProjectService, UserFacingError


@dataclass(frozen=True, slots=True)
class _Candidate:
    image: ImageAsset
    hash_value: str
    hash_size: int
    warning_count: int
    failed: bool
    blur_score: float
    is_exact_representative: bool


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for value in self.parent:
            groups.setdefault(self.find(value), []).append(value)
        return [sorted(values) for values in groups.values() if len(values) > 1]


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


class SimilarityDetectionService:
    """Build connected components from compatible pHash values."""

    def __init__(
        self,
        settings: AppSettings,
        projects: ProjectService | None = None,
        hashes: PerceptualHashService | None = None,
        images: ImageService | None = None,
    ) -> None:
        self.settings = settings
        self.projects = projects or ProjectService(settings)
        self.hashes = hashes or PerceptualHashService(settings, self.projects)
        self.images = images or ImageService(settings, self.projects)
        self.session_factory = create_session_factory(settings)

    def run_project(
        self, project_id: UUID, image_ids: list[UUID] | None = None
    ) -> SimilarityRunResult:
        if self.projects.get(project_id) is None:
            raise UserFacingError("プロジェクトを選択してください。")
        calculated, failed, skipped = self.hashes.calculate_project(
            project_id, image_ids
        )
        groups = self._regroup(project_id)
        return SimilarityRunResult(
            summary=self.get_summary(project_id),
            calculated_image_count=calculated,
            failed_image_count=failed,
            skipped_image_count=skipped,
            group_count=len(groups),
        )

    def _regroup(self, project_id: UUID) -> list[SimilarityGroup]:
        with self.session_factory() as session:
            hash_repo = PerceptualHashRepository(session)
            values = hash_repo.list_for_project(
                project_id,
                self.hashes.algorithm,
                self.hashes.hash_size,
                self.hashes.detector_version,
                calculated_only=True,
            )
            value_by_id = {str(value.image_id): value for value in values}
            candidates = self._candidates(session, project_id, value_by_id)
            rejected = SimilarityRepository(session).rejected_pairs(
                project_id, self.hashes.detector_version
            )
            ids = sorted(candidate.image.id.hex for candidate in candidates.values())
            uf = _UnionFind(ids)
            candidate_by_key = {
                candidate.image.id.hex: candidate for candidate in candidates.values()
            }
            for left_key, right_key in combinations(ids, 2):
                left, right = candidate_by_key[left_key], candidate_by_key[right_key]
                pair = _pair_key(str(left.image.id), str(right.image.id))
                if pair in rejected:
                    continue
                distance = PerceptualHashService.hamming_distance(
                    left.hash_value, left.hash_size, right.hash_value, right.hash_size
                )
                if distance <= self.settings.phash_distance_threshold:
                    uf.union(left_key, right_key)
            groups = self._make_groups(uf.components(), candidate_by_key)
            SimilarityRepository(session).replace_groups(
                project_id,
                self.hashes.detector_version,
                self.settings.phash_distance_threshold,
                groups,
            )
            session.commit()
            return groups

    def _candidates(
        self, session: Session, project_id: UUID, hashes: dict[str, PerceptualHash]
    ) -> dict[str, _Candidate]:
        inspection = ImageInspectionRepository(session).list_for_project(
            project_id, "phase2a-v1"
        )
        images: dict[str, ImageAsset] = {}
        exact_representatives: dict[str, ImageAsset] = {}
        repository = ImageRepository(session)
        for batch in repository.iter_batches_for_project(
            project_id, self.settings.phash_batch_size
        ):
            for image in batch:
                images[str(image.id)] = image
                current: ImageAsset | None = exact_representatives.get(image.sha256)
                if current is None or (image.created_at, str(image.id)) < (
                    current.created_at,
                    str(current.id),
                ):
                    exact_representatives[image.sha256] = image
        result: dict[str, _Candidate] = {}
        for image_key, value in hashes.items():
            image_record = images.get(image_key)
            if (
                image_record is None
                or value.status is not PerceptualHashStatus.CALCULATED
            ):
                continue
            results = inspection.get(image_record.id, [])
            failed = any(item.status is InspectionStatus.FAILED for item in results)
            warning_count = sum(
                item.status is InspectionStatus.WARNING for item in results
            )
            blur_scores = [
                item.score
                for item in results
                if item.rule.value == "blur_score" and item.score is not None
            ]
            result[image_key] = _Candidate(
                image=image_record,
                hash_value=value.hash_value,
                hash_size=value.hash_size,
                warning_count=warning_count,
                failed=failed,
                blur_score=blur_scores[0] if blur_scores else 0.0,
                is_exact_representative=exact_representatives[image_record.sha256].id
                == image_record.id,
            )
        return result

    def _make_groups(
        self, components: list[list[str]], candidates: dict[str, _Candidate]
    ) -> list[SimilarityGroup]:
        groups: list[SimilarityGroup] = []
        for component in sorted(components, key=lambda item: item[0]):
            members = [candidates[item] for item in component]
            distances: dict[tuple[str, str], int] = {}
            for left, right in combinations(members, 2):
                distance = PerceptualHashService.hamming_distance(
                    left.hash_value, left.hash_size, right.hash_value, right.hash_size
                )
                distances[_pair_key(str(left.image.id), str(right.image.id))] = distance
            representative = self._representative(members)
            rep_key = str(representative.image_id)
            group_type = (
                "exact_only"
                if len({item.image.sha256 for item in members}) == 1
                else "approximate"
            )
            group_id = uuid4()
            group_members: list[SimilarityGroupMember] = []
            for candidate in sorted(members, key=lambda item: str(item.image.id)):
                key = str(candidate.image.id)
                rep_distance = (
                    None if key == rep_key else distances.get(_pair_key(key, rep_key))
                )
                minimum = min(
                    (distance for pair, distance in distances.items() if key in pair),
                    default=None,
                )
                score, reason = self._candidate_score(candidate)
                group_members.append(
                    SimilarityGroupMember(
                        group_id=group_id,
                        image_id=candidate.image.id,
                        representative_candidate_score=score,
                        is_representative=key == rep_key,
                        representative_distance=rep_distance,
                        minimum_distance=minimum,
                        review_status=SimilarityReviewStatus.UNREVIEWED,
                        image=candidate.image,
                    )
                )
            groups.append(
                SimilarityGroup(
                    id=group_id,
                    project_id=members[0].image.project_id,
                    group_type=group_type,
                    detector_version=self.hashes.detector_version,
                    distance_threshold=self.settings.phash_distance_threshold,
                    representative_image_id=representative.image_id,
                    representative_source=RepresentativeSource.AUTOMATIC,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    members=tuple(group_members),
                )
            )
        return groups

    @staticmethod
    def _candidate_score(candidate: _Candidate) -> tuple[float, str]:
        score = (
            (0.0 if candidate.failed else 1_000_000.0)
            - candidate.warning_count * 10_000.0
            + candidate.image.width * candidate.image.height / 1_000_000.0
            + min(max(candidate.blur_score, 0.0), 100_000.0) / 1_000.0
            + (100.0 if candidate.is_exact_representative else 0.0)
        )
        reason = (
            f"failed={'yes' if candidate.failed else 'no'}, "
            f"warnings={candidate.warning_count}, "
            f"resolution={candidate.image.width}x{candidate.image.height}, "
            f"blur={candidate.blur_score:.2f}, "
            f"exact_rep={'yes' if candidate.is_exact_representative else 'no'}"
        )
        return score, reason

    @classmethod
    def _representative(cls, candidates: list[_Candidate]) -> RepresentativeCandidate:
        scored = [
            (cls._candidate_score(candidate), candidate) for candidate in candidates
        ]
        selected = max(
            scored,
            key=lambda item: (
                item[0][0],
                item[1].image.width * item[1].image.height,
                item[1].blur_score,
                1 if item[1].is_exact_representative else 0,
                -item[1].image.created_at.timestamp(),
                str(item[1].image.id),
            ),
        )[1]
        score, reason = cls._candidate_score(selected)
        return RepresentativeCandidate(selected.image.id, score, reason)

    def get_summary(self, project_id: UUID) -> SimilaritySummary:
        with self.session_factory() as session:
            return SimilarityRepository(session).summary(
                project_id,
                self.hashes.algorithm,
                self.hashes.hash_size,
                self.hashes.detector_version,
            )

    def list_groups(
        self, project_id: UUID, page: int = 1, page_size: int | None = None
    ) -> tuple[list[SimilarityGroup], int]:
        with self.session_factory() as session:
            return SimilarityRepository(session).list_groups(
                project_id,
                self.hashes.detector_version,
                page,
                page_size or self.settings.similarity_group_page_size,
            )

    def get_group(self, group_id: UUID) -> SimilarityGroup | None:
        with self.session_factory() as session:
            return SimilarityRepository(session).get_group(group_id)

    def change_representative(self, group_id: UUID, image_id: UUID) -> None:
        with self.session_factory() as session:
            SimilarityRepository(session).set_representative(group_id, image_id)
            session.commit()

    def review_group(self, group_id: UUID, similar: bool) -> None:
        status = (
            SimilarityReviewStatus.CONFIRMED_SIMILAR
            if similar
            else SimilarityReviewStatus.REJECTED_SIMILARITY
        )
        with self.session_factory() as session:
            SimilarityRepository(session).set_group_review(group_id, status)
            session.commit()

    def change_image_state(
        self, project_id: UUID, image_ids: list[UUID], state: SelectionState
    ) -> int:
        return self.images.change_state(project_id, image_ids, state)
