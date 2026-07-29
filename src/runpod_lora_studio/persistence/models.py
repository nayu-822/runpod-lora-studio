from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):  # type: ignore[misc]
    pass


class ProjectRecord(Base):
    __tablename__ = "projects"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    concept_type: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_words: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    images: Mapped[list[ImageAssetRecord]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ImageAssetRecord(Base):
    __tablename__ = "image_assets"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id"), index=True, nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    thumbnail_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    selection_state: Mapped[str] = mapped_column(String(32), nullable=False)
    exclusion_reasons: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(32), default="upload", nullable=False
    )
    selection_source: Mapped[str] = mapped_column(
        String(32), default="manual", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    project: Mapped[ProjectRecord] = relationship(back_populates="images")
    inspection_results: Mapped[list[ImageInspectionResultRecord]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class ImageInspectionResultRecord(Base):
    __tablename__ = "image_inspection_results"
    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "rule_code",
            "detector_version",
            name="uq_image_inspection_rule_version",
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_assets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    rule_code: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason_ja: Mapped[str] = mapped_column(Text, nullable=False)
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    inspected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    image: Mapped[ImageAssetRecord] = relationship(back_populates="inspection_results")


class ImagePerceptualHashRecord(Base):
    __tablename__ = "image_perceptual_hashes"
    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "algorithm",
            "hash_size",
            "detector_version",
            name="uq_image_phash_configuration",
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_assets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    hash_value: Mapped[str | None] = mapped_column(String(256), nullable=True)
    hash_size: Mapped[int] = mapped_column(Integer, nullable=False)
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    image: Mapped[ImageAssetRecord] = relationship()


class SimilarityGroupRecord(Base):
    __tablename__ = "similarity_groups"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    group_type: Mapped[str] = mapped_column(String(32), nullable=False)
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    distance_threshold: Mapped[int] = mapped_column(Integer, nullable=False)
    representative_image_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="SET NULL"), nullable=True
    )
    representative_source: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    members: Mapped[list[SimilarityGroupMemberRecord]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )


class SimilarityGroupMemberRecord(Base):
    __tablename__ = "similarity_group_members"
    __table_args__ = (
        UniqueConstraint("group_id", "image_id", name="uq_similarity_group_image"),
        UniqueConstraint(
            "image_id", "detector_version", name="uq_similarity_image_version"
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("similarity_groups.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    image_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_assets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    representative_candidate_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_representative: Mapped[bool] = mapped_column(Integer, nullable=False, default=0)
    representative_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minimum_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_status: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    group: Mapped[SimilarityGroupRecord] = relationship(back_populates="members")
    image: Mapped[ImageAssetRecord] = relationship()


class SimilarityPairReviewRecord(Base):
    __tablename__ = "similarity_pair_reviews"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "image_left_id",
            "image_right_id",
            "detector_version",
            name="uq_similarity_pair_review",
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    image_left_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="CASCADE"), nullable=False
    )
    image_right_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="CASCADE"), nullable=False
    )
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    review_status: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TaggerRunRecord(Base):
    __tablename__ = "tagger_runs"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    adapter_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_identifier: Mapped[str] = mapped_column(String(256), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(128), nullable=False)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    device: Mapped[str] = mapped_column(String(16), nullable=False)
    general_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    character_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    save_rating: Mapped[bool] = mapped_column(Integer, nullable=False)
    save_character: Mapped[bool] = mapped_column(Integer, nullable=False)
    save_general: Mapped[bool] = mapped_column(Integer, nullable=False)
    underscore_to_space: Mapped[bool] = mapped_column(Integer, nullable=False)
    escape_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    max_workers: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    target_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    succeeded_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    skipped_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    current_image_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    implementation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    results: Mapped[list[ImageTaggingResultRecord]] = relationship(
        back_populates="tagger_run", cascade="all, delete-orphan"
    )


class ImageTaggingResultRecord(Base):
    __tablename__ = "image_tagging_results"
    __table_args__ = (
        UniqueConstraint(
            "image_id", "tagger_run_id", name="uq_image_tagging_result_run"
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="CASCADE"), index=True
    )
    tagger_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tagger_runs.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tagged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[DetectedTagRecord]] = relationship(
        back_populates="result", cascade="all, delete-orphan"
    )
    tagger_run: Mapped[TaggerRunRecord] = relationship(back_populates="results")
    image: Mapped[ImageAssetRecord] = relationship()


class DetectedTagRecord(Base):
    __tablename__ = "detected_tags"
    __table_args__ = (
        UniqueConstraint(
            "image_tagging_result_id",
            "original_order",
            name="uq_detected_tag_order",
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_tagging_result_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("image_tagging_results.internal_id", ondelete="CASCADE"),
        index=True,
    )
    tag_name_raw: Mapped[str] = mapped_column(String(512), nullable=False)
    tag_name_normalized: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str] = mapped_column(String(24), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    original_order: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    model_identifier: Mapped[str] = mapped_column(String(256), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(128), nullable=False)
    tagger_run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    result: Mapped[ImageTaggingResultRecord] = relationship(back_populates="tags")


class ImageCaptionRecord(Base):
    __tablename__ = "image_captions"
    __table_args__ = (
        Index(
            "uq_image_caption_current",
            "image_id",
            unique=True,
            sqlite_where=text("is_current = 1"),
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="CASCADE"), index=True
    )
    source_tagger_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tagger_runs.id", ondelete="SET NULL"), nullable=True
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    caption_text: Mapped[str] = mapped_column(Text, nullable=False)
    caption_format_version: Mapped[str] = mapped_column(String(32), nullable=False)
    is_current: Mapped[bool] = mapped_column(Integer, nullable=False)
    edit_source: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    tags: Mapped[list[CaptionTagRecord]] = relationship(
        back_populates="caption", cascade="all, delete-orphan"
    )
    image: Mapped[ImageAssetRecord] = relationship()


class CaptionTagRecord(Base):
    __tablename__ = "caption_tags"
    __table_args__ = (
        UniqueConstraint(
            "image_caption_id", "position", name="uq_caption_tag_position"
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_caption_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("image_captions.internal_id", ondelete="CASCADE"),
        index=True,
    )
    tag_name: Mapped[str] = mapped_column(String(512), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str] = mapped_column(String(24), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    manually_added: Mapped[bool] = mapped_column(Integer, nullable=False)
    manually_removed: Mapped[bool] = mapped_column(Integer, nullable=False)
    caption: Mapped[ImageCaptionRecord] = relationship(back_populates="tags")


class ProjectTagRuleRecord(Base):
    __tablename__ = "project_tag_rules"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "normalized_tag_name", name="uq_project_tag_rule_name"
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    normalized_tag_name: Mapped[str] = mapped_column(String(512), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    category: Mapped[str] = mapped_column(String(24), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_by: Mapped[str] = mapped_column(String(24), nullable=False)


class CaptionEditHistoryRecord(Base):
    __tablename__ = "caption_edit_history"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="CASCADE"), index=True
    )
    image_caption_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("image_captions.internal_id", ondelete="SET NULL"),
        nullable=True,
    )
    previous_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    new_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    before_text: Mapped[str] = mapped_column(Text, nullable=False)
    after_text: Mapped[str] = mapped_column(Text, nullable=False)
    diff_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    edit_source: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class DatasetSnapshotRecord(Base):
    __tablename__ = "dataset_snapshots"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    snapshot_version: Mapped[str] = mapped_column(String(32), nullable=False)
    generator_version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_project_version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_tagger_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tagger_runs.id", ondelete="SET NULL"), nullable=True
    )
    source_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    target_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    copied_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_image_count: Mapped[int] = mapped_column(Integer, nullable=False)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_root: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_toml_path: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_path: Mapped[str] = mapped_column(Text, nullable=False)
    report_path: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_toml_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    settings_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    validation_summary: Mapped[str] = mapped_column(Text, nullable=False)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    items: Mapped[list[DatasetSnapshotItemRecord]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )
    issues: Mapped[list[DatasetValidationIssueRecord]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[SnapshotCreationJobRecord]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )


class DatasetSnapshotItemRecord(Base):
    __tablename__ = "dataset_snapshot_items"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id", "image_id", name="uq_dataset_snapshot_item_image"
        ),
        UniqueConstraint(
            "snapshot_id", "sequence_number", name="uq_dataset_snapshot_item_seq"
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    snapshot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dataset_snapshots.id", ondelete="CASCADE"), index=True
    )
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="RESTRICT"), index=True
    )
    source_image_path: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_image_relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    caption_relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    aspect_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    caption_id: Mapped[str] = mapped_column(String(36), nullable=False)
    caption_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    caption_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    caption_text: Mapped[str] = mapped_column(Text, nullable=False)
    tag_count: Mapped[int] = mapped_column(Integer, nullable=False)
    trigger_word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_status: Mapped[str] = mapped_column(String(24), nullable=False)
    exact_duplicate_status: Mapped[str] = mapped_column(String(32), nullable=False)
    similarity_group_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    is_similarity_representative: Mapped[bool | None] = mapped_column(
        Integer, nullable=True
    )
    warnings_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    snapshot: Mapped[DatasetSnapshotRecord] = relationship(back_populates="items")


class DatasetValidationIssueRecord(Base):
    __tablename__ = "dataset_validation_issues"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    snapshot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dataset_snapshots.id", ondelete="CASCADE"), index=True
    )
    image_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="SET NULL"), nullable=True
    )
    issue_code: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    category: Mapped[str] = mapped_column(String(24), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    measured_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    threshold_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    snapshot: Mapped[DatasetSnapshotRecord] = relationship(back_populates="issues")


class SnapshotCreationJobRecord(Base):
    __tablename__ = "snapshot_creation_jobs"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    snapshot_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dataset_snapshots.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    cancel_requested: Mapped[bool] = mapped_column(Integer, nullable=False)
    current_step: Mapped[str] = mapped_column(String(64), nullable=False)
    processed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_count: Mapped[int] = mapped_column(Integer, nullable=False)
    current_image_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    snapshot: Mapped[DatasetSnapshotRecord] = relationship(back_populates="jobs")


class ManagedModelRecord(Base):
    __tablename__ = "managed_models"
    __table_args__ = (
        UniqueConstraint(
            "remote_name",
            "remote_relative_path",
            name="uq_managed_model_remote_path",
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(512), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    remote_name: Mapped[str] = mapped_column(String(128), nullable=False)
    remote_relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    remote_file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    remote_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    remote_modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    remote_hash_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    remote_hash_value: Mapped[str | None] = mapped_column(String(512), nullable=True)
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    local_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    rclone_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    downloaded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ModelTransferRecord(Base):
    __tablename__ = "model_transfers"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    managed_model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("managed_models.id", ondelete="CASCADE"), index=True
    )
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    destination_path: Mapped[str] = mapped_column(Text, nullable=False)
    expected_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    transferred_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_hash: Mapped[str | None] = mapped_column(String(512), nullable=True)
    actual_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    rclone_exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rclone_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    settings_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class StorageTransferJobRecord(Base):
    __tablename__ = "storage_transfer_jobs"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    snapshot_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    training_run_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    transfer_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    destination_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    current_step: Mapped[str] = mapped_column(String(128), nullable=False)
    item_count: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_item_count: Mapped[int] = mapped_column(Integer, nullable=False)
    succeeded_item_count: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_item_count: Mapped[int] = mapped_column(Integer, nullable=False)
    skipped_item_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    transferred_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_transferred_bytes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    current_file_transferred_bytes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    cancel_requested: Mapped[bool] = mapped_column(Integer, nullable=False)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    manifest_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TransferItemRecord(Base):
    __tablename__ = "storage_transfer_items"
    __table_args__ = (
        UniqueConstraint(
            "transfer_job_id",
            "relative_path",
            name="uq_storage_transfer_item_path",
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    transfer_job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("storage_transfer_jobs.id", ondelete="CASCADE"),
        index=True,
    )
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    item_type: Mapped[str] = mapped_column(String(32), nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    expected_size: Mapped[int] = mapped_column(Integer, nullable=False)
    transferred_size: Mapped[int] = mapped_column(Integer, nullable=False)
    source_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    destination_hash_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    destination_hash_value: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ProjectStorageSettingsRecord(Base):
    __tablename__ = "project_storage_settings"

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), unique=True
    )
    project_remote_root: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_remote_root: Mapped[str] = mapped_column(Text, nullable=False)
    training_remote_root: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_remote_root: Mapped[str] = mapped_column(Text, nullable=False)
    selected_managed_model_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    overwrite_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    verification_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingConfigRecord(Base):
    __tablename__ = "training_configs"
    __table_args__ = (
        Index("ix_training_configs_project_id", "project_id"),
        Index("ix_training_configs_snapshot_id", "dataset_snapshot_id"),
        Index("ix_training_configs_model_id", "managed_model_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    dataset_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("dataset_snapshots.id", ondelete="RESTRICT"),
        nullable=False,
    )
    managed_model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("managed_models.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    output_name: Mapped[str] = mapped_column(String(255), nullable=False)
    output_directory: Mapped[str] = mapped_column(Text, nullable=False)
    sd_scripts_root: Mapped[str] = mapped_column(Text, nullable=False)
    trainer_script: Mapped[str] = mapped_column(String(128), nullable=False)
    resolution: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
    optimizer: Mapped[str] = mapped_column(String(128), nullable=False)
    scheduler: Mapped[str] = mapped_column(String(128), nullable=False)
    network_module: Mapped[str] = mapped_column(String(128), nullable=False)
    network_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    network_alpha: Mapped[int] = mapped_column(Integer, nullable=False)
    mixed_precision: Mapped[str] = mapped_column(String(16), nullable=False)
    save_every_n_epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_latents: Mapped[bool] = mapped_column(Integer, nullable=False)
    gradient_checkpointing: Mapped[bool] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    extra_options: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    recommendation_engine_version: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    recommendation_change_diff: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ComputeEnvironmentSnapshotRecord(Base):
    __tablename__ = "compute_environment_snapshots"
    __table_args__ = (Index("ix_compute_environment_snapshots_project", "project_id"),)

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    warning_json: Mapped[str] = mapped_column(Text, nullable=False)
    error_json: Mapped[str] = mapped_column(Text, nullable=False)
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingEnvironmentSnapshotRecord(Base):
    __tablename__ = "training_environment_snapshots"
    __table_args__ = (Index("ix_training_environment_snapshots_project", "project_id"),)

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )
    compute_snapshot_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("compute_environment_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    warning_json: Mapped[str] = mapped_column(Text, nullable=False)
    error_json: Mapped[str] = mapped_column(Text, nullable=False)
    detector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingJobEnvironmentSnapshotRecord(Base):
    __tablename__ = "training_job_environment_snapshots"
    __table_args__ = (
        UniqueConstraint("training_job_id", name="uq_training_job_environment_job"),
        Index("ix_training_job_environment_job", "training_job_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    training_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False
    )
    logical_gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    physical_gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpu_uuid_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gpu_architecture: Mapped[str | None] = mapped_column(String(128), nullable=True)
    compute_capability: Mapped[str | None] = mapped_column(String(32), nullable=True)
    total_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cuda_available: Mapped[bool] = mapped_column(Integer, nullable=False)
    sd_scripts_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    xformers_available: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    cuda_visible_devices: Mapped[str] = mapped_column(
        String(512), nullable=False, default=""
    )
    visible_gpu_uuids_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    detector_version: Mapped[str] = mapped_column(String(64), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    warning_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")


class TrainingJobSelectedGpuRecord(Base):
    __tablename__ = "training_job_selected_gpus"
    __table_args__ = (
        UniqueConstraint("training_job_id", name="uq_training_job_selected_gpu_job"),
        Index("ix_training_job_selected_gpu_job", "training_job_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    training_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False
    )
    logical_gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    physical_gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpu_uuid_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    gpu_architecture: Mapped[str | None] = mapped_column(String(128), nullable=True)
    compute_capability: Mapped[str | None] = mapped_column(String(32), nullable=True)
    total_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    selection_source: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    warning_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    last_observed_gpu_uuid_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    gpu_change_detected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    gpu_change_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class TrainingRecommendationRequestRecord(Base):
    __tablename__ = "training_recommendation_requests"
    __table_args__ = (
        Index("ix_training_recommendation_requests_project", "project_id"),
        Index("ix_training_recommendation_requests_snapshot", "dataset_snapshot_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    dataset_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("dataset_snapshots.id", ondelete="RESTRICT"),
        nullable=False,
    )
    managed_model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("managed_models.id", ondelete="RESTRICT"), nullable=False
    )
    environment_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("compute_environment_snapshots.id", ondelete="RESTRICT"),
        nullable=False,
    )
    concept_type: Mapped[str] = mapped_column(String(32), nullable=False)
    quality_profile: Mapped[str] = mapped_column(String(32), nullable=False)
    speed_profile: Mapped[str] = mapped_column(String(32), nullable=False)
    user_constraints_json: Mapped[str] = mapped_column(Text, nullable=False)
    current_config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingRecommendationRecord(Base):
    __tablename__ = "training_recommendations"
    __table_args__ = (
        UniqueConstraint("request_id", "rank", name="uq_training_recommendation_rank"),
        Index("ix_training_recommendations_request", "request_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("training_recommendation_requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    profile_name: Mapped[str] = mapped_column(String(64), nullable=False)
    settings_json: Mapped[str] = mapped_column(Text, nullable=False)
    reasons_json: Mapped[str] = mapped_column(Text, nullable=False)
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False)
    settings_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(32), nullable=False)
    calibration_snapshot_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    calibration_applied: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    calibration_confidence: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )
    calibration_reason_codes_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    baseline_duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    calibrated_duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    baseline_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    calibrated_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    baseline_batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    calibrated_batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    calibration_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingExecutionSummaryRecord(Base):
    __tablename__ = "training_execution_summaries"
    __table_args__ = (
        UniqueConstraint("training_job_id", name="uq_execution_summary_job"),
        UniqueConstraint(
            "summary_fingerprint", name="uq_execution_summary_fingerprint"
        ),
        Index("ix_execution_summaries_project", "project_id"),
        Index(
            "ix_execution_summaries_gpu_settings",
            "gpu_identity_fingerprint",
            "settings_fingerprint",
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    training_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    training_config_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("training_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    recommendation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    training_job_environment_snapshot_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    environment_snapshot_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    dataset_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("dataset_snapshots.id", ondelete="RESTRICT"),
        nullable=False,
    )
    managed_model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("managed_models.id", ondelete="RESTRICT"), nullable=False
    )
    job_result_status: Mapped[str] = mapped_column(String(32), nullable=False)
    gpu_identity_fingerprint: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    selected_gpu_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    selected_gpu_warning_codes_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    gpu_architecture: Mapped[str | None] = mapped_column(String(128), nullable=True)
    gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    physical_gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    compute_capability: Mapped[str | None] = mapped_column(String(32), nullable=True)
    gpu_total_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_scale_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    settings_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolution: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gradient_accumulation_steps: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    effective_batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    network_module: Mapped[str | None] = mapped_column(String(128), nullable=True)
    network_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    network_alpha: Mapped[int | None] = mapped_column(Integer, nullable=True)
    optimizer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    scheduler: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mixed_precision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    cache_latents: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    gradient_checkpointing: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    world_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sd_scripts_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    xformers_available: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    total_epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    planned_total_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resume_initial_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resume_step_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    measured_steps_per_second: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    measured_images_per_second: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    peak_allocated_vram_bytes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    peak_reserved_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    free_vram_before_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    free_vram_after_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minimum_free_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    whole_gpu_peak_used_vram_bytes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    other_process_peak_vram_bytes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    memory_sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    memory_failed_sample_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    memory_confidence: Mapped[str] = mapped_column(
        String(16), nullable=False, default="none"
    )
    memory_first_sampled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    memory_last_sampled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    memory_coverage_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    process_identity_verified: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    gpu_identity_verified: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    measurement_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="phase7b-memory-v1"
    )
    memory_warning_codes_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    memory_failure_codes_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    oom_detected: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    failure_category: Mapped[str] = mapped_column(
        String(32), nullable=False, default="none"
    )
    failure_evidence_codes_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    usable_for_speed_calibration: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    usable_for_memory_calibration: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    exclusion_reasons_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    calibration_included: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=True
    )
    manual_exclusion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    collector_version: Mapped[str] = mapped_column(String(32), nullable=False)
    classifier_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="phase7b-failure-v1"
    )
    summary_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    summary_content_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default=""
    )
    calibration_state_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default=""
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingMemoryAggregateRecord(Base):
    __tablename__ = "training_memory_aggregates"
    __table_args__ = (
        UniqueConstraint("training_job_id", name="uq_training_memory_aggregate_job"),
        Index("ix_training_memory_aggregates_job", "training_job_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    training_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False
    )
    gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpu_uuid_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gpu_total_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    free_vram_before_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minimum_free_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    free_vram_after_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_process_peak_used_bytes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    whole_gpu_peak_used_bytes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    other_process_peak_used_bytes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_sampled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_sampled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    process_identity_verified: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    gpu_identity_verified: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    confidence: Mapped[str] = mapped_column(String(16), nullable=False, default="none")
    last_sample_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    measurement_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="phase7b-memory-v1"
    )
    warning_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    failure_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class RecommendationCalibrationSnapshotRecord(Base):
    __tablename__ = "recommendation_calibration_snapshots"
    __table_args__ = (
        UniqueConstraint("calibration_fingerprint", name="uq_calibration_fingerprint"),
        Index("ix_calibration_snapshots_scope", "scope_project_id"),
        Index(
            "ix_calibration_snapshots_match", "gpu_identity_fingerprint", "resolution"
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    scope_project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True
    )
    gpu_identity_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    gpu_total_vram_class: Mapped[str | None] = mapped_column(String(32), nullable=True)
    gpu_architecture: Mapped[str | None] = mapped_column(String(128), nullable=True)
    compute_capability: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resolution: Mapped[int | None] = mapped_column(Integer, nullable=True)
    batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gradient_accumulation_steps: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    effective_batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    network_module: Mapped[str | None] = mapped_column(String(128), nullable=True)
    network_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    network_alpha: Mapped[int | None] = mapped_column(Integer, nullable=True)
    world_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sd_scripts_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    xformers_available: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    optimizer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mixed_precision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    cache_latents: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    gradient_checkpointing: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    successful_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    oom_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    median_steps_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    lower_percentile_steps_per_second: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    median_peak_vram_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    upper_percentile_peak_vram_bytes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    calibration_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    calibration_version: Mapped[str] = mapped_column(String(32), nullable=False)
    source_summary_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    stale: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)


class RecommendationCalibrationSourceRecord(Base):
    __tablename__ = "recommendation_calibration_sources"
    __table_args__ = (
        UniqueConstraint("calibration_id", "summary_id", name="uq_calibration_source"),
        Index("ix_calibration_sources_summary", "summary_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    calibration_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("recommendation_calibration_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    summary_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("training_execution_summaries.id", ondelete="CASCADE"),
        nullable=False,
    )


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"
    __table_args__ = (
        Index("ix_training_jobs_project_id", "project_id"),
        Index(
            "uq_training_jobs_active_project",
            "project_id",
            unique=True,
            sqlite_where=text(
                "status IN ('queued', 'starting', 'running', 'cancel_requested')"
            ),
        ),
        Index(
            "uq_training_jobs_resume_request_fingerprint",
            "resume_request_fingerprint",
            unique=True,
            sqlite_where=text("resume_request_fingerprint IS NOT NULL"),
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    training_config_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("training_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    dataset_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("dataset_snapshots.id", ondelete="RESTRICT"),
        nullable=False,
    )
    managed_model_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("managed_models.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    cancel_requested: Mapped[bool] = mapped_column(Integer, nullable=False)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    worker_heartbeat: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    command_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    stdout_log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    stderr_log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    runtime_directory: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="RESTRICT"), nullable=True
    )
    resume_artifact_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("training_artifacts.id", ondelete="RESTRICT"),
        nullable=True,
    )
    resume_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resume_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resume_validation_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    resume_validation_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    resume_validation_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    initial_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    initial_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    initial_epoch_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    initial_step_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    progress_step_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_epoch_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resume_request_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    process_start_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    process_group_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    process_identity: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingResumeValidationRecord(Base):
    __tablename__ = "training_resume_validations"
    __table_args__ = (
        Index("ix_training_resume_validations_source_job", "source_job_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    source_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="RESTRICT")
    )
    source_artifact_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_artifacts.id", ondelete="RESTRICT")
    )
    target_training_config_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_configs.id", ondelete="RESTRICT")
    )
    source_state_relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_state_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    source_job_config_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    target_config_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    state_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_epoch_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    state_step_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    compatibility_status: Mapped[str] = mapped_column(String(32), nullable=False)
    compatibility_issues: Mapped[str] = mapped_column(Text, nullable=False)
    validated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    validator_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingProgressRecord(Base):
    __tablename__ = "training_progress"
    __table_args__ = (
        UniqueConstraint("training_job_id", name="uq_training_progress_job"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    training_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False
    )
    current_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    smoothed_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    steps_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    samples_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_remaining_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    latest_log_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    stdout_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stderr_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parser_version: Mapped[str] = mapped_column(String(32), nullable=False)
    parse_status: Mapped[str] = mapped_column(String(16), nullable=False)
    parse_warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_source: Mapped[str] = mapped_column(String(16), nullable=False)
    parser_state: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingMetricPointRecord(Base):
    __tablename__ = "training_metric_points"
    __table_args__ = (
        UniqueConstraint(
            "training_job_id", "metric_name", "step", name="uq_training_metric_step"
        ),
        Index(
            "ix_training_metrics_job_name_step",
            "training_job_id",
            "metric_name",
            "step",
        ),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    training_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    logged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TrainingArtifactRecord(Base):
    __tablename__ = "training_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "training_job_id", "relative_path", name="uq_training_artifact_path"
        ),
        Index("ix_training_artifacts_job", "training_job_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    training_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    validation_status: Mapped[str] = mapped_column(String(16), nullable=False)
    validation_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    validation_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ExternalImagePostRecord(Base):
    __tablename__ = "external_image_posts"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "external_post_id", name="uq_external_image_source_post"
        ),
        Index("ix_external_image_posts_source_md5", "source_type", "source_md5"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    external_post_id: Mapped[str] = mapped_column(String(32), nullable=False)
    post_url: Mapped[str] = mapped_column(Text, nullable=False)
    file_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    preview_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    sample_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_extension: Mapped[str | None] = mapped_column(String(16), nullable=True)
    rating: Mapped[str | None] = mapped_column(String(8), nullable=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_md5: Mapped[str | None] = mapped_column(String(128), nullable=True)
    normalized_tags_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    metadata_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_metadata_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    is_deleted: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    is_pending: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    is_flagged: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ImageSourceSearchRecord(Base):
    __tablename__ = "image_source_searches"
    __table_args__ = (Index("ix_image_source_searches_project", "project_id"),)

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    normalized_query: Mapped[str] = mapped_column(Text, nullable=False)
    query_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    query_version: Mapped[str] = mapped_column(String(64), nullable=False)
    adapter_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False)
    returned_post_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accepted_candidate_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    excluded_candidate_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    api_request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rate_limit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_cursor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    cancellation_requested: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ImageSourceSearchResultRecord(Base):
    __tablename__ = "image_source_search_results"
    __table_args__ = (
        UniqueConstraint(
            "search_id", "external_post_id", name="uq_image_search_result_post"
        ),
        Index("ix_image_source_search_results_search", "search_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    search_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_source_searches.id", ondelete="CASCADE"),
        nullable=False,
    )
    external_post_id: Mapped[str] = mapped_column(String(32), nullable=False)
    result_order: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_status: Mapped[str] = mapped_column(String(24), nullable=False)
    exclusion_reasons_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]"
    )
    already_imported: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    already_planned: Mapped[bool] = mapped_column(
        Integer, nullable=False, default=False
    )
    selected: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    metadata_fingerprint_at_search: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ImageAcquisitionPlanRecord(Base):
    __tablename__ = "image_acquisition_plans"
    __table_args__ = (Index("ix_image_acquisition_plans_project", "project_id"),)

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_search_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_source_searches.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    selected_count: Mapped[int] = mapped_column(Integer, nullable=False)
    skipped_existing_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    blocked_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    plan_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    plan_version: Mapped[str] = mapped_column(String(64), nullable=False)
    query_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    adapter_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ImageAcquisitionPlanItemRecord(Base):
    __tablename__ = "image_acquisition_plan_items"
    __table_args__ = (
        UniqueConstraint("plan_id", "external_post_id", name="uq_image_plan_item_post"),
        Index("ix_image_acquisition_plan_items_plan", "plan_id"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    plan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_acquisition_plans.id", ondelete="CASCADE"),
        nullable=False,
    )
    external_post_id: Mapped[str] = mapped_column(String(32), nullable=False)
    search_result_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_source_search_results.id", ondelete="RESTRICT"),
        nullable=False,
    )
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    planned_status: Mapped[str] = mapped_column(String(24), nullable=False)
    expected_metadata_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    expected_file_url_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    expected_md5: Mapped[str | None] = mapped_column(String(128), nullable=True)
    expected_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expected_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expected_extension: Mapped[str | None] = mapped_column(String(16), nullable=True)
    skip_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ExternalImageAssetLinkRecord(Base):
    __tablename__ = "external_image_asset_links"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "external_post_id",
            name="uq_external_image_asset_source_post",
        ),
        UniqueConstraint("image_asset_id", "source_type", name="uq_image_asset_source"),
    )

    internal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    image_asset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("image_assets.id", ondelete="CASCADE"), nullable=False
    )
    external_post_id: Mapped[str] = mapped_column(String(32), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


# Short compatibility name for service and test code.
InspectionResultRecord = ImageInspectionResultRecord


def record_values(record: Any) -> dict[str, Any]:
    return {
        column.name: getattr(record, column.name) for column in record.__table__.columns
    }
