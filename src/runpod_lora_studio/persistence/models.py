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


# Short compatibility name for service and test code.
InspectionResultRecord = ImageInspectionResultRecord


def record_values(record: Any) -> dict[str, Any]:
    return {
        column.name: getattr(record, column.name) for column in record.__table__.columns
    }
