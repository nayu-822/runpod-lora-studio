from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
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


# Short compatibility name for service and test code.
InspectionResultRecord = ImageInspectionResultRecord


def record_values(record: Any) -> dict[str, Any]:
    return {
        column.name: getattr(record, column.name) for column in record.__table__.columns
    }
