from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from runpod_lora_studio.domain.models import (
    TagCategory,
    TaggerInferenceSettings,
    TaggerModelIdentity,
    TaggingResult,
    TagPrediction,
)
from runpod_lora_studio.external.tagger import (
    ValidationResult,
    preprocess_image,
)


class FakeTaggerAdapter:
    """Deterministic adapter used by unit tests and local smoke checks."""

    adapter_name = "fake"

    def __init__(
        self,
        predictions: Mapping[str, tuple[TagPrediction, ...]] | None = None,
        failing_filenames: set[str] | None = None,
    ) -> None:
        self.predictions = dict(predictions or {})
        self.failing_filenames = set(failing_filenames or set())
        self.load_count = 0
        self.unload_count = 0
        self.tag_count = 0
        self._loaded = False

    def model_identity(self) -> TaggerModelIdentity:
        return TaggerModelIdentity(
            adapter_name=self.adapter_name,
            model_identifier="fake-tagger",
            model_revision="test-v1",
            model_path="fake://deterministic",
            implementation_version="fake-v1",
        )

    def validate_environment(self) -> ValidationResult:
        return ValidationResult(True, "FakeTagger is ready", "cpu")

    def load(self) -> None:
        self.load_count += 1
        self._loaded = True

    def tag_image(
        self, image_path: Path, settings: TaggerInferenceSettings
    ) -> TaggingResult:
        if not self._loaded:
            raise RuntimeError("fake tagger is not loaded")
        self.tag_count += 1
        if image_path.name in self.failing_filenames:
            raise RuntimeError("deterministic fake inference failure")
        preprocess_image(image_path)
        tags = self.predictions.get(
            image_path.name,
            (
                TagPrediction(
                    tag_name_raw="character",
                    tag_name_normalized="character",
                    category=TagCategory.CHARACTER,
                    confidence=0.99,
                    original_order=0,
                ),
                TagPrediction(
                    tag_name_raw="blue_hair",
                    tag_name_normalized="blue_hair",
                    category=TagCategory.GENERAL,
                    confidence=0.9,
                    original_order=1,
                ),
            ),
        )
        return TaggingResult(tags=tags, raw_output=image_path.name)

    def unload(self) -> None:
        self.unload_count += 1
        self._loaded = False
