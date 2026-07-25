from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("imagehash")

from PIL import Image

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import ImageAsset, SelectionState
from runpod_lora_studio.services.perceptual_hash_service import PerceptualHashService


def _image(path: Path) -> ImageAsset:
    now = datetime.now(UTC)
    return ImageAsset(
        id=uuid4(),
        project_id=uuid4(),
        original_filename=path.name,
        stored_filename=path.name,
        original_path=path,
        thumbnail_path=path,
        sha256="0" * 64,
        width=32,
        height=32,
        file_size=path.stat().st_size,
        mime_type="image/png",
        selection_state=SelectionState.PENDING,
        exclusion_reasons=(),
        source_type="upload",
        created_at=now,
        updated_at=now,
    )


def test_same_normalized_image_has_same_phash(test_workspace: Path) -> None:
    path = test_workspace / "solid.png"
    Image.new("RGB", (32, 32), "red").save(path)
    settings = AppSettings(database_path=test_workspace / "db.sqlite3")
    service = PerceptualHashService(settings)

    first = service.calculate(_image(path))
    second = service.calculate(_image(path))

    assert first.hash_value == second.hash_value
    assert len(first.hash_value) == 16


def test_hamming_distance_validates_configuration_and_includes_threshold() -> None:
    assert PerceptualHashService.hamming_distance("0", 2, "1", 2) == 1
    with pytest.raises(ValueError):
        PerceptualHashService.hamming_distance("0", 2, "00", 4)
    with pytest.raises(ValueError):
        PerceptualHashService.hamming_distance("z", 2, "0", 2)
