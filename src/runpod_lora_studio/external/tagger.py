from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from runpod_lora_studio.domain.models import (
    TaggerInferenceSettings,
    TaggerModelIdentity,
    TaggingResult,
    TagPrediction,
)


class TaggerError(RuntimeError):
    """Base error for model validation and inference failures."""


class TaggerEnvironmentError(TaggerError):
    """The adapter cannot run in the current environment."""


class TaggerInferenceError(TaggerError):
    """An image could not be tagged."""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    message: str
    resolved_device: str


@dataclass(frozen=True, slots=True)
class PreparedTaggerImage:
    image: Image.Image
    input_size: int
    channel_order: str = "RGB"
    normalization: str = "uint8_rgb_white_background"


class TaggerBackend(Protocol):
    def load(self, model_path: Path, settings: TaggerInferenceSettings) -> None: ...

    def predict(
        self, image: PreparedTaggerImage, settings: TaggerInferenceSettings
    ) -> Iterable[TagPrediction]: ...

    def unload(self) -> None: ...


class TaggerAdapter(Protocol):
    adapter_name: str

    def model_identity(self) -> TaggerModelIdentity: ...

    def validate_environment(self) -> ValidationResult: ...

    def load(self) -> None: ...

    def tag_image(
        self, image_path: Path, settings: TaggerInferenceSettings
    ) -> TaggingResult: ...

    def unload(self) -> None: ...


def normalize_tag_name(raw: str, underscore_to_space: bool = False) -> str:
    value = raw.replace("\r", " ").replace("\n", " ").strip()
    value = re.sub(r"\s+", " ", value)
    if underscore_to_space:
        value = value.replace("_", " ")
        value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise ValueError("tag is empty")
    if len(value) > 512:
        raise ValueError("tag is too long")
    if any(ord(char) < 32 and char not in "\t" for char in value):
        raise ValueError("tag contains a control character")
    return value.casefold()


def preprocess_image(image_path: Path, input_size: int = 448) -> PreparedTaggerImage:
    if input_size < 32 or input_size > 2048:
        raise ValueError("invalid tagger input size")
    try:
        with Image.open(image_path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            rgb = background.convert("RGB")
            contained = ImageOps.contain(
                rgb, (input_size, input_size), Image.Resampling.LANCZOS
            )
            canvas = Image.new("RGB", (input_size, input_size), (255, 255, 255))
            canvas.paste(
                contained,
                (
                    (input_size - contained.width) // 2,
                    (input_size - contained.height) // 2,
                ),
            )
            return PreparedTaggerImage(canvas, input_size)
    except Exception as exc:
        raise TaggerInferenceError("画像の前処理に失敗しました。") from exc
