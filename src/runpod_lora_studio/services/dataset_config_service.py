from __future__ import annotations

# ruff: noqa: E501
import json
import re
import tomllib
from dataclasses import asdict
from pathlib import Path

from runpod_lora_studio.domain.models import (
    DatasetIssueCategory,
    DatasetIssueSeverity,
    DatasetSettings,
    DatasetValidationIssue,
)


class DatasetConfigService:
    """Validate and serialize the Phase 4 dataset TOML safely."""

    def validate(
        self, settings: DatasetSettings, image_count: int = 1
    ) -> tuple[DatasetValidationIssue, ...]:
        issues: list[DatasetValidationIssue] = []

        def error(code: str, message: str, measured: object = None) -> None:
            issues.append(
                DatasetValidationIssue(
                    code,
                    DatasetIssueSeverity.ERROR,
                    DatasetIssueCategory.CONFIGURATION,
                    message,
                    measured_value=None if measured is None else str(measured),
                )
            )

        def warning(code: str, message: str, measured: object = None) -> None:
            issues.append(
                DatasetValidationIssue(
                    code,
                    DatasetIssueSeverity.WARNING,
                    DatasetIssueCategory.CONFIGURATION,
                    message,
                    measured_value=None if measured is None else str(measured),
                )
            )

        if settings.resolution <= 0:
            error("resolution_invalid", "resolutionは正の整数で指定してください。")
        elif settings.resolution < 256:
            warning(
                "resolution_low",
                "resolutionがSDXL用途として低すぎる可能性があります。",
                settings.resolution,
            )
        if settings.min_bucket_reso <= 0:
            error("min_bucket_invalid", "min_bucket_resoは正の整数で指定してください。")
        if settings.max_bucket_reso < settings.min_bucket_reso:
            error(
                "bucket_range_invalid",
                "max_bucket_resoはmin_bucket_reso以上にしてください。",
            )
        if settings.bucket_reso_steps <= 0:
            error(
                "bucket_steps_invalid",
                "bucket_reso_stepsは正の整数で指定してください。",
            )
        if settings.enable_bucket and settings.resolution % settings.bucket_reso_steps:
            warning(
                "bucket_resolution_alignment",
                "resolutionがbucket_reso_stepsで割り切れません。",
            )
        if settings.num_repeats < 1:
            error("repeats_invalid", "num_repeatsは1以上にしてください。")
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,8}", settings.caption_extension):
            error(
                "caption_extension_invalid",
                "caption_extensionは安全な拡張子を指定してください。",
            )
        if any(ord(char) < 32 for char in settings.caption_separator):
            error(
                "caption_separator_invalid",
                "caption_separatorに制御文字は使用できません。",
            )
        if settings.keep_tokens < 0:
            error("keep_tokens_invalid", "keep_tokensは0以上にしてください。")
        if image_count <= 0:
            error("empty_subset", "学習対象画像が0件です。")
        if settings.num_repeats >= 1000:
            warning(
                "repeats_high",
                "num_repeatsが極端に大きく、実質画像数が増えます。",
                settings.num_repeats,
            )
        return tuple(issues)

    def to_toml(self, settings: DatasetSettings, image_dir: str = "images") -> str:
        if Path(image_dir).is_absolute() or ".." in Path(image_dir).parts:
            raise ValueError("image_dir must remain inside the snapshot root")
        values = {
            "resolution": settings.resolution,
            "enable_bucket": settings.enable_bucket,
            "min_bucket_reso": settings.min_bucket_reso,
            "max_bucket_reso": settings.max_bucket_reso,
            "bucket_reso_steps": settings.bucket_reso_steps,
            "bucket_no_upscale": settings.bucket_no_upscale,
            "caption_extension": settings.caption_extension,
            "shuffle_caption": settings.shuffle_caption,
            "keep_tokens": settings.keep_tokens,
            "caption_separator": settings.caption_separator,
            "flip_aug": settings.flip_aug,
            "color_aug": settings.color_aug,
            "random_crop": settings.random_crop,
            "debug_dataset": settings.debug_dataset,
        }
        general = "\n".join(
            f"{key} = {self._toml_value(value)}" for key, value in values.items()
        )
        subset = {
            "image_dir": image_dir,
            "num_repeats": settings.num_repeats,
            "caption_extension": settings.caption_extension,
            "class_tokens": settings.class_tokens,
            "is_reg": settings.is_reg,
            "keep_tokens": settings.keep_tokens,
            "shuffle_caption": settings.shuffle_caption,
        }
        subset_text = "\n".join(
            f"{key} = {self._toml_value(value)}" for key, value in subset.items()
        )
        text = f"[general]\n{general}\n\n[[datasets]]\n\n[[datasets.subsets]]\n{subset_text}\n"
        tomllib.loads(text)
        return text

    @staticmethod
    def settings_snapshot(settings: DatasetSettings) -> str:
        return json.dumps(asdict(settings), sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _toml_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (int, float)):
            return str(value)
        raise ValueError("unsupported TOML value")
