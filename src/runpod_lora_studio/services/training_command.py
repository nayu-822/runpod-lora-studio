from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from runpod_lora_studio.domain.training_models import TrainingConfig


class TrainingCommandValidationError(ValueError):
    """Raised when a training command cannot be safely constructed."""


@dataclass(frozen=True, slots=True)
class TrainingCommand:
    arguments: tuple[str, ...]
    summary: str


class SdScriptsCommandBuilder:
    allowed_trainer_scripts: ClassVar[frozenset[str]] = frozenset(
        {"sdxl_train_network.py"}
    )
    allowed_mixed_precision: ClassVar[frozenset[str]] = frozenset(
        {"no", "fp16", "bf16"}
    )
    extra_option_types: ClassVar[dict[str, type]] = {
        "max_train_steps": int,
        "max_token_length": int,
        "clip_skip": int,
        "network_dropout": float,
        "caption_dropout_rate": float,
        "enable_bucket": bool,
        "min_bucket_reso": int,
        "max_bucket_reso": int,
        "bucket_reso_steps": int,
        "no_token_padding": bool,
    }

    def build(
        self,
        config: TrainingConfig,
        *,
        model_path: Path,
        dataset_config_path: Path,
    ) -> TrainingCommand:
        self._validate_text(config.python_executable, "python executable")
        self._validate_text(config.trainer_script, "trainer script")
        if config.trainer_script not in self.allowed_trainer_scripts:
            raise TrainingCommandValidationError("許可されていないtrainer scriptです")
        if config.mixed_precision not in self.allowed_mixed_precision:
            raise TrainingCommandValidationError("mixed precisionが不正です")
        if not model_path.is_file() or model_path.stat().st_size <= 0:
            raise TrainingCommandValidationError("学習元モデルが利用できません")
        if not dataset_config_path.is_file():
            raise TrainingCommandValidationError("dataset TOMLが利用できません")
        root = config.sd_scripts_root.resolve()
        trainer_path = (root / config.trainer_script).resolve()
        self._ensure_under(trainer_path, root, "trainer script")
        if not trainer_path.is_file():
            raise TrainingCommandValidationError("trainer scriptが見つかりません")
        output_directory = config.output_directory.resolve()
        if not output_directory.is_dir():
            output_directory.mkdir(parents=True, exist_ok=True)

        arguments: list[str] = [
            config.python_executable,
            str(trainer_path),
            "--pretrained_model_name_or_path",
            str(model_path.resolve()),
            "--dataset_config",
            str(dataset_config_path.resolve()),
            "--output_dir",
            str(output_directory),
            "--output_name",
            config.output_name,
            "--resolution",
            str(config.resolution),
            "--train_batch_size",
            str(config.batch_size),
            "--max_train_epochs",
            str(config.epochs),
            "--learning_rate",
            str(config.learning_rate),
            "--optimizer_type",
            config.optimizer,
            "--lr_scheduler",
            config.scheduler,
            "--network_module",
            config.network_module,
            "--network_dim",
            str(config.network_dim),
            "--network_alpha",
            str(config.network_alpha),
            "--mixed_precision",
            config.mixed_precision,
            "--seed",
            str(config.seed),
            "--save_every_n_epochs",
            str(config.save_every_n_epochs),
        ]
        if config.cache_latents:
            arguments.append("--cache_latents")
        if config.gradient_checkpointing:
            arguments.append("--gradient_checkpointing")
        arguments.extend(self._extra_arguments(config.extra_options))
        return TrainingCommand(tuple(arguments), self._summary(arguments))

    def _extra_arguments(self, options: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for name in sorted(options):
            if name not in self.extra_option_types:
                raise TrainingCommandValidationError(f"未知のextra optionです: {name}")
            value = options[name]
            expected = self.extra_option_types[name]
            if expected is int and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise TrainingCommandValidationError(
                    f"extra optionの型が不正です: {name}"
                )
            if expected is float and (
                not isinstance(value, (float, int)) or isinstance(value, bool)
            ):
                raise TrainingCommandValidationError(
                    f"extra optionの型が不正です: {name}"
                )
            if expected is bool and not isinstance(value, bool):
                raise TrainingCommandValidationError(
                    f"extra optionの型が不正です: {name}"
                )
            if expected in (int, float) and value <= 0:
                raise TrainingCommandValidationError(
                    f"extra optionの値が不正です: {name}"
                )
            if expected is bool and not value:
                continue
            result.append(f"--{name}")
            if expected is not bool:
                result.append(str(value))
        return result

    @staticmethod
    def _validate_text(value: str, label: str) -> None:
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise TrainingCommandValidationError(f"{label}が不正です")

    @staticmethod
    def _ensure_under(path: Path, root: Path, label: str) -> None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TrainingCommandValidationError(f"{label}が許可領域外です") from exc

    @staticmethod
    def _summary(arguments: list[str]) -> str:
        redacted: list[str] = []
        redact_next = False
        for argument in arguments:
            if redact_next:
                redacted.append("<path>")
                redact_next = False
                continue
            if argument in {
                "--pretrained_model_name_or_path",
                "--dataset_config",
            }:
                redacted.append(argument)
                redact_next = True
                continue
            redacted.append(argument)
        return " ".join(shlex.quote(value) for value in redacted)
