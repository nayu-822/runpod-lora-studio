from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from collections.abc import Sequence
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
    version = "phase6c-v1"
    allowed_trainer_scripts: ClassVar[frozenset[str]] = frozenset(
        {"sdxl_train_network.py"}
    )
    allowed_mixed_precision: ClassVar[frozenset[str]] = frozenset(
        {"no", "fp16", "bf16"}
    )
    allowed_network_modules: ClassVar[frozenset[str]] = frozenset({"networks.lora"})
    allowed_optimizers: ClassVar[frozenset[str]] = frozenset(
        {"AdamW", "AdamW8bit", "Lion", "Prodigy"}
    )
    allowed_schedulers: ClassVar[frozenset[str]] = frozenset(
        {"constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"}
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

    def __init__(
        self,
        *,
        trusted_trainer_root: Path,
        python_executable: Path | str | None = None,
        trusted_python_executables: Sequence[Path] | None = None,
    ) -> None:
        self.trusted_trainer_root = trusted_trainer_root.resolve()
        self.python_executable = Path(python_executable or sys.executable)
        trusted = {Path(sys.executable).resolve()}
        for executable in trusted_python_executables or ():
            if executable.is_absolute():
                trusted.add(executable.resolve())
        self.trusted_python_executables = frozenset(trusted)

    def build(
        self,
        config: TrainingConfig,
        *,
        model_path: Path,
        dataset_config_path: Path,
        allowed_model_roots: Sequence[Path],
        allowed_dataset_roots: Sequence[Path],
        allowed_output_roots: Sequence[Path],
        resume_path: Path | None = None,
        allowed_resume_roots: Sequence[Path] = (),
    ) -> TrainingCommand:
        python_executable = self.validate_python_executable()
        self._validate_text(config.trainer_script, "trainer script")
        if config.trainer_script not in self.allowed_trainer_scripts:
            raise TrainingCommandValidationError("trainer script is not allowed")
        if config.network_module not in self.allowed_network_modules:
            raise TrainingCommandValidationError("network module is not allowed")
        if config.optimizer not in self.allowed_optimizers:
            raise TrainingCommandValidationError("optimizer is not allowed")
        if config.scheduler not in self.allowed_schedulers:
            raise TrainingCommandValidationError("scheduler is not allowed")
        if config.mixed_precision not in self.allowed_mixed_precision:
            raise TrainingCommandValidationError("mixed precision is not allowed")
        model_path = self._validated_file(model_path, allowed_model_roots, "model")
        dataset_config_path = self._validated_file(
            dataset_config_path, allowed_dataset_roots, "dataset TOML"
        )
        root = config.sd_scripts_root.resolve()
        if root != self.trusted_trainer_root:
            raise TrainingCommandValidationError(
                "sd-scripts root does not match the configured root"
            )
        trainer_path = (root / config.trainer_script).resolve()
        self._ensure_under_any(
            trainer_path, (self.trusted_trainer_root,), "trainer script"
        )
        if not trainer_path.is_file():
            raise TrainingCommandValidationError("trainer script does not exist")
        output_directory = config.output_directory.resolve()
        self._ensure_under_any(
            output_directory, allowed_output_roots, "output directory"
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        arguments: list[str] = [
            str(python_executable),
            str(trainer_path),
            "--pretrained_model_name_or_path",
            str(model_path),
            "--dataset_config",
            str(dataset_config_path),
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
        if resume_path is not None:
            if resume_path.is_symlink():
                raise TrainingCommandValidationError(
                    "resume state symlink is not allowed"
                )
            resume_path = resume_path.resolve()
            self._ensure_under_any(resume_path, allowed_resume_roots, "resume state")
            if not resume_path.is_dir():
                raise TrainingCommandValidationError("resume state is not a directory")
            arguments.extend(("--resume", str(resume_path)))
        arguments.extend(self._extra_arguments(config.extra_options))
        return TrainingCommand(tuple(arguments), self._summary(arguments))

    def validate_python_executable(self) -> Path:
        raw = str(self.python_executable)
        self._validate_text(raw, "python executable")
        candidate = self.python_executable
        if not candidate.is_absolute():
            resolved_from_path = shutil.which(str(candidate))
            if resolved_from_path is None:
                raise TrainingCommandValidationError("python executable does not exist")
            candidate = Path(resolved_from_path)
        try:
            executable = candidate.resolve(strict=True)
        except OSError as exc:
            raise TrainingCommandValidationError(
                "python executable does not exist"
            ) from exc
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise TrainingCommandValidationError("python executable is not executable")
        if not re.fullmatch(
            r"python(?:3(?:\.\d+)*)?(?:\.exe)?", executable.name, re.IGNORECASE
        ):
            raise TrainingCommandValidationError(
                "only an allowed Python executable may be used"
            )
        if executable not in self.trusted_python_executables:
            raise TrainingCommandValidationError(
                "python executable is not a trusted executable"
            )
        return executable

    def _extra_arguments(self, options: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for name in sorted(options):
            if name not in self.extra_option_types:
                raise TrainingCommandValidationError(f"unknown extra option: {name}")
            value = options[name]
            expected = self.extra_option_types[name]
            if expected is int and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise TrainingCommandValidationError(f"invalid extra option: {name}")
            if expected is float and (
                not isinstance(value, (float, int)) or isinstance(value, bool)
            ):
                raise TrainingCommandValidationError(f"invalid extra option: {name}")
            if expected is bool and not isinstance(value, bool):
                raise TrainingCommandValidationError(f"invalid extra option: {name}")
            if expected in (int, float) and value <= 0:
                raise TrainingCommandValidationError(f"invalid extra option: {name}")
            if expected is bool and not value:
                continue
            result.append(f"--{name}")
            if expected is not bool:
                result.append(str(value))
        return result

    @staticmethod
    def _validate_text(value: str, label: str) -> None:
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise TrainingCommandValidationError(f"invalid {label}")

    @classmethod
    def _validated_file(cls, path: Path, roots: Sequence[Path], label: str) -> Path:
        resolved = path.resolve()
        cls._ensure_under_any(resolved, roots, label)
        if not resolved.is_file():
            raise TrainingCommandValidationError(f"{label} does not exist")
        if label == "model" and resolved.stat().st_size <= 0:
            raise TrainingCommandValidationError("model is empty")
        return resolved

    @staticmethod
    def _ensure_under_any(path: Path, roots: Sequence[Path], label: str) -> None:
        if not any(_is_under(path, root.resolve()) for root in roots):
            raise TrainingCommandValidationError(f"{label} is outside an allowed root")

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
            if argument == "--resume":
                redacted.append(argument)
                redact_next = True
                continue
            redacted.append(argument)
        return " ".join(shlex.quote(value) for value in redacted)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
