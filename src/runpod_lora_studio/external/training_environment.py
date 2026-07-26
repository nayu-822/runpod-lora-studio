from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.recommendation_models import TrainingEnvironmentInfo


class TrainingEnvironmentAdapter:
    def detect(self) -> TrainingEnvironmentInfo:
        raise NotImplementedError


class SdScriptsTrainingEnvironmentAdapter(TrainingEnvironmentAdapter):
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def detect(self) -> TrainingEnvironmentInfo:
        root = self.settings.training_sd_scripts_root.resolve()
        python = self.settings.training_python_executable.resolve()
        trainer = root / "sdxl_train_network.py"
        warnings: list[str] = []
        errors: list[str] = []
        if not root.is_dir():
            errors.append("sd-scripts root is missing")
        if not trainer.is_file():
            errors.append("sdxl_train_network.py is missing")
        if not python.is_file():
            errors.append("trusted Python executable is missing")
        version = _read_sd_scripts_version(root)
        if version is None:
            warnings.append("sd-scripts version is unavailable")
        safetensors = importlib.util.find_spec("safetensors") is not None
        torch = importlib.util.find_spec("torch") is not None
        xformers = importlib.util.find_spec("xformers") is not None
        bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None
        if not safetensors:
            errors.append("safetensors is unavailable")
        if not torch:
            errors.append("torch is unavailable")
        return TrainingEnvironmentInfo(
            sd_scripts_root=root,
            trainer_script=trainer if trainer.is_file() else None,
            sd_scripts_version=version,
            python_executable=python,
            safetensors_available=safetensors,
            torch_available=torch,
            xformers_available=xformers,
            bitsandbytes_available=bitsandbytes,
            bf16_supported=None,
            fp16_supported=torch,
            cuda_available=False,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )


class FakeTrainingEnvironmentAdapter(TrainingEnvironmentAdapter):
    def __init__(self, info: TrainingEnvironmentInfo) -> None:
        self.info = info

    def detect(self) -> TrainingEnvironmentInfo:
        return self.info


def _read_sd_scripts_version(root: Path) -> str | None:
    for name in ("VERSION", "version.txt", "pyproject.toml"):
        path = root / name
        try:
            if not path.is_file() or path.stat().st_size > 64 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if name == "pyproject.toml":
            match = re.search(r"^version\s*=\s*[\"']([^\"']+)", text, re.M)
        else:
            match = re.search(r"[0-9]+(?:\.[0-9]+)+(?:[-+][A-Za-z0-9.-]+)?", text)
        if match:
            return match.group(1)
    return None
