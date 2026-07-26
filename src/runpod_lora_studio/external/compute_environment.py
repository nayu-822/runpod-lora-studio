from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Protocol

from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    GPUDeviceInfo,
)


class ComputeEnvironmentAdapter(Protocol):
    def detect(self) -> ComputeEnvironmentInfo: ...


class TorchComputeEnvironmentAdapter:
    """Collect CUDA information without importing torch at module import time."""

    def detect(self) -> ComputeEnvironmentInfo:
        warnings: list[str] = []
        errors: list[str] = []
        try:
            import torch
        except ImportError:
            return ComputeEnvironmentInfo(
                operating_system=platform.platform(),
                python_version=sys.version.split()[0],
                warnings=("torch is not installed",),
            )
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            cuda_available = False
            warnings.append(f"CUDA detection failed: {type(exc).__name__}")
        devices: list[GPUDeviceInfo] = []
        if cuda_available:
            try:
                for index in range(int(torch.cuda.device_count())):
                    properties = torch.cuda.get_device_properties(index)
                    total, free = torch.cuda.mem_get_info(index)
                    major = getattr(properties, "major", None)
                    minor = getattr(properties, "minor", None)
                    devices.append(
                        GPUDeviceInfo(
                            index=index,
                            name=str(torch.cuda.get_device_name(index)),
                            uuid=_optional_str(getattr(properties, "uuid", None)),
                            architecture=_optional_str(
                                getattr(properties, "name", None)
                            ),
                            compute_capability=(
                                f"{major}.{minor}"
                                if major is not None and minor is not None
                                else None
                            ),
                            total_vram_bytes=int(total),
                            free_vram_bytes=int(free),
                        )
                    )
            except Exception as exc:
                warnings.append(f"GPU detail detection failed: {type(exc).__name__}")
        bf16_supported: bool | None = None
        if cuda_available:
            try:
                checker = getattr(torch.cuda, "is_bf16_supported", None)
                bf16_supported = bool(checker()) if checker else None
            except Exception as exc:
                warnings.append(f"bf16 detection failed: {type(exc).__name__}")
        driver = _query_driver_version()
        if driver is None:
            warnings.append("CUDA driver version is unavailable")
        return ComputeEnvironmentInfo(
            gpu_devices=tuple(devices),
            cuda_available=cuda_available,
            cuda_runtime_version=_optional_str(getattr(torch.version, "cuda", None)),
            cuda_driver_version=driver,
            torch_version=_optional_str(getattr(torch, "__version__", None)),
            torch_cuda_version=_optional_str(getattr(torch.version, "cuda", None)),
            bf16_supported=bf16_supported,
            fp16_supported=cuda_available,
            xformers_available=importlib.util.find_spec("xformers") is not None,
            bitsandbytes_available=importlib.util.find_spec("bitsandbytes") is not None,
            operating_system=platform.platform(),
            python_version=sys.version.split()[0],
            warnings=tuple(warnings),
            errors=tuple(errors),
        )


@dataclass(frozen=True, slots=True)
class FakeComputeEnvironmentAdapter:
    info: ComputeEnvironmentInfo

    def detect(self) -> ComputeEnvironmentInfo:
        return self.info


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _query_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().splitlines()
    return value[0].strip() if value and value[0].strip() else None
