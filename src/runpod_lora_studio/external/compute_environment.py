from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Protocol

from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    GPUDeviceInfo,
    PhysicalGpuInfo,
)


class ComputeEnvironmentAdapter(Protocol):
    def detect(self) -> ComputeEnvironmentInfo: ...


class PhysicalGpuInventoryAdapter(Protocol):
    def detect(self) -> tuple[PhysicalGpuInfo, ...]: ...


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
                    free, total = torch.cuda.mem_get_info(index)
                    if (
                        isinstance(free, bool)
                        or isinstance(total, bool)
                        or int(free) < 0
                        or int(total) <= 0
                        or int(free) > int(total)
                    ):
                        warnings.append(f"GPU {index} returned invalid VRAM values")
                        continue
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


class NvidiaSmiGpuInventoryAdapter:
    """Read a bounded, fixed-format physical GPU inventory."""

    query = "--query-gpu=index,uuid,name,memory.total,compute_cap"
    format_option = "--format=csv,noheader,nounits"

    def __init__(
        self,
        *,
        executable: str = "nvidia-smi",
        timeout_seconds: float = 5.0,
        max_output_bytes: int = 32 * 1024,
    ) -> None:
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def detect(self) -> tuple[PhysicalGpuInfo, ...]:
        executable = shutil.which(self.executable) or self.executable
        try:
            result = subprocess.run(
                [executable, self.query, self.format_option],
                capture_output=True,
                check=False,
                shell=False,
                text=False,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError):
            return ()
        if result.returncode != 0:
            return ()
        output = result.stdout[: self.max_output_bytes].decode(
            "utf-8", errors="replace"
        )
        rows: list[PhysicalGpuInfo] = []
        indexes: set[int] = set()
        uuids: set[str] = set()
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 5:
                continue
            index_text, uuid, name, total_text, capability = fields
            try:
                index = int(index_text)
                total_mib = int(total_text)
            except ValueError:
                continue
            normalized_uuid = uuid.strip().lower()
            if (
                index < 0
                or not normalized_uuid
                or not name
                or total_mib <= 0
                or index in indexes
                or normalized_uuid in uuids
            ):
                return ()
            indexes.add(index)
            uuids.add(normalized_uuid)
            rows.append(
                PhysicalGpuInfo(
                    index=index,
                    uuid=uuid,
                    name=name,
                    architecture=name,
                    compute_capability=capability or None,
                    total_vram_bytes=total_mib * 1024**2,
                )
            )
        return tuple(rows)


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
