from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from runpod_lora_studio.config.settings import AppSettings, get_settings


@dataclass(slots=True)
class CommandStatus:
    name: str
    available: bool
    required: bool
    resolved_path: str | None


@dataclass(slots=True)
class GPUInfo:
    index: int
    name: str
    vram_mb: int | None


@dataclass(slots=True)
class EnvironmentReport:
    python_version: str
    python_supported: bool
    platform: str
    is_runpod: bool
    runpod_pod_id: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    torch_cuda_available: bool
    bf16_supported: bool | None = None
    gpus: list[GPUInfo] = field(default_factory=list)
    disk_free_bytes: int | None = None
    disk_total_bytes: int | None = None
    workspace_exists: bool = False
    workspace_writable: bool = False
    runtime_dir: str = ""
    runtime_dir_parent_writable: bool = False
    commands: list[CommandStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def gpu_name(self) -> str | None:
        return self.gpus[0].name if self.gpus else None

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def gpu_total_vram_mb(self) -> int | None:
        values = [gpu.vram_mb for gpu in self.gpus if gpu.vram_mb is not None]
        return sum(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gpus"] = [asdict(gpu) for gpu in self.gpus]
        payload["gpu_name"] = self.gpu_name
        payload["gpu_count"] = self.gpu_count
        payload["gpu_total_vram_mb"] = self.gpu_total_vram_mb
        payload["bf16_supported"] = self.bf16_supported
        return payload


def _is_writable_path(path: Path) -> bool:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.exists() and os.access(candidate, os.W_OK)


def _query_bf16_support(torch_module: Any) -> bool | None:
    try:
        if not torch_module.cuda.is_available():
            return None
        checker = getattr(torch_module.cuda, "is_bf16_supported", None)
        if checker is None:
            return None
        return bool(checker())
    except Exception:
        # Driver-specific failures are reported as unknown, not as environment failures.
        return None


def _command_status(command_name: str, *, required: bool) -> CommandStatus:
    resolved = shutil.which(command_name)
    return CommandStatus(command_name, resolved is not None, required, resolved)


def _query_nvidia_smi() -> list[GPUInfo]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return []

    gpus: list[GPUInfo] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        vram_mb = int(parts[2]) if parts[2].isdigit() else None
        if parts[1]:
            gpus.append(GPUInfo(index=index, name=parts[1], vram_mb=vram_mb))
    return gpus


def _query_torch_gpus(torch_module: Any) -> tuple[list[GPUInfo], bool]:
    if not torch_module.cuda.is_available():
        return [], False
    gpus: list[GPUInfo] = []
    for index in range(torch_module.cuda.device_count()):
        total_memory = int(torch_module.cuda.mem_get_info(index)[1] / (1024 * 1024))
        gpus.append(
            GPUInfo(
                index=index,
                name=str(torch_module.cuda.get_device_name(index)),
                vram_mb=total_memory,
            )
        )
    return gpus, True


def collect_environment_report(
    settings: AppSettings | None = None,
    writable_check: Callable[[Path], bool] | None = None,
) -> EnvironmentReport:
    runtime_settings = settings or get_settings()
    workspace = runtime_settings.workspace_root
    commands = [
        _command_status("git", required=True),
        _command_status("rclone", required=False),
        _command_status("nvidia-smi", required=False),
    ]
    warnings: list[str] = []
    errors: list[str] = []
    python_supported = sys.version_info >= (3, 11)
    if not python_supported:
        errors.append("Python 3.11 以上が必要です。")

    torch_version: str | None = None
    torch_cuda_version: str | None = None
    torch_cuda_available = False
    bf16_supported: bool | None = None
    gpus: list[GPUInfo] = []
    try:
        import torch
    except ImportError:
        warnings.append("PyTorch が未インストールです。CPU環境として扱います。")
    else:
        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda
        gpus, torch_cuda_available = _query_torch_gpus(torch)
        bf16_supported = _query_bf16_support(torch)
        if not torch_cuda_available:
            warnings.append("CUDA が利用できません。CPU環境として扱います。")

    if not gpus:
        gpus = _query_nvidia_smi()
        torch_cuda_available = bool(gpus) and torch_cuda_available

    if not runtime_settings.runpod_pod_id:
        warnings.append("RUNPOD_POD_ID が未設定です。RunPod 外の実行とみなします。")
    is_runpod = bool(runtime_settings.runpod_pod_id)

    check_writable = writable_check or _is_writable_path
    workspace_exists = workspace.exists()
    workspace_writable = check_writable(workspace)
    if not workspace_exists:
        warnings.append(f"作業ディレクトリが存在しません: {workspace}")
    if not workspace_writable:
        errors.append(f"作業ディレクトリへ書き込みできません: {workspace}")

    try:
        disk = shutil.disk_usage(workspace if workspace.exists() else workspace.parent)
        disk_free_bytes = disk.free
        disk_total_bytes = disk.total
    except OSError:
        disk_free_bytes = None
        disk_total_bytes = None
        warnings.append("作業ディレクトリのディスク容量を取得できません。")

    for command in commands:
        if not command.available:
            message = f"コマンドが見つかりません: {command.name}"
            (errors if command.required else warnings).append(message)

    return EnvironmentReport(
        python_version=sys.version.split()[0],
        python_supported=python_supported,
        platform=platform.platform(),
        is_runpod=is_runpod,
        runpod_pod_id=runtime_settings.runpod_pod_id,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        torch_cuda_available=torch_cuda_available,
        bf16_supported=bf16_supported,
        gpus=gpus,
        disk_free_bytes=disk_free_bytes,
        disk_total_bytes=disk_total_bytes,
        workspace_exists=workspace_exists,
        workspace_writable=workspace_writable,
        runtime_dir=str(runtime_settings.workspace_root),
        runtime_dir_parent_writable=workspace_writable,
        commands=commands,
        warnings=warnings,
        errors=errors,
    )
