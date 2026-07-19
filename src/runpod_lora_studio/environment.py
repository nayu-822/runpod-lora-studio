from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CommandStatus:
    name: str
    available: bool
    resolved_path: str | None


@dataclass(slots=True)
class EnvironmentReport:
    python_version: str
    platform: str
    runpod_pod_id: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    torch_cuda_available: bool
    gpu_name: str | None
    gpu_count: int
    gpu_total_vram_mb: int | None
    bf16_supported: bool | None
    workspace_exists: bool
    workspace_writable: bool
    runtime_dir: str
    runtime_dir_parent_writable: bool
    commands: list[CommandStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["commands"] = [asdict(command) for command in self.commands]
        return payload


def _is_writable_path(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    if not path.parent.exists():
        return False
    return os.access(path.parent, os.W_OK)


def _command_status(command_name: str) -> CommandStatus:
    resolved = shutil.which(command_name)
    return CommandStatus(
        name=command_name,
        available=resolved is not None,
        resolved_path=resolved,
    )


def _query_nvidia_smi() -> tuple[str | None, int | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None

    first_line = result.stdout.strip().splitlines()[0]
    name, _, memory = first_line.partition(",")
    gpu_name = name.strip() or None
    total_vram_mb = int(memory.strip()) if memory.strip().isdigit() else None
    return gpu_name, total_vram_mb


def collect_environment_report() -> EnvironmentReport:
    workspace = Path("/workspace")
    runtime_dir = Path("/workspace/ldts-runtime")
    commands = [_command_status(name) for name in ("git", "rclone", "nvidia-smi")]

    torch_version: str | None = None
    torch_cuda_version: str | None = None
    torch_cuda_available = False
    gpu_name: str | None = None
    gpu_count = 0
    gpu_total_vram_mb: int | None = None
    bf16_supported: bool | None = None
    warnings: list[str] = []
    errors: list[str] = []

    try:
        import torch
    except ImportError:
        warnings.append("PyTorch が未インストールです。")
    else:
        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda
        torch_cuda_available = torch.cuda.is_available()
        if torch_cuda_available:
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                del free_mem
                gpu_total_vram_mb = int(total_mem / (1024 * 1024))
            if hasattr(torch.cuda, "is_bf16_supported"):
                bf16_supported = bool(torch.cuda.is_bf16_supported())
        else:
            warnings.append("CUDA が利用できません。ローカル CPU 環境として扱います。")

    if gpu_name is None:
        fallback_name, fallback_vram = _query_nvidia_smi()
        gpu_name = fallback_name
        gpu_total_vram_mb = gpu_total_vram_mb or fallback_vram

    if os.getenv("RUNPOD_POD_ID") is None:
        warnings.append("RUNPOD_POD_ID が未設定です。RunPod 外の実行とみなします。")

    if not workspace.exists():
        warnings.append("/workspace が存在しません。ローカル環境では想定内です。")

    if not _is_writable_path(runtime_dir):
        warnings.append(
            "ランタイムディレクトリの親に書き込みできません。"
            " RunPod 本番環境で再確認してください。"
        )

    for command in commands:
        if not command.available:
            warnings.append(f"コマンドが見つかりません: {command.name}")

    return EnvironmentReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        runpod_pod_id=os.getenv("RUNPOD_POD_ID"),
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        torch_cuda_available=torch_cuda_available,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_total_vram_mb=gpu_total_vram_mb,
        bf16_supported=bf16_supported,
        workspace_exists=workspace.exists(),
        workspace_writable=_is_writable_path(workspace),
        runtime_dir=str(runtime_dir),
        runtime_dir_parent_writable=_is_writable_path(runtime_dir),
        commands=commands,
        warnings=warnings,
        errors=errors,
    )
