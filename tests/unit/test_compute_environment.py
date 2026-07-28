from __future__ import annotations

import sys
from types import SimpleNamespace

from runpod_lora_studio.domain.recommendation_models import (
    ComputeEnvironmentInfo,
    GPUDeviceInfo,
    PhysicalGpuInfo,
)
from runpod_lora_studio.external.compute_environment import (
    NvidiaSmiGpuInventoryAdapter,
    TorchComputeEnvironmentAdapter,
)
from runpod_lora_studio.services.training_job_environment_service import _map_devices


def _physical_inventory() -> tuple[PhysicalGpuInfo, ...]:
    return (
        PhysicalGpuInfo(
            index=2,
            uuid="GPU-b",
            name="B",
            architecture="Arch-B",
            compute_capability="8.0",
            total_vram_bytes=20 * 1024**3,
        ),
        PhysicalGpuInfo(
            index=0,
            uuid="GPU-a",
            name="A",
            architecture="Arch-A",
            compute_capability="7.5",
            total_vram_bytes=16 * 1024**3,
        ),
    )


def test_torch_mem_get_info_stores_free_then_total(monkeypatch) -> None:
    properties = SimpleNamespace(uuid="GPU-b", name="Arch-B", major=8, minor=0)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda index: properties,
        get_device_name=lambda index: "B",
        mem_get_info=lambda index: (3 * 1024**3, 20 * 1024**3),
        is_bf16_supported=lambda: True,
    )
    fake_torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.4"),
        __version__="2.7.0",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "runpod_lora_studio.external.compute_environment._query_driver_version",
        lambda: "555.0",
    )

    info = TorchComputeEnvironmentAdapter().detect()

    assert info.gpu_devices[0].free_vram_bytes == 3 * 1024**3
    assert info.gpu_devices[0].total_vram_bytes == 20 * 1024**3


def test_torch_invalid_vram_values_are_skipped_with_warning(monkeypatch) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(
            uuid=f"GPU-{index}", name=f"Arch-{index}", major=8, minor=0
        ),
        get_device_name=lambda index: f"GPU {index}",
        mem_get_info=lambda index: (
            (-1, 20 * 1024**3) if index == 0 else (21 * 1024**3, 20 * 1024**3)
        ),
        is_bf16_supported=lambda: True,
    )
    fake_torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.4"),
        __version__="2.7.0",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "runpod_lora_studio.external.compute_environment._query_driver_version",
        lambda: "555.0",
    )

    info = TorchComputeEnvironmentAdapter().detect()

    assert info.gpu_devices == ()
    assert any("invalid VRAM" in warning for warning in info.warnings)


def test_visible_devices_map_logical_order_to_physical_order() -> None:
    info = ComputeEnvironmentInfo(
        gpu_devices=(
            GPUDeviceInfo(index=0, name="B", uuid="GPU-b"),
            GPUDeviceInfo(index=1, name="A", uuid="GPU-a"),
        ),
        cuda_available=True,
    )

    mappings = _map_devices(info, "2,0", _physical_inventory())

    assert [(item.logical_index, item.physical_index) for item in mappings] == [
        (0, 2),
        (1, 0),
    ]
    assert [item.gpu_uuid for item in mappings] == ["GPU-b", "GPU-a"]


def test_unset_visible_devices_matches_torch_uuid_without_assuming_index() -> None:
    info = ComputeEnvironmentInfo(
        gpu_devices=(
            GPUDeviceInfo(index=0, name="B", uuid="GPU-b"),
            GPUDeviceInfo(index=1, name="A", uuid="GPU-a"),
        ),
        cuda_available=True,
    )

    mappings = _map_devices(info, "", _physical_inventory())

    assert [(item.logical_index, item.physical_index) for item in mappings] == [
        (0, 2),
        (1, 0),
    ]


def test_uuid_prefix_is_resolved_and_invalid_tokens_are_unverified() -> None:
    info = ComputeEnvironmentInfo(
        gpu_devices=(GPUDeviceInfo(index=0, name="B", uuid="GPU-B123"),),
        cuda_available=True,
    )
    inventory = (
        PhysicalGpuInfo(index=2, uuid="GPU-b123", name="B"),
        PhysicalGpuInfo(index=0, uuid="GPU-a456", name="A"),
    )

    prefix = _map_devices(info, "GPU-B1", inventory)[0]
    invalid = _map_devices(info, "GPU-dead", inventory)[0]

    assert prefix.physical_index == 2
    assert prefix.identity_verified
    assert invalid.physical_index is None
    assert invalid.gpu_uuid_fingerprint is None
    assert "GPU_UUID_NOT_FOUND" in invalid.warning_codes


def test_nvidia_inventory_adapter_uses_fixed_bounded_query(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(b"2, GPU-b, B, 20480, 8.0\n0, GPU-a, A, 16384, 7.5\ninvalid\n"),
        )

    monkeypatch.setattr(
        "runpod_lora_studio.external.compute_environment.subprocess.run", fake_run
    )
    inventory = NvidiaSmiGpuInventoryAdapter().detect()

    assert [item.index for item in inventory] == [2, 0]
    assert inventory[0].total_vram_bytes == 20480 * 1024**2
    assert calls[0][0][1:] == [
        "--query-gpu=index,uuid,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    assert calls[0][1]["shell"] is False


def test_nvidia_inventory_duplicate_identity_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "runpod_lora_studio.external.compute_environment.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"0, GPU-a, A, 16000, 8.0\n0, GPU-b, B, 16000, 8.0\n",
        ),
    )

    assert NvidiaSmiGpuInventoryAdapter().detect() == ()
