from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from runpod_lora_studio.app import build_status_markdown, create_app
from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.environment import EnvironmentReport, GPUInfo
from runpod_lora_studio.ui.training import (
    clear_recommendation_state,
    mark_recommendation_edited,
)


def _cpu_report(test_workspace: Path) -> EnvironmentReport:
    return EnvironmentReport(
        python_version="3.11.0",
        python_supported=True,
        platform="test",
        is_runpod=False,
        runpod_pod_id=None,
        torch_version=None,
        torch_cuda_version=None,
        torch_cuda_available=False,
        disk_free_bytes=10 * 1024**3,
        disk_total_bytes=20 * 1024**3,
        runtime_dir=str(test_workspace),
    )


def test_build_status_markdown_contains_required_values(test_workspace: Path) -> None:
    settings = AppSettings(
        app_title="Test Studio", app_env="test", workspace_root=test_workspace
    )
    report = _cpu_report(test_workspace)

    markdown = build_status_markdown(settings, report)

    assert "正常稼働中" in markdown
    assert "アプリケーションバージョン" in markdown
    assert "Pythonバージョン: `3.11.0`" in markdown
    assert "実行環境: `ローカル`" in markdown
    assert "GPU認識状況: `未認識`" in markdown
    assert "データ保存先" in markdown


def test_build_status_markdown_contains_multiple_gpu_summary(
    test_workspace: Path,
) -> None:
    settings = AppSettings(workspace_root=test_workspace)
    report = _cpu_report(test_workspace)
    report.gpus = [GPUInfo(0, "GPU A", 8192), GPUInfo(1, "GPU B", 12288)]

    markdown = build_status_markdown(settings, report)

    assert "GPU A" in markdown
    assert "GPU B" in markdown


def test_build_status_markdown_requires_attention_for_environment_error(
    test_workspace: Path,
) -> None:
    settings = AppSettings(workspace_root=test_workspace)
    report = _cpu_report(test_workspace)
    report.errors.append("作業ディレクトリへ書き込みできません")

    markdown = build_status_markdown(settings, report)

    assert "状態: **要確認**" in markdown


def test_create_app_accepts_cpu_report_without_real_gpu(test_workspace: Path) -> None:
    settings = AppSettings(workspace_root=test_workspace)

    app = create_app(settings, _cpu_report(test_workspace))

    assert app.title == settings.app_title


def test_import_does_not_start_gradio_server() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import runpod_lora_studio.app"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_recommendation_state_clear_switches_to_manual_mode() -> None:
    state, view, mode, button = clear_recommendation_state()

    assert state is None
    assert view == ""
    assert mode == "manual"
    assert button["interactive"] is False


def test_recommendation_parameter_edit_marks_recommendation_as_edited() -> None:
    assert mark_recommendation_edited("recommended") == "recommended_edited"
    assert mark_recommendation_edited("recommended_edited") == "recommended_edited"
    assert mark_recommendation_edited("manual") == "manual"
    assert mark_recommendation_edited(None) == "manual"
