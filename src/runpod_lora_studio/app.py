from __future__ import annotations

from pathlib import Path
from typing import cast

import gradio as gr

from runpod_lora_studio.config.settings import (
    AppSettings,
    ensure_runtime_directories,
    get_settings,
)
from runpod_lora_studio.environment import EnvironmentReport, collect_environment_report
from runpod_lora_studio.logging.config import configure_logging


def build_status_markdown(settings: AppSettings, report: EnvironmentReport) -> str:
    gpu_summary = "未検出"
    if report.gpus:
        gpu_summary = ", ".join(
            f"GPU {gpu.index}: {gpu.name} ({gpu.vram_mb or '不明'} MB)"
            for gpu in report.gpus
        )
    disk_summary = "不明"
    if report.disk_free_bytes is not None and report.disk_total_bytes is not None:
        free_gib = report.disk_free_bytes // (1024**3)
        total_gib = report.disk_total_bytes // (1024**3)
        disk_summary = f"空き {free_gib} GiB / 総容量 {total_gib} GiB"
    status = "正常稼働中" if not report.errors else "要確認"
    environment = "RunPod" if report.is_runpod else "ローカル"
    return "\n".join(
        [
            "## Runtime Status",
            "",
            f"- 状態: **{status}**",
            f"- アプリケーションバージョン: `{settings.app_version}`",
            f"- Pythonバージョン: `{report.python_version}`",
            f"- 実行環境: `{environment}`",
            f"- GPU認識状況: `{'認識済み' if report.gpus else '未認識'}`",
            f"- GPU概要: `{gpu_summary}`",
            f"- データ保存先: `{settings.workspace_root}`",
            f"- ディスク容量: `{disk_summary}`",
            "- Server bind: "
            f"`{settings.gradio_server_name}:{settings.gradio_server_port}`",
        ]
    )


def build_paths_dataframe(settings: AppSettings) -> list[list[str]]:
    paths: list[Path] = [
        settings.workspace_root,
        settings.projects_dir,
        settings.models_dir,
        settings.outputs_dir,
        settings.logs_dir,
        settings.temp_dir,
        settings.database_path.parent,
    ]
    return [[str(path), str(path.exists())] for path in paths]


def create_app(
    settings: AppSettings | None = None,
    report: EnvironmentReport | None = None,
) -> gr.Blocks:
    runtime_settings = settings or get_settings()
    environment_report = report or collect_environment_report(runtime_settings)
    path_rows = build_paths_dataframe(runtime_settings)

    with gr.Blocks(title=runtime_settings.app_title) as demo:
        gr.Markdown(f"# {runtime_settings.app_title}")
        gr.Markdown(
            "RunPod 上で SDXL LoRA のデータ収集・学習ワークフローを動かすための "
            "Phase 0 ベースラインです。"
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown(build_status_markdown(runtime_settings, environment_report))
            with gr.Column():
                gr.Dataframe(
                    headers=["Path", "Exists"],
                    value=path_rows,
                    label="Runtime Paths",
                    interactive=False,
                    row_count=len(path_rows),
                )
        gr.Markdown("Phase 1以降の機能はまだ実装されていません。")

    return cast(gr.Blocks, demo)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    ensure_runtime_directories(settings)
    create_app(settings).launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
    )


if __name__ == "__main__":
    main()
