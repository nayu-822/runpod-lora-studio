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
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.project_service import ProjectService
from runpod_lora_studio.services.similarity_detection_service import (
    SimilarityDetectionService,
)
from runpod_lora_studio.services.storage_service import StorageService
from runpod_lora_studio.services.tagging_service import TaggingService
from runpod_lora_studio.ui.dataset import build_dataset_tab
from runpod_lora_studio.ui.phase1 import build_image_tab, build_project_tab
from runpod_lora_studio.ui.similarity import build_similarity_tab
from runpod_lora_studio.ui.storage import build_storage_tab
from runpod_lora_studio.ui.tagging import build_tagging_tab


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
    bf16 = {True: "対応", False: "非対応", None: "不明"}[report.bf16_supported]
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
            f"- bf16対応: `{bf16}`",
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
    for optional_path in (settings.model_cache_dir, settings.transfer_temp_dir):
        if optional_path is not None:
            paths.append(optional_path)
    return [[str(path), str(path.exists())] for path in paths]


def create_app(
    settings: AppSettings | None = None,
    report: EnvironmentReport | None = None,
    project_service: ProjectService | None = None,
    image_service: ImageService | None = None,
) -> gr.Blocks:
    runtime_settings = settings or get_settings()
    environment_report = report or collect_environment_report(runtime_settings)
    projects = project_service or ProjectService(runtime_settings)
    images = image_service or ImageService(runtime_settings, projects)
    similarity = SimilarityDetectionService(runtime_settings, projects, images=images)
    tagging = TaggingService(runtime_settings, projects)
    datasets = DatasetSnapshotService(runtime_settings, projects)
    storage = StorageService(runtime_settings, datasets=datasets)
    datasets.recover_finalized_snapshots()
    datasets.recover_stale()
    storage.recover_stale_jobs()
    path_rows = build_paths_dataframe(runtime_settings)

    with gr.Blocks(title=runtime_settings.app_title) as demo:
        with gr.Tab("概要"):
            gr.Markdown(f"# {runtime_settings.app_title}")
            gr.Markdown(
                "RunPod 上で SDXL LoRA のデータ収集・学習ワークフローを動かすための "
                "Phase 1 ベースラインです。"
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        build_status_markdown(runtime_settings, environment_report)
                    )
                with gr.Column():
                    gr.Dataframe(
                        headers=["Path", "Exists"],
                        value=path_rows,
                        label="Runtime Paths",
                        interactive=False,
                        row_count=len(path_rows),
                    )
        with gr.Tab("プロジェクト"):
            selected_project, project_table = build_project_tab(projects)
        image_refresh = gr.State(value=0)
        with gr.Tab("画像"):
            build_image_tab(images, selected_project, project_table, image_refresh)
        with gr.Tab("近似重複"):
            build_similarity_tab(similarity, selected_project, image_refresh)
        with gr.Tab("タグ付け・キャプション"):
            build_tagging_tab(tagging, selected_project, image_refresh)
        with gr.Tab("データセット"):
            build_dataset_tab(datasets, selected_project)
        with gr.Tab("モデル・Google Drive"):
            build_storage_tab(storage, selected_project)
        gr.Markdown(
            "Phase 5のモデル管理・Google Drive転送基盤まで実装済みです。"
            "学習実行、学習成果物の最終同期、RunPod制御は後続Phaseの対象です。"
        )

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
