from __future__ import annotations

from pathlib import Path
from typing import cast

import gradio as gr

from runpod_lora_studio.config.settings import AppSettings, get_settings
from runpod_lora_studio.logging.config import configure_logging


def build_status_markdown(settings: AppSettings) -> str:
    server_bind = f"{settings.gradio_server_name}:{settings.gradio_server_port}"
    return "\n".join(
        [
            "## Runtime Status",
            "",
            f"- Environment: `{settings.app_env}`",
            f"- Server bind: `{server_bind}`",
            f"- Workspace root: `{settings.workspace_root}`",
            f"- Projects dir: `{settings.projects_dir}`",
            f"- Models dir: `{settings.models_dir}`",
            f"- Outputs dir: `{settings.outputs_dir}`",
            f"- Database path: `{settings.database_path}`",
            f"- RunPod Pod ID detected: `{bool(settings.runpod_pod_id)}`",
        ]
    )


def build_paths_dataframe(settings: AppSettings) -> list[list[str]]:
    paths: list[Path] = [
        settings.workspace_root,
        settings.projects_dir,
        settings.models_dir,
        settings.outputs_dir,
        settings.logs_dir,
        settings.database_path.parent,
    ]
    return [[str(path), str(path.exists())] for path in paths]


def create_app(settings: AppSettings | None = None) -> gr.Blocks:
    runtime_settings = settings or get_settings()
    path_rows = build_paths_dataframe(runtime_settings)

    with gr.Blocks(title=runtime_settings.app_title) as demo:
        gr.Markdown(f"# {runtime_settings.app_title}")
        gr.Markdown(
            "RunPod 上で SDXL LoRA のデータ収集・学習ワークフローを動かすための "
            "Phase 0 ベースラインです。"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown(build_status_markdown(runtime_settings))
            with gr.Column():
                gr.Dataframe(
                    headers=["Path", "Exists"],
                    value=path_rows,
                    label="Runtime Paths",
                    interactive=False,
                    row_count=len(path_rows),
                )

        gr.Markdown(
            "次フェーズでは acquisition / tagging / training / storage の各機能を "
            "段階的に追加していきます。"
        )

    return cast(gr.Blocks, demo)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_runtime_directories()

    app = create_app(settings)
    app.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
    )


if __name__ == "__main__":
    main()
