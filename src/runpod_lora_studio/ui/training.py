from __future__ import annotations

from typing import Any
from uuid import UUID

import gradio as gr

from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.training_service import TrainingService
from runpod_lora_studio.ui.training_controller import TrainingController


def build_training_tab(service: TrainingService, selected_project: gr.State) -> None:
    controller = TrainingController(service)
    gr.Markdown("### Phase 6A SDXL LoRA学習")
    with gr.Row():
        snapshot = gr.Dropdown(label="completed dataset snapshot", choices=[])
        model = gr.Dropdown(label="検証済みローカルモデル", choices=[])
    with gr.Row():
        name = gr.Textbox(value="default", label="学習設定名")
        output_name = gr.Textbox(value="lora-output", label="output name")
        output_directory = gr.Textbox(
            value=str(service.settings.outputs_dir), label="output directory"
        )
    with gr.Row():
        sd_scripts_root = gr.Textbox(
            value=str(service.settings.training_sd_scripts_root),
            label="sd-scripts root",
        )
        resolution = gr.Number(value=1024, precision=0, label="resolution")
        batch_size = gr.Number(value=1, precision=0, label="batch size")
        epochs = gr.Number(value=1, precision=0, label="epochs")
        repeats = gr.Number(value=1, precision=0, label="repeats")
    with gr.Row():
        learning_rate = gr.Number(value=0.0001, label="learning rate")
        optimizer = gr.Textbox(value="AdamW8bit", label="optimizer")
        scheduler = gr.Textbox(value="cosine", label="scheduler")
        network_dim = gr.Number(value=16, precision=0, label="network dim")
        network_alpha = gr.Number(value=16, precision=0, label="network alpha")
    with gr.Row():
        mixed_precision = gr.Dropdown(
            choices=["fp16", "bf16", "no"], value="fp16", label="mixed precision"
        )
        seed = gr.Number(value=42, precision=0, label="seed")
        save_every = gr.Number(value=1, precision=0, label="save every N epochs")
        cache_latents = gr.Checkbox(value=False, label="cache latents")
        gradient_checkpointing = gr.Checkbox(
            value=False, label="gradient checkpointing"
        )
    with gr.Row():
        save_config = gr.Button("学習設定を保存")
        create_job = gr.Button("学習ジョブを作成")
        start_job = gr.Button("学習開始", variant="primary")
        refresh = gr.Button("状態更新")
        cancel = gr.Button("停止要求")
    config_id = gr.State(value=None)
    job_id = gr.Textbox(label="job ID")
    message = gr.Markdown()
    jobs_table = gr.Dataframe(
        headers=[
            "ID",
            "status",
            "PID",
            "started",
            "finished",
            "exit code",
            "heartbeat",
            "failure",
        ],
        interactive=False,
        label="学習ジョブ",
    )
    with gr.Row():
        stdout = gr.Textbox(label="stdout末尾", lines=8, interactive=False)
        stderr = gr.Textbox(label="stderr末尾", lines=8, interactive=False)

    def choices(project_id: str | None) -> tuple[object, object]:
        try:
            return (
                gr.update(choices=controller.snapshot_choices(project_id)),
                gr.update(choices=controller.model_choices(project_id)),
            )
        except (UserFacingError, ValueError) as exc:
            return gr.update(choices=[], label=f"エラー: {exc}"), gr.update(choices=[])

    def save_action(project_id: str | None, *values: Any) -> str | None:
        if not project_id:
            return "プロジェクトを選択してください"
        try:
            return controller.save_config(project_id, values)
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    def create_action(saved_config: str | None) -> tuple[str | None, str]:
        if not saved_config:
            return None, "先に学習設定を保存してください"
        try:
            job = controller.create_job(saved_config)
            return job, f"学習ジョブを作成しました: `{job}`"
        except (UserFacingError, ValueError) as exc:
            return None, f"エラー: {exc}"

    def start_action(current_job: str | None) -> str:
        if not current_job:
            return "学習ジョブを指定してください"
        try:
            controller.start_job(current_job)
            return "学習をバックグラウンドで開始しました"
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    def refresh_action(
        project_id: str | None, current_job: str | None
    ) -> tuple[list[list[str]], str, str]:
        rows = controller.job_rows(project_id)
        if not current_job:
            return rows, "", ""
        try:
            job_uuid = UUID(current_job.strip())
            return rows, service.tail_stdout(job_uuid), service.tail_stderr(job_uuid)
        except (UserFacingError, ValueError) as exc:
            return rows, f"エラー: {exc}", ""

    def cancel_action(current_job: str | None) -> str:
        if not current_job:
            return "学習ジョブを指定してください"
        try:
            return controller.cancel_job(current_job)
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    selected_project.change(
        choices, inputs=[selected_project], outputs=[snapshot, model]
    )
    save_inputs = [
        selected_project,
        snapshot,
        model,
        name,
        output_name,
        output_directory,
        sd_scripts_root,
        resolution,
        batch_size,
        epochs,
        repeats,
        learning_rate,
        optimizer,
        scheduler,
        network_dim,
        network_alpha,
        mixed_precision,
        seed,
        save_every,
        cache_latents,
        gradient_checkpointing,
    ]
    save_config.click(save_action, inputs=save_inputs, outputs=[config_id])
    create_job.click(create_action, inputs=[config_id], outputs=[job_id, message])
    start_job.click(start_action, inputs=[job_id], outputs=[message])
    refresh.click(
        refresh_action,
        inputs=[selected_project, job_id],
        outputs=[jobs_table, stdout, stderr],
    )
    cancel.click(cancel_action, inputs=[job_id], outputs=[message])
