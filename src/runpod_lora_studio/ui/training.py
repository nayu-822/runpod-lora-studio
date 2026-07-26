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
    with gr.Row():
        learning_rate = gr.Number(value=0.0001, label="learning rate")
        optimizer = gr.Dropdown(
            choices=["AdamW", "AdamW8bit", "Lion", "Prodigy"],
            value="AdamW8bit",
            label="optimizer",
        )
        scheduler = gr.Dropdown(
            choices=[
                "constant",
                "constant_with_warmup",
                "cosine",
                "cosine_with_restarts",
                "linear",
            ],
            value="cosine",
            label="scheduler",
        )
        network_module = gr.Dropdown(
            choices=["networks.lora"], value="networks.lora", label="network module"
        )
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
    progress_view = gr.Dataframe(
        headers=[
            "status",
            "epoch",
            "step",
            "progress",
            "loss",
            "smoothed loss",
            "learning rate",
            "steps/sec",
            "elapsed",
            "ETA",
            "latest log",
            "warning",
            "source",
        ],
        value=[],
        interactive=False,
        label="training progress",
    )
    loss_graph = gr.LinePlot(
        x="step", y="loss", title="loss history", interactive=False
    )
    artifacts_table = gr.Dataframe(
        headers=[
            "filename",
            "type",
            "epoch",
            "step",
            "size",
            "SHA-256",
            "validation",
            "message",
            "modified",
        ],
        value=[],
        interactive=False,
        label="training artifacts",
    )
    with gr.Row():
        reparse = gr.Button("進捗を再解析")
        rescan = gr.Button("成果物を再走査")

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
    ) -> tuple[
        list[list[str]],
        str,
        str,
        list[list[str]],
        list[dict[str, object]],
        list[list[str]],
    ]:
        rows = controller.job_rows(project_id)
        if not current_job:
            return rows, "", "", [], [], []
        try:
            job_uuid = UUID(current_job.strip())
            progress = controller.progress_row(current_job)
            metrics = controller.metric_rows(current_job)
            graph = [{"step": step, "loss": value} for step, value, _ in metrics]
            return (
                rows,
                service.tail_stdout(job_uuid),
                service.tail_stderr(job_uuid),
                [progress],
                graph,
                controller.artifact_rows(current_job),
            )
        except (UserFacingError, ValueError) as exc:
            return rows, f"エラー: {exc}", "", [], [], []

    def reparse_action(current_job: str | None) -> str:
        if not current_job:
            return "jobを選択してください"
        service.refresh_progress(UUID(current_job.strip()))
        return "進捗を再解析しました"

    def rescan_action(current_job: str | None) -> str:
        if not current_job:
            return "jobを選択してください"
        service.rescan_artifacts(UUID(current_job.strip()))
        return "成果物を再走査しました"

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
        learning_rate,
        optimizer,
        scheduler,
        network_module,
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
        outputs=[
            jobs_table,
            stdout,
            stderr,
            progress_view,
            loss_graph,
            artifacts_table,
        ],
    )
    cancel.click(cancel_action, inputs=[job_id], outputs=[message])
    reparse.click(reparse_action, inputs=[job_id], outputs=[message])
    rescan.click(rescan_action, inputs=[job_id], outputs=[message])

    gr.Markdown("### Phase 6C 学習stateからの安全な再開")
    with gr.Row():
        resume_job = gr.Dropdown(label="再開元job", choices=[])
        resume_state = gr.Dropdown(label="再開元state", choices=[])
        resume_config = gr.Dropdown(label="再開先設定（任意）", choices=[])
        resume_preview = gr.Button("再開プレビュー")
    resume_signature = gr.State(value="")
    resume_details = gr.Markdown()
    with gr.Row():
        resume_create = gr.Button("再開jobを作成")
        resume_start = gr.Button("再開jobを開始", variant="primary")
    resume_job_id = gr.Textbox(label="再開後job ID", interactive=False)

    def resume_jobs_action(project_id: str | None) -> object:
        return gr.update(
            choices=controller.resumable_job_choices(project_id), value=None
        )

    def resume_configs_action(project_id: str | None) -> object:
        return gr.update(choices=controller.config_choices(project_id), value=None)

    def resume_states_action(source: str | None) -> object:
        return gr.update(choices=controller.resume_state_choices(source), value=None)

    def resume_preview_action(
        source: str | None, state: str | None, target_config: str | None
    ) -> tuple[str, str]:
        if not source or not state:
            return "再開元jobとstateを選択してください", ""
        try:
            preview = controller.preview_resume(source, state, target_config)
            issues = "、".join(preview.compatibility.issues) or "なし"
            return (
                "\n".join(
                    [
                        f"元job: `{preview.source_job_id}` ({preview.source_status})",
                        f"state: `{preview.source_state_name}`",
                        f"fingerprint: `{preview.state_fingerprint[:12]}`",
                        (
                            f"epoch/step: `{preview.current_epoch}` / "
                            f"`{preview.current_step}`"
                        ),
                        f"compatibility: `{preview.compatibility.status.value}`",
                        f"不一致: {issues}",
                        f"command: `{preview.command_summary}`",
                    ]
                ),
                preview.signature,
            )
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}", ""

    def resume_create_action(
        source: str | None,
        state: str | None,
        target_config: str | None,
        signature: str,
    ) -> tuple[str, str]:
        if not source or not state or not signature:
            return "", "先に再開プレビューを実行してください"
        try:
            created = controller.create_resume_job(
                source, state, signature, target_config
            )
            return created, f"再開jobを作成しました: `{created}`"
        except (UserFacingError, ValueError) as exc:
            return "", f"エラー: {exc}"

    def resume_start_action(child: str | None) -> str:
        if not child:
            return "再開jobを作成してください"
        try:
            controller.start_resume_job(child)
            return "再開jobをバックグラウンドで開始しました"
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    selected_project.change(
        resume_jobs_action, inputs=[selected_project], outputs=[resume_job]
    )
    selected_project.change(
        resume_configs_action, inputs=[selected_project], outputs=[resume_config]
    )
    resume_job.change(resume_states_action, inputs=[resume_job], outputs=[resume_state])
    resume_preview.click(
        resume_preview_action,
        inputs=[resume_job, resume_state, resume_config],
        outputs=[resume_details, resume_signature],
    )
    resume_create.click(
        resume_create_action,
        inputs=[resume_job, resume_state, resume_config, resume_signature],
        outputs=[resume_job_id, message],
    )
    resume_start.click(resume_start_action, inputs=[resume_job_id], outputs=[message])
