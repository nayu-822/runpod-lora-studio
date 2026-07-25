from __future__ import annotations

from typing import Any
from uuid import UUID

import gradio as gr

from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.storage_service import StorageService
from runpod_lora_studio.ui.storage_controller import StorageController


def build_storage_tab(service: StorageService, selected_project: gr.State) -> None:
    controller = StorageController(service)
    with gr.Row():
        refresh_environment = gr.Button("rclone環境を確認")
        environment_table = gr.Dataframe(
            headers=["項目", "状態", "説明"],
            interactive=False,
            label="Storage environment",
        )
    with gr.Row():
        query = gr.Textbox(label="モデル名検索")
        extension = gr.Dropdown(
            choices=["", ".safetensors", ".yaml", ".json", ".ckpt"],
            value=".safetensors",
            label="拡張子",
        )
        recursive = gr.Checkbox(value=True, label="サブフォルダを再帰検索")
        page = gr.Number(value=1, precision=0, label="ページ")
        list_button = gr.Button("モデル一覧")
    model_table = gr.Dataframe(
        headers=[
            "ID",
            "ファイル名",
            "種別",
            "remote path",
            "サイズ",
            "状態",
            "local path",
            "local SHA-256",
            "remote hash type",
            "remote hash",
            "検証日時",
        ],
        interactive=False,
        label="Google Driveモデル",
    )
    selected_model = gr.Textbox(label="モデルID")
    model_plan_token = gr.State(value=None)
    model_plan_message = gr.Markdown()
    with gr.Row():
        model_dry_run = gr.Button("モデル取得ドライラン")
        model_download = gr.Button("モデル取得")
        model_verify = gr.Button("完全検証")
    snapshot_id = gr.Textbox(label="completed snapshot ID")
    snapshot_plan_token = gr.State(value=None)
    snapshot_plan_message = gr.Markdown()
    with gr.Row():
        snapshot_dry_run = gr.Button("snapshot uploadドライラン")
        snapshot_upload = gr.Button("snapshot upload")
    jobs_table = gr.Dataframe(
        headers=["ID", "種別", "状態", "step", "処理件数", "bytes", "error"],
        interactive=False,
        label="転送ジョブ",
    )
    message = gr.Markdown()

    def environment_action() -> list[list[str]]:
        return controller.environment_rows()

    def model_action(
        search: str, suffix: str, _recursive: bool, page_number: Any
    ) -> list[list[str]]:
        try:
            models = service.list_models(
                recursive=bool(_recursive),
                query=search,
                extension=suffix or None,
                page=max(1, int(page_number)),
            )
            return controller.model_rows(models)
        except (UserFacingError, ValueError) as exc:
            return [["ERROR", str(exc)]]

    def model_plan_action(model: str) -> tuple[str, str | None]:
        try:
            plan = controller.dry_run_model(UUID(model.strip()))
            return controller.plan_message(plan), plan.token
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}", None

    def model_download_action(model: str, token: str | None) -> str:
        try:
            job_id = controller.download_model(UUID(model.strip()), token)
            return f"モデル取得ジョブ: {job_id}"
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    def model_verify_action(model: str) -> str:
        try:
            return controller.verify_model(UUID(model.strip()))
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    def snapshot_plan_action(snapshot: str) -> tuple[str, str | None]:
        try:
            plan = controller.dry_run_snapshot(UUID(snapshot.strip()))
            return controller.plan_message(plan), plan.token
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}", None

    def snapshot_upload_action(snapshot: str, token: str | None) -> str:
        try:
            job_id = controller.upload_snapshot(UUID(snapshot.strip()), token)
            return f"snapshot uploadジョブ: {job_id}"
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    def jobs_action(project: str | None) -> list[list[str]]:
        jobs = controller.jobs(UUID(project) if project else None)
        return [
            [
                str(job.id),
                job.transfer_type.value,
                job.status.value,
                job.current_step,
                f"{job.processed_item_count}/{job.item_count}",
                f"{job.transferred_bytes}/{job.total_bytes}",
                job.error_summary or "",
            ]
            for job in jobs
        ]

    refresh_environment.click(environment_action, outputs=[environment_table])
    list_button.click(
        model_action,
        inputs=[query, extension, recursive, page],
        outputs=[model_table],
    )
    model_dry_run.click(
        model_plan_action,
        inputs=[selected_model],
        outputs=[model_plan_message, model_plan_token],
    )
    model_download.click(
        model_download_action,
        inputs=[selected_model, model_plan_token],
        outputs=[message],
    )
    model_verify.click(model_verify_action, inputs=[selected_model], outputs=[message])
    snapshot_dry_run.click(
        snapshot_plan_action,
        inputs=[snapshot_id],
        outputs=[snapshot_plan_message, snapshot_plan_token],
    )
    snapshot_upload.click(
        snapshot_upload_action,
        inputs=[snapshot_id, snapshot_plan_token],
        outputs=[message],
    )
    selected_project.change(
        jobs_action, inputs=[selected_project], outputs=[jobs_table]
    )
