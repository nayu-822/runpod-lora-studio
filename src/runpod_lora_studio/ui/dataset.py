from __future__ import annotations

from typing import Any
from uuid import UUID

import gradio as gr

from runpod_lora_studio.domain.models import DatasetSettings
from runpod_lora_studio.services.dataset_snapshot_service import DatasetSnapshotService
from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.ui.dataset_controller import (
    DatasetController,
    preview_rows,
    preview_summary,
    snapshot_rows,
)


def build_dataset_tab(service: DatasetSnapshotService, selected_id: gr.State) -> None:
    controller = DatasetController(service)
    preview_state = gr.State(value=None)
    with gr.Row():
        name = gr.Textbox(label="スナップショット名", value="")
        description = gr.Textbox(label="説明", value="")
    with gr.Row():
        resolution = gr.Number(value=1024, precision=0, label="resolution")
        enable_bucket = gr.Checkbox(value=True, label="bucketを有効化")
        min_bucket = gr.Number(value=256, precision=0, label="min bucket")
        max_bucket = gr.Number(value=2048, precision=0, label="max bucket")
        bucket_steps = gr.Number(value=64, precision=0, label="bucket steps")
    with gr.Row():
        repeats = gr.Number(value=1, precision=0, label="num repeats")
        caption_extension = gr.Textbox(value=".txt", label="caption extension")
        shuffle = gr.Checkbox(value=True, label="shuffle caption")
        keep_tokens = gr.Number(value=0, precision=0, label="keep tokens")
        allow_empty = gr.Checkbox(value=False, label="空キャプションを許可")
        confirm_warnings = gr.Checkbox(value=False, label="警告を確認済み")
    with gr.Row():
        preview_button = gr.Button("作成前検査")
        preview_page = gr.Number(value=1, precision=0, label="検査ページ")
        preview_page_button = gr.Button("検査ページ表示")
        create_button = gr.Button("スナップショット作成", variant="primary")
        reload_button = gr.Button("一覧再読込")
        cancel_id = gr.Textbox(label="キャンセル対象ID", scale=2)
        cancel_button = gr.Button("作成キャンセル")
        revalidate_id = gr.Textbox(label="再検証対象ID", scale=2)
        revalidate_button = gr.Button("スナップショット再検証")
    message = gr.Markdown()
    preview_message = gr.Markdown()
    preview_table = gr.Dataframe(
        headers=[
            "ファイル名",
            "画像ID",
            "サイズ",
            "縦横比",
            "容量",
            "状態",
            "caption revision",
            "文字数",
            "タグ数",
            "trigger数",
            "品質",
            "完全重複",
            "対象可否",
            "問題",
        ],
        interactive=False,
        label="画像別作成前検査",
    )
    snapshot_table = gr.Dataframe(
        headers=[
            "ID",
            "名前",
            "状態",
            "作成日時",
            "画像数",
            "容量",
            "警告数",
            "内容ハッシュ",
            "TaggerRun",
        ],
        interactive=False,
        label="データセットスナップショット一覧",
    )

    def make_settings(values: tuple[Any, ...]) -> DatasetSettings:
        return DatasetSettings(
            resolution=max(1, int(values[0])),
            enable_bucket=bool(values[1]),
            min_bucket_reso=max(1, int(values[2])),
            max_bucket_reso=max(1, int(values[3])),
            bucket_reso_steps=max(1, int(values[4])),
            num_repeats=max(1, int(values[5])),
            caption_extension=str(values[6]).strip(),
            shuffle_caption=bool(values[7]),
            keep_tokens=max(0, int(values[8])),
            allow_empty_caption=bool(values[9]),
        )

    def preview_action(
        project: str | None, *values: Any
    ) -> tuple[str, list[list[str | int | float]], Any]:
        if not project:
            return "エラー: プロジェクトを選択してください。", [], None
        try:
            preview = controller.preview(UUID(project), make_settings(values))
            return preview_summary(preview), preview_rows(preview, 1), preview
        except Exception as exc:
            return f"エラー: {exc}", [], None

    def create_action(
        project: str | None,
        preview: Any,
        snapshot_name: str,
        snapshot_description: str,
        confirmed: bool,
    ) -> str:
        if not project or preview is None:
            return "エラー: 先に作成前検査を実行してください。"
        try:
            snapshot_id = controller.create(
                preview, snapshot_name, snapshot_description, confirmed
            )
            return (
                f"作成を開始しました: {snapshot_id}（完了後に一覧を再読込してください）"
            )
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    def cancel_action(snapshot_id: str) -> str:
        try:
            controller.cancel(UUID(snapshot_id.strip()))
            return "キャンセルを要求しました。"
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    def revalidate_action(snapshot_id: str) -> str:
        if not snapshot_id.strip():
            return "エラー: 再検証対象IDを入力してください。"
        try:
            return f"再検証結果: {controller.revalidate(UUID(snapshot_id.strip()))}"
        except (UserFacingError, ValueError) as exc:
            return f"エラー: {exc}"

    preview_inputs = [
        resolution,
        enable_bucket,
        min_bucket,
        max_bucket,
        bucket_steps,
        repeats,
        caption_extension,
        shuffle,
        keep_tokens,
        allow_empty,
    ]
    preview_button.click(
        preview_action,
        inputs=[selected_id, *preview_inputs],
        outputs=[preview_message, preview_table, preview_state],
    )
    preview_page_button.click(
        lambda preview, page: preview_rows(preview, int(page)) if preview else [],
        inputs=[preview_state, preview_page],
        outputs=[preview_table],
    )
    create_button.click(
        create_action,
        inputs=[selected_id, preview_state, name, description, confirm_warnings],
        outputs=[message],
    ).then(
        lambda project: snapshot_rows(service, UUID(project)) if project else [],
        inputs=[selected_id],
        outputs=[snapshot_table],
    )
    reload_button.click(
        lambda project: snapshot_rows(service, UUID(project)) if project else [],
        inputs=[selected_id],
        outputs=[snapshot_table],
    )
    cancel_button.click(cancel_action, inputs=[cancel_id], outputs=[message])
    revalidate_button.click(
        revalidate_action, inputs=[revalidate_id], outputs=[message]
    )
    selected_id.change(
        lambda project: snapshot_rows(service, UUID(project)) if project else [],
        inputs=[selected_id],
        outputs=[snapshot_table],
    )
