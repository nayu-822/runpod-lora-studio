from __future__ import annotations

from typing import Any
from uuid import UUID

import gradio as gr

from runpod_lora_studio.domain.models import SelectionState
from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.similarity_detection_service import (
    SimilarityDetectionService,
)
from runpod_lora_studio.ui.similarity_controller import (
    SimilarityController,
    similarity_detail_rows,
    similarity_gallery,
    similarity_group_rows,
    similarity_summary_markdown,
)


def build_similarity_tab(
    service: SimilarityDetectionService,
    selected_id: gr.State,
    refresh_event: gr.State | None = None,
) -> None:
    controller = SimilarityController(service)
    with gr.Row():
        run_all = gr.Button("プロジェクト全体を検査", variant="primary")
        run_selected = gr.Button("選択画像のpHashを再計算")
        reload_groups = gr.Button("類似グループを再読込")
    message = gr.Markdown()
    summary = gr.Markdown("プロジェクトを選択してください。")
    with gr.Row():
        page = gr.Number(value=1, minimum=0, precision=0, label="ページ")
        page_size = gr.Number(
            value=service.settings.similarity_group_page_size,
            minimum=1,
            maximum=100,
            precision=0,
            label="表示件数",
        )
        previous = gr.Button("前のページ")
        following = gr.Button("次のページ")
    group_table = gr.Dataframe(
        headers=[
            "グループID",
            "画像数",
            "代表画像",
            "種別",
            "最大距離",
            "確認状態",
            "代表設定",
        ],
        interactive=False,
        label="類似グループ一覧",
    )
    group_choices = gr.Dropdown(choices=[], label="グループを選択")
    page_info = gr.Markdown("0 / 0ページ")
    group_gallery = gr.Gallery(
        label="グループ内比較（サムネイル）",
        columns=6,
        height="auto",
        object_fit="contain",
    )
    group_detail = gr.Dataframe(
        headers=[
            "画像ID",
            "ファイル名",
            "サイズ",
            "採用状態",
            "代表距離",
            "最小距離",
            "候補スコア",
            "代表",
            "確認状態",
            "Phase 2A理由",
        ],
        interactive=False,
        label="グループ詳細",
    )
    selected_group_images = gr.CheckboxGroup(choices=[], label="状態変更対象")
    representative = gr.Dropdown(choices=[], label="代表画像（手動変更）")
    with gr.Row():
        set_rep = gr.Button("選択画像を代表にする")
        confirm = gr.Button("類似と確認")
        reject = gr.Button("類似ではないと判定")
        accept = gr.Button("採用")
        pending = gr.Button("保留")
        excluded = gr.Button("除外")

    def empty_detail() -> tuple[list[tuple[str, str]], list[list[Any]], Any, Any]:
        return (
            [],
            [],
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
        )

    def load(
        project_id: str | None, page_value: float, size_value: float
    ) -> tuple[
        str,
        str,
        list[list[str | int]],
        Any,
        str,
        Any,
        list[list[Any]],
        Any,
        Any,
    ]:
        if not project_id:
            return (
                "プロジェクトを選択してください。",
                "プロジェクトを選択してください。",
                [],
                gr.update(choices=[], value=None),
                "0 / 0ページ",
                [],
                [],
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=[]),
            )
        project = UUID(project_id)
        size = max(1, min(100, int(size_value)))
        view = controller.list_page(project, int(page_value), size)
        choices = [
            (f"{group.id} ({len(group.members)}画像)", str(group.id))
            for group in view.groups
        ]
        summary_text = similarity_summary_markdown(controller.summary(project))
        first = view.groups[0] if view.groups else None
        return (
            "",
            summary_text,
            similarity_group_rows(view.groups),
            gr.update(choices=choices, value=str(first.id) if first else None),
            f"{view.page} / {view.total_pages}ページ（{view.total}グループ）",
            similarity_gallery(first) if first else [],
            similarity_detail_rows(first) if first else [],
            gr.update(
                choices=[
                    (
                        f"{member.image.original_filename} "
                        f"({str(member.image_id)[:8]})",
                        str(member.image_id),
                    )
                    for member in first.members
                    if member.image is not None
                ]
                if first
                else [],
                value=str(first.representative_image_id)
                if first and first.representative_image_id
                else None,
            ),
            gr.update(choices=[], value=[]),
        )

    def detail(
        group_id: str | None,
    ) -> tuple[list[tuple[str, str]], list[list[Any]], Any, Any]:
        if not group_id:
            return empty_detail()
        group = controller.group(UUID(group_id))
        if group is None:
            return empty_detail()
        choices = [
            (
                f"{member.image.original_filename} ({str(member.image_id)[:8]})",
                str(member.image_id),
            )
            for member in group.members
            if member.image is not None
        ]
        return (
            similarity_gallery(group),
            similarity_detail_rows(group),
            gr.update(
                choices=choices,
                value=str(group.representative_image_id)
                if group.representative_image_id
                else None,
            ),
            gr.update(choices=choices, value=[]),
        )

    def run(project_id: str | None, selected: list[str] | None) -> str:
        if not project_id:
            return "エラー: プロジェクトを選択してください。"
        try:
            result = controller.run(
                UUID(project_id), [UUID(item) for item in (selected or [])] or None
            )
            return (
                f"pHash検査完了: 計算 {result.calculated_image_count}件、"
                f"失敗 {result.failed_image_count}件、グループ {result.group_count}件"
            )
        except Exception as exc:
            return f"エラー: {exc}"

    def set_representative_action(group_id: str | None, image_id: str | None) -> str:
        if not group_id or not image_id:
            return "エラー: グループと代表画像を選択してください。"
        try:
            controller.set_representative(UUID(group_id), UUID(image_id))
            return "代表画像を変更しました。"
        except (ValueError, UserFacingError) as exc:
            return f"エラー: {exc}"

    def review_action(group_id: str | None, similar: bool) -> str:
        if not group_id:
            return "エラー: グループを選択してください。"
        try:
            controller.review(UUID(group_id), similar)
            return "手動確認状態を保存しました。"
        except ValueError as exc:
            return f"エラー: {exc}"

    def state_action(
        project_id: str | None,
        ids: list[str] | None,
        state: SelectionState,
        refresh_value: int | None,
    ) -> tuple[str, int | None]:
        if not project_id or not ids:
            return "エラー: プロジェクトと画像を選択してください。", refresh_value
        try:
            count = controller.change_state(
                UUID(project_id), [UUID(item) for item in ids], state
            )
            return (
                f"{count}件の状態を{state.value}へ変更しました。",
                (refresh_value or 0) + 1,
            )
        except Exception as exc:
            return f"エラー: {exc}", refresh_value

    def state_action_message(
        project_id: str | None, ids: list[str] | None, state: SelectionState
    ) -> str:
        return state_action(project_id, ids, state, None)[0]

    outputs = [
        message,
        summary,
        group_table,
        group_choices,
        page_info,
        group_gallery,
        group_detail,
        representative,
        selected_group_images,
    ]
    run_all.click(
        lambda project: run(project, None), inputs=[selected_id], outputs=[message]
    ).then(load, inputs=[selected_id, page, page_size], outputs=outputs)
    run_selected.click(
        run, inputs=[selected_id, selected_group_images], outputs=[message]
    ).then(load, inputs=[selected_id, page, page_size], outputs=outputs)
    reload_groups.click(load, inputs=[selected_id, page, page_size], outputs=outputs)
    selected_id.change(
        lambda project: load(project, 1, service.settings.similarity_group_page_size),
        inputs=[selected_id],
        outputs=outputs,
    )
    group_choices.change(
        detail,
        inputs=[group_choices],
        outputs=[group_gallery, group_detail, representative, selected_group_images],
    )
    set_rep.click(
        set_representative_action,
        inputs=[group_choices, representative],
        outputs=[message],
    ).then(
        detail,
        inputs=[group_choices],
        outputs=[group_gallery, group_detail, representative, selected_group_images],
    )
    confirm.click(
        lambda group: review_action(group, True),
        inputs=[group_choices],
        outputs=[message],
    ).then(
        detail,
        inputs=[group_choices],
        outputs=[group_gallery, group_detail, representative, selected_group_images],
    )
    reject.click(
        lambda group: review_action(group, False),
        inputs=[group_choices],
        outputs=[message],
    ).then(
        detail,
        inputs=[group_choices],
        outputs=[group_gallery, group_detail, representative, selected_group_images],
    )
    for button, state in (
        (accept, SelectionState.ACCEPTED),
        (pending, SelectionState.PENDING),
        (excluded, SelectionState.EXCLUDED),
    ):
        state_handler: Any
        if refresh_event is None:
            state_inputs = [selected_id, selected_group_images]
            state_outputs: list[Any] = [message]

            def state_handler(
                project: str | None,
                ids: list[str] | None,
                target: SelectionState = state,
            ) -> str:
                return state_action_message(project, ids, target)

        else:
            state_inputs = [selected_id, selected_group_images, refresh_event]
            state_outputs = [message, refresh_event]

            def state_handler(
                project: str | None,
                ids: list[str] | None,
                refresh: int | None,
                target: SelectionState = state,
            ) -> tuple[str, int | None]:
                return state_action(project, ids, target, refresh)

        button.click(
            state_handler,
            inputs=state_inputs,
            outputs=state_outputs,
        ).then(
            detail,
            inputs=[group_choices],
            outputs=[
                group_gallery,
                group_detail,
                representative,
                selected_group_images,
            ],
        )
    previous.click(
        lambda project, current, size: load(project, max(1, int(current) - 1), size),
        inputs=[selected_id, page, page_size],
        outputs=outputs,
    )
    following.click(
        lambda project, current, size: load(project, int(current) + 1, size),
        inputs=[selected_id, page, page_size],
        outputs=outputs,
    )
