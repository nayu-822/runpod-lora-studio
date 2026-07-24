from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import gradio as gr

from runpod_lora_studio.domain.models import (
    ImageAsset,
    ImageInspectionResult,
    InspectionRunResult,
    Project,
    SelectionState,
)
from runpod_lora_studio.services.image_inspection_service import ImageInspectionService
from runpod_lora_studio.services.image_service import ImageService, UploadResult
from runpod_lora_studio.services.project_service import (
    ProjectService,
    UserFacingError,
)
from runpod_lora_studio.ui.phase1_controller import (
    CONCEPT_LABELS,
    ImageController,
    ImageInspectionController,
    ProjectController,
    project_table_rows,
)

STATE_LABELS = {
    "all": "すべて",
    SelectionState.ACCEPTED.value: "採用",
    SelectionState.PENDING.value: "保留",
    SelectionState.EXCLUDED.value: "除外",
}


def project_rows(projects: list[Project]) -> list[list[str | int]]:
    return project_table_rows(projects)


def image_rows(
    images: list[ImageAsset],
    inspection_results: dict[UUID, list[ImageInspectionResult]] | None = None,
) -> list[list[str]]:
    inspection_results = inspection_results or {}

    def inspection_label(image_id: UUID) -> str:
        results = inspection_results.get(image_id, [])
        if not results:
            return "検査未実施"
        if any(item.status.value == "failed" for item in results):
            return "検査失敗"
        if any(item.status.value == "warning" for item in results):
            return "警告あり"
        return "問題なし"

    return [
        [
            str(image.id),
            image.original_filename,
            f"{image.width} x {image.height}",
            str(image.file_size),
            image.mime_type,
            image.sha256[:12],
            inspection_label(image.id),
            image.selection_state.value,
            image.created_at.isoformat(),
            str(image.id)[:8],
        ]
        for image in images
    ]


def gallery_items(images: list[ImageAsset]) -> list[tuple[str, str]]:
    return [
        (str(image.thumbnail_path), f"{image.original_filename} [{str(image.id)[:8]}]")
        for image in images
        if image.thumbnail_path.is_file()
    ]


def gallery_image_ids(images: list[ImageAsset]) -> list[str]:
    return [str(image.id) for image in images if image.thumbnail_path.is_file()]


def format_upload_result(result: UploadResult) -> str:
    lines = [
        f"成功: {len(result.successes)}件",
        f"失敗: {len(result.failures)}件",
        f"同一内容の可能性がある画像: {result.duplicate_warning_count}件",
    ]
    if result.failures:
        lines.extend(["", "失敗:"])
        for failure in result.failures:
            reason = failure.reason
            if Path(reason).is_absolute() or "\\" in reason:
                reason = "画像登録に失敗しました。"
            lines.append(f"- {failure.filename}: {reason}")
    return "\n".join(lines)


def selected_gallery_ids(index: int | tuple[int, ...], ids: list[str]) -> list[str]:
    selected = index[0] if isinstance(index, tuple) else index
    if selected < 0 or selected >= len(ids):
        return []
    return [ids[selected]]


def format_inspection_summary(result: InspectionRunResult) -> str:
    summary = result.summary
    return "\n".join(
        [
            f"検査済み: {summary.inspected_images}/{summary.total_images}画像",
            f"警告: {summary.warning_count}件 / 失敗: {summary.failed_count}件",
            f"完全重複: {summary.exact_duplicate_count}件",
            f"最低解像度未満: {summary.resolution_too_small_count}件",
            f"極端な縦横比: {summary.aspect_ratio_extreme_count}件",
            f"低情報量候補: {summary.low_information_count}件",
            f"ぼけ候補: {summary.blur_score_count}件",
        ]
    )


def inspection_rows(results: list[ImageInspectionResult]) -> list[list[str | float]]:
    return [
        [
            result.rule.value,
            result.status.value,
            "" if result.score is None else round(result.score, 4),
            "" if result.threshold is None else round(result.threshold, 4),
            result.reason,
            result.detector_version,
        ]
        for result in results
    ]


def build_project_tab(projects: ProjectService) -> tuple[gr.State, gr.Dataframe]:
    controller = ProjectController(projects)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 新規プロジェクト")
            new_name = gr.Textbox(label="名前")
            new_description = gr.Textbox(label="説明", lines=2)
            new_concept = gr.Dropdown(
                choices=list(CONCEPT_LABELS.values()), value="その他", label="概念種別"
            )
            new_trigger_words = gr.Textbox(label="トリガーワード（カンマ区切り）")
            create_button = gr.Button("作成", variant="primary")
            project_message = gr.Markdown()
        with gr.Column():
            project_table = gr.Dataframe(
                headers=[
                    "ID",
                    "名前",
                    "概念種別",
                    "状態",
                    "採用",
                    "保留",
                    "除外",
                    "作成日時",
                    "更新日時",
                ],
                interactive=False,
                label="プロジェクト一覧",
            )
            project_reload = gr.Button("一覧を再読込")
            project_selector = gr.Dropdown(label="使用するプロジェクト", choices=[])
            selected_id = gr.State(value=None)
    with gr.Row():
        edit_name = gr.Textbox(label="選択中プロジェクト名")
        edit_description = gr.Textbox(label="説明")
        edit_concept = gr.Dropdown(
            choices=list(CONCEPT_LABELS.values()), label="概念種別"
        )
        edit_trigger_words = gr.Textbox(label="トリガーワード")
        edit_button = gr.Button("編集内容を保存")
    edit_message = gr.Markdown()

    def reload(current_id: str | None) -> tuple[list[list[str | int]], object, object]:
        view = controller.reload(current_id)
        return (
            view.rows,
            gr.update(choices=view.choices, value=view.selected_id),
            view.selected_id,
        )

    def create(
        name: str, description: str, concept: str, triggers: str
    ) -> tuple[object, object, str | None, str, str, object, str, str]:
        try:
            created, view = controller.create(name, description, concept, triggers)
            project_id = str(created.id)
            return (
                view.rows,
                gr.update(choices=view.choices, value=view.selected_id),
                view.selected_id,
                created.name,
                created.description,
                gr.update(value=CONCEPT_LABELS[created.concept_type.value]),
                ", ".join(created.trigger_words),
                f"プロジェクトを作成しました。選択ID: `{project_id}`",
            )
        except UserFacingError as exc:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                f"エラー: {exc}",
            )
        except Exception:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "エラー: プロジェクトを作成できませんでした。",
            )

    def select(project_id: str | None) -> tuple[str, str, object, str, str]:
        if not project_id:
            return (
                "",
                "",
                gr.update(value="その他"),
                "",
                "プロジェクトを選択してください。",
            )
        try:
            project = projects.get(UUID(project_id))
            return (
                project.name,
                project.description,
                gr.update(value=CONCEPT_LABELS[project.concept_type.value]),
                ", ".join(project.trigger_words),
                f"選択中: `{project.name}`",
            )
        except (ValueError, UserFacingError) as exc:
            return "", "", gr.update(value="その他"), "", f"エラー: {exc}"

    def save_edit(
        project_id: str | None, name: str, description: str, concept: str, triggers: str
    ) -> tuple[object, object, str | None, str, str, object, str, str]:
        if not project_id:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "エラー: プロジェクトを選択してください。",
            )
        try:
            updated, view = controller.update(
                project_id, name, description, concept, triggers
            )
            return (
                view.rows,
                gr.update(choices=view.choices, value=view.selected_id),
                view.selected_id,
                updated.name,
                updated.description,
                gr.update(value=CONCEPT_LABELS[updated.concept_type.value]),
                ", ".join(updated.trigger_words),
                "プロジェクトを更新しました。",
            )
        except (ValueError, UserFacingError) as exc:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                f"エラー: {exc}",
            )

    project_reload.click(
        reload,
        inputs=[selected_id],
        outputs=[project_table, project_selector, selected_id],
    ).then(
        select,
        inputs=[selected_id],
        outputs=[
            edit_name,
            edit_description,
            edit_concept,
            edit_trigger_words,
            edit_message,
        ],
    )
    create_button.click(
        create,
        inputs=[new_name, new_description, new_concept, new_trigger_words],
        outputs=[
            project_table,
            project_selector,
            selected_id,
            edit_name,
            edit_description,
            edit_concept,
            edit_trigger_words,
            project_message,
        ],
    )
    project_selector.change(
        select,
        inputs=project_selector,
        outputs=[
            edit_name,
            edit_description,
            edit_concept,
            edit_trigger_words,
            edit_message,
        ],
    ).then(lambda value: value, inputs=project_selector, outputs=selected_id)
    edit_button.click(
        save_edit,
        inputs=[
            selected_id,
            edit_name,
            edit_description,
            edit_concept,
            edit_trigger_words,
        ],
        outputs=[
            project_table,
            project_selector,
            selected_id,
            edit_name,
            edit_description,
            edit_concept,
            edit_trigger_words,
            edit_message,
        ],
    )
    return selected_id, project_table


def build_image_tab(
    images: ImageService, selected_id: gr.State, project_table: gr.Dataframe
) -> None:
    controller = ImageController(images)
    inspection = ImageInspectionService(images.settings, images.projects)
    inspection_controller = ImageInspectionController(inspection)
    project_label = gr.Markdown("プロジェクト未選択")
    upload = gr.File(
        file_count="multiple", type="filepath", label="画像（JPEG / PNG / WebP）"
    )
    upload_button = gr.Button("画像を登録", variant="primary")
    upload_message = gr.Markdown()
    with gr.Row():
        state_filter = gr.Dropdown(
            choices=list(STATE_LABELS.values()), value="すべて", label="状態フィルター"
        )
        search = gr.Textbox(label="ファイル名検索")
        page = gr.Number(value=1, minimum=0, precision=0, label="ページ")
        page_size = gr.Number(
            value=30, minimum=1, maximum=100, precision=0, label="件数"
        )
    image_table = gr.Dataframe(
        headers=[
            "ID",
            "元ファイル名",
            "サイズ",
            "容量",
            "MIME",
            "SHA-256",
            "Inspection",
            "状態",
            "登録日時",
            "ID短縮",
        ],
        interactive=False,
        label="画像一覧",
    )
    gallery = gr.Gallery(
        label="サムネイル一覧",
        columns=5,
        height="auto",
        show_label=True,
        object_fit="contain",
    )
    gr.Markdown(
        "サムネイルを選択すると単一画像を操作できます。"
        "複数画像を変更する場合は、下の一覧から対象を選択してください。"
    )
    gallery_ids = gr.State(value=[])
    image_ids = gr.CheckboxGroup(label="状態変更対象（複数選択）", choices=[])
    with gr.Row():
        accepted = gr.Button("採用へ変更")
        pending = gr.Button("保留へ変更")
        excluded = gr.Button("除外へ変更")
        image_reload = gr.Button("一覧を再読込")
    image_message = gr.Markdown()
    with gr.Row():
        inspect_project = gr.Button("プロジェクト全体を検査", variant="primary")
        inspect_selected = gr.Button("選択画像を再検査")
    inspection_summary = gr.Markdown("検査未実施")
    inspection_details = gr.Dataframe(
        headers=["検査ルール", "結果", "計測値", "閾値", "理由", "バージョン"],
        interactive=False,
        label="選択画像の検査結果",
    )
    with gr.Row():
        page_previous = gr.Button("前ページ")
        page_next = gr.Button("次ページ")
        page_info = gr.Markdown("0件")

    def load(
        project_id: str | None,
        filter_value: str,
        query: str,
        page_value: float,
        size: float,
    ) -> tuple[
        list[list[str | int]],
        str,
        list[list[str]],
        list[tuple[str, str]],
        list[str],
        Any,
        Any,
        str,
        str,
    ]:
        if not project_id:
            return (
                gr.skip(),
                "プロジェクト未選択",
                [],
                [],
                [],
                gr.update(choices=[]),
                0,
                "0 / 0ページ、全0件",
                "プロジェクトを選択してください。",
            )
        state = None
        if filter_value != STATE_LABELS["all"]:
            state = next(
                (
                    SelectionState(key)
                    for key, label in STATE_LABELS.items()
                    if label == filter_value and key != "all"
                ),
                None,
            )
        page_size_value = max(1, min(100, int(size)))
        page_view = controller.list_page(
            UUID(project_id),
            state=state,
            search=query,
            page=int(page_value),
            page_size=page_size_value,
        )
        values = page_view.images
        total = page_view.total
        choices = [
            (f"{value.original_filename} ({str(value.id)[:8]})", str(value.id))
            for value in values
        ]
        return (
            project_rows(images.projects.list_projects()),
            f"{images.projects.get(UUID(project_id)).name}（全{total}件）",
            image_rows(values, inspection.get_project_results(UUID(project_id))),
            gallery_items(values),
            gallery_image_ids(values),
            gr.update(choices=choices, value=[]),
            page_view.page.page,
            page_view.page.label,
            "",
        )

    def register(project_id: str | None, files: list[str] | None) -> str:
        if not project_id:
            return "エラー: 先にプロジェクトを選択してください。"
        if not files:
            return "登録する画像を選択してください。"
        try:
            result = controller.register(UUID(project_id), files)
            return format_upload_result(result)
        except UserFacingError as exc:
            return f"エラー: {exc}"

    def change(
        project_id: str | None, ids: list[str] | None, state: SelectionState
    ) -> str:
        if not project_id or not ids:
            return "エラー: プロジェクトと画像を選択してください。"
        try:
            count = controller.change_state(
                UUID(project_id), [UUID(value) for value in ids], state
            )
            return f"{count}件を{STATE_LABELS[state.value]}へ変更しました。"
        except UserFacingError as exc:
            return f"エラー: {exc}"

    def run_inspection(
        project_id: str | None, selected: list[str] | None
    ) -> tuple[str, list[list[str | float]], str]:
        if not project_id:
            return "エラー: 先にプロジェクトを選択してください。", [], "検査未実施"
        selected_ids = [UUID(value) for value in (selected or [])]
        try:
            run = inspection_controller.inspect_project(
                UUID(project_id), selected_ids or None
            )
            details = inspection.get_project_results(UUID(project_id))
            rows = [
                row
                for image_id in (selected_ids or list(details))
                for row in inspection_rows(details.get(image_id, []))
            ]
            return "検査が完了しました。", rows, format_inspection_summary(run)
        except UserFacingError as exc:
            return f"エラー: {exc}", [], "検査未実施"

    def clear_inspection() -> tuple[str, list[list[str | float]]]:
        return "検査結果を選び直してください。", []

    inspect_project.click(
        lambda project_id: run_inspection(project_id, None),
        inputs=[selected_id],
        outputs=[image_message, inspection_details, inspection_summary],
    )
    inspect_selected.click(
        run_inspection,
        inputs=[selected_id, image_ids],
        outputs=[image_message, inspection_details, inspection_summary],
    )

    for control in (selected_id, state_filter, search, page, page_size):
        control.change(
            clear_inspection,
            outputs=[inspection_summary, inspection_details],
        )

    def select_gallery(ids: list[str], event: gr.SelectData) -> list[str]:
        return selected_gallery_ids(event.index, ids)

    gallery.select(select_gallery, inputs=[gallery_ids], outputs=[image_ids])

    upload_button.click(
        register, inputs=[selected_id, upload], outputs=[upload_message]
    ).then(
        load,
        inputs=[selected_id, state_filter, search, page, page_size],
        outputs=[
            project_table,
            project_label,
            image_table,
            gallery,
            gallery_ids,
            image_ids,
            page,
            page_info,
            image_message,
        ],
    )
    image_reload.click(
        load,
        inputs=[selected_id, state_filter, search, page, page_size],
        outputs=[
            project_table,
            project_label,
            image_table,
            gallery,
            gallery_ids,
            image_ids,
            page,
            page_info,
            image_message,
        ],
    )
    selected_id.change(
        lambda project_id: load(project_id, "すべて", "", 1, 30),
        inputs=selected_id,
        outputs=[
            project_table,
            project_label,
            image_table,
            gallery,
            gallery_ids,
            image_ids,
            page,
            page_info,
            image_message,
        ],
    )
    for control in (state_filter, search, page_size):
        control.change(
            lambda project_id, filter_value, query, size: load(
                project_id, filter_value, query, 1, size
            ),
            inputs=[selected_id, state_filter, search, page_size],
            outputs=[
                project_table,
                project_label,
                image_table,
                gallery,
                gallery_ids,
                image_ids,
                page,
                page_info,
                image_message,
            ],
        )
    page_previous.click(
        lambda project_id, filter_value, query, current, size: load(
            project_id, filter_value, query, max(1, int(current) - 1), size
        ),
        inputs=[selected_id, state_filter, search, page, page_size],
        outputs=[
            project_table,
            project_label,
            image_table,
            gallery,
            gallery_ids,
            image_ids,
            page,
            page_info,
            image_message,
        ],
    )
    page_next.click(
        lambda project_id, filter_value, query, current, size: load(
            project_id, filter_value, query, int(current) + 1, size
        ),
        inputs=[selected_id, state_filter, search, page, page_size],
        outputs=[
            project_table,
            project_label,
            image_table,
            gallery,
            gallery_ids,
            image_ids,
            page,
            page_info,
            image_message,
        ],
    )
    for button, state in (
        (accepted, SelectionState.ACCEPTED),
        (pending, SelectionState.PENDING),
        (excluded, SelectionState.EXCLUDED),
    ):
        button.click(
            lambda project_id, ids, target=state: change(project_id, ids, target),
            inputs=[selected_id, image_ids],
            outputs=[image_message],
        ).then(
            load,
            inputs=[selected_id, state_filter, search, page, page_size],
            outputs=[
                project_table,
                project_label,
                image_table,
                gallery,
                gallery_ids,
                image_ids,
                page,
                page_info,
                image_message,
            ],
        )
