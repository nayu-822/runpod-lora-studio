from __future__ import annotations

from typing import Any
from uuid import UUID

import gradio as gr

from runpod_lora_studio.domain.models import (
    ConceptType,
    ImageAsset,
    Project,
    SelectionState,
)
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.project_service import (
    ProjectInput,
    ProjectService,
    UserFacingError,
)

CONCEPT_LABELS = {
    ConceptType.CHARACTER.value: "キャラクター",
    ConceptType.STYLE.value: "画風",
    ConceptType.COSTUME.value: "衣装",
    ConceptType.OBJECT.value: "物体",
    ConceptType.OTHER.value: "その他",
}
STATE_LABELS = {
    "all": "すべて",
    SelectionState.ACCEPTED.value: "採用",
    SelectionState.PENDING.value: "保留",
    SelectionState.EXCLUDED.value: "除外",
}


def project_rows(projects: list[Project]) -> list[list[str | int]]:
    return [
        [
            str(project.id),
            project.name,
            CONCEPT_LABELS[project.concept_type.value],
            project.status.value,
            project.image_counts.get(SelectionState.ACCEPTED, 0),
            project.image_counts.get(SelectionState.PENDING, 0),
            project.image_counts.get(SelectionState.EXCLUDED, 0),
            project.created_at.isoformat(),
            project.updated_at.isoformat(),
        ]
        for project in projects
    ]


def image_rows(images: list[ImageAsset]) -> list[list[str]]:
    return [
        [
            str(image.id),
            str(image.thumbnail_path),
            image.original_filename,
            f"{image.width} x {image.height}",
            str(image.file_size),
            image.mime_type,
            image.sha256[:12],
            image.selection_state.value,
            image.created_at.isoformat(),
        ]
        for image in images
    ]


def _parse_trigger_words(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(","))


def _project_choices(projects: list[Project]) -> list[tuple[str, str]]:
    return [(project.name, str(project.id)) for project in projects]


def build_project_tab(projects: ProjectService) -> gr.State:
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

    def reload() -> tuple[list[list[str | int]], object, object]:
        values = projects.list_projects()
        choices = _project_choices(values)
        return (
            project_rows(values),
            gr.update(choices=choices),
            choices[0][1] if choices else None,
        )

    def create(
        name: str, description: str, concept: str, triggers: str
    ) -> tuple[object, object, str]:
        try:
            concept_type = next(
                key for key, label in CONCEPT_LABELS.items() if label == concept
            )
            projects.create(
                ProjectInput(
                    name,
                    description,
                    ConceptType(concept_type),
                    _parse_trigger_words(triggers),
                )
            )
            rows, choices, first = reload()
            return rows, choices, f"プロジェクトを作成しました。選択ID: `{first}`"
        except UserFacingError as exc:
            return gr.skip(), gr.skip(), f"エラー: {exc}"
        except Exception:
            return gr.skip(), gr.skip(), "エラー: プロジェクトを作成できませんでした。"

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
    ) -> str:
        if not project_id:
            return "エラー: プロジェクトを選択してください。"
        try:
            concept_value = next(
                key for key, label in CONCEPT_LABELS.items() if label == concept
            )
            projects.update(
                UUID(project_id),
                ProjectInput(
                    name,
                    description,
                    ConceptType(concept_value),
                    _parse_trigger_words(triggers),
                ),
            )
            return "プロジェクトを更新しました。"
        except (ValueError, UserFacingError) as exc:
            return f"エラー: {exc}"

    project_reload.click(reload, outputs=[project_table, project_selector, selected_id])
    create_button.click(
        create,
        inputs=[new_name, new_description, new_concept, new_trigger_words],
        outputs=[project_table, project_selector, project_message],
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
        outputs=edit_message,
    )
    return selected_id


def build_image_tab(images: ImageService, selected_id: gr.State) -> None:
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
        page = gr.Number(value=1, minimum=1, precision=0, label="ページ")
        page_size = gr.Number(
            value=30, minimum=1, maximum=100, precision=0, label="件数"
        )
    image_table = gr.Dataframe(
        headers=[
            "ID",
            "サムネイル",
            "元ファイル名",
            "サイズ",
            "容量",
            "MIME",
            "SHA-256",
            "状態",
            "登録日時",
        ],
        interactive=False,
        label="画像一覧",
    )
    image_ids = gr.CheckboxGroup(label="状態変更対象（複数選択）", choices=[])
    with gr.Row():
        accepted = gr.Button("採用へ変更")
        pending = gr.Button("保留へ変更")
        excluded = gr.Button("除外へ変更")
        image_reload = gr.Button("一覧を再読込")
    image_message = gr.Markdown()

    def load(
        project_id: str | None,
        filter_value: str,
        query: str,
        page_value: float,
        size: float,
    ) -> tuple[str, list[list[str]], Any, str]:
        if not project_id:
            return (
                "プロジェクト未選択",
                [],
                gr.update(choices=[]),
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
        values, total = images.list_images(
            UUID(project_id),
            state=state,
            search=query,
            page=int(page_value),
            page_size=int(size),
        )
        choices = [
            (f"{value.original_filename} ({str(value.id)[:8]})", str(value.id))
            for value in values
        ]
        return (
            f"全{total}件",
            image_rows(values),
            gr.update(choices=choices),
            f"{total}件",
        )

    def register(project_id: str | None, files: list[str] | None) -> str:
        if not project_id:
            return "エラー: 先にプロジェクトを選択してください。"
        if not files:
            return "登録する画像を選択してください。"
        try:
            result = images.register_uploads(UUID(project_id), files)
            failed = ", ".join(f.filename for f in result.failures)
            message = f"成功: {len(result.successes)}件、失敗: {len(result.failures)}件"
            if failed:
                message += f"\n失敗ファイル: {failed}"
            return message
        except UserFacingError as exc:
            return f"エラー: {exc}"

    def change(
        project_id: str | None, ids: list[str] | None, state: SelectionState
    ) -> str:
        if not project_id or not ids:
            return "エラー: プロジェクトと画像を選択してください。"
        try:
            count = images.change_state(
                UUID(project_id), (UUID(value) for value in ids), state
            )
            return f"{count}件を{STATE_LABELS[state.value]}へ変更しました。"
        except UserFacingError as exc:
            return f"エラー: {exc}"

    upload_button.click(register, inputs=[selected_id, upload], outputs=upload_message)
    image_reload.click(
        load,
        inputs=[selected_id, state_filter, search, page, page_size],
        outputs=[project_label, image_table, image_ids, image_message],
    )
    for button, state in (
        (accepted, SelectionState.ACCEPTED),
        (pending, SelectionState.PENDING),
        (excluded, SelectionState.EXCLUDED),
    ):
        button.click(
            lambda project_id, ids, target=state: change(project_id, ids, target),
            inputs=[selected_id, image_ids],
            outputs=image_message,
        ).then(
            load,
            inputs=[selected_id, state_filter, search, page, page_size],
            outputs=[project_label, image_table, image_ids, image_message],
        )
