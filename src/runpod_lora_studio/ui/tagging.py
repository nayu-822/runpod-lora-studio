from __future__ import annotations

from typing import Any, Literal, cast
from uuid import UUID

import gradio as gr

from runpod_lora_studio.domain.models import (
    ManualCaptionPolicy,
    SelectionState,
    TagCategory,
    TaggerRunMode,
)
from runpod_lora_studio.services.caption_service import (
    CaptionEditingService,
    TagFrequencyService,
)
from runpod_lora_studio.services.image_service import ImageService
from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.tagging_service import TaggingService
from runpod_lora_studio.ui.tagging_controller import (
    TaggingController,
    preview_rows,
    preview_summary,
    tag_frequency_rows,
    tagger_run_rows,
)


def build_tagging_tab(
    service: TaggingService,
    selected_id: gr.State,
    refresh_event: gr.State | None = None,
) -> None:
    controller = TaggingController(
        service,
        TagFrequencyService(service.settings, service.projects),
        CaptionEditingService(service.settings, service.projects),
    )
    images = ImageService(service.settings, service.projects)
    draft_state = gr.State(value={})
    preview_state = gr.State(value=None)
    visible_names = gr.State(value=[])

    with gr.Row():
        adapter = gr.Textbox(
            value=service.settings.tagger_adapter_name, label="Taggerアダプター"
        )
        model = gr.Textbox(
            value=service.settings.tagger_model_identifier, label="モデル識別子"
        )
        revision = gr.Textbox(
            value=service.settings.tagger_model_revision, label="モデルリビジョン"
        )
    with gr.Row():
        model_path = gr.Textbox(
            value=str(service.settings.tagger_model_dir), label="モデル保存先"
        )
        device = gr.Dropdown(
            choices=["auto", "cuda", "cpu"],
            value=service.settings.tagger_device,
            label="デバイス",
        )
        batch_size = gr.Number(
            value=service.settings.tagger_batch_size, precision=0, label="バッチサイズ"
        )
        general_threshold = gr.Number(
            value=service.settings.tagger_general_threshold, label="general閾値"
        )
        character_threshold = gr.Number(
            value=service.settings.tagger_character_threshold, label="character閾値"
        )
    with gr.Row():
        save_rating = gr.Checkbox(
            value=service.settings.tagger_save_rating, label="ratingを保存"
        )
        save_character = gr.Checkbox(
            value=service.settings.tagger_save_character, label="characterを保存"
        )
        save_general = gr.Checkbox(
            value=service.settings.tagger_save_general, label="generalを保存"
        )
        underscore = gr.Checkbox(
            value=service.settings.tagger_underscore_to_space, label="_を空白へ変換"
        )
    mode = gr.Dropdown(
        choices=[
            ("未タグ画像のみ", TaggerRunMode.UNTAGGED_ONLY.value),
            ("失敗画像を再実行", TaggerRunMode.FAILED_ONLY.value),
            ("採用画像をすべて再実行", TaggerRunMode.ALL_ACCEPTED.value),
        ],
        value=TaggerRunMode.UNTAGGED_ONLY.value,
        label="実行モード",
    )
    with gr.Row():
        validate = gr.Button("環境確認")
        start = gr.Button("タグ付け開始", variant="primary")
        cancel = gr.Button("キャンセル")
        reload_runs = gr.Button("状態再読込")
    message = gr.Markdown()
    run_choices = gr.Dropdown(choices=[], label="TaggerRun")
    run_table = gr.Dataframe(
        headers=[
            "Run ID",
            "状態",
            "デバイス",
            "モデル",
            "対象",
            "処理済み",
            "成功",
            "失敗",
            "スキップ",
            "エラー",
        ],
        interactive=False,
        label="Tagger実行状況",
    )

    with gr.Row():
        search = gr.Textbox(label="タグ検索")
        category = gr.Dropdown(
            choices=["all", "general", "character", "rating", "meta", "unknown"],
            value="all",
            label="カテゴリ",
        )
        minimum_rate = gr.Number(value=0, minimum=0, maximum=100, label="最低出現率(%)")
        frequency_page = gr.Number(value=1, minimum=1, precision=0, label="タグページ")
    frequency_table = gr.Dataframe(
        headers=[
            "正規化タグ",
            "表示名",
            "カテゴリ",
            "出現画像数",
            "対象画像数",
            "出現率(%)",
            "平均confidence",
            "最小confidence",
            "最大confidence",
            "保持状態",
            "ルール由来",
        ],
        interactive=False,
        label="タグ頻度一覧",
    )
    tag_checks = gr.CheckboxGroup(choices=[], label="表示中のタグ（チェックあり=保持）")
    with gr.Row():
        visible_keep = gr.Button("表示中を全選択")
        visible_remove = gr.Button("表示中を全解除")
        all_keep = gr.Button("全体を全選択")
        all_remove = gr.Button("全体を全解除")
    with gr.Row():
        rate_keep = gr.Button("出現率以上を選択")
        rate_remove = gr.Button("出現率未満を解除")
        category_keep = gr.Button("指定カテゴリを選択")
        category_remove = gr.Button("指定カテゴリを解除")
        search_keep = gr.Button("検索結果を選択")
        search_remove = gr.Button("検索結果を解除")
        reset_draft = gr.Button("確定済みへ戻す")

    trigger_words = gr.Textbox(label="トリガーワード（カンマまたは改行）")
    policy = gr.Dropdown(
        choices=[
            ("手動編集を維持", ManualCaptionPolicy.KEEP_MANUAL.value),
            ("元タグから完全再構築", ManualCaptionPolicy.REBUILD_FROM_SOURCE.value),
            ("手動編集済みを除外", ManualCaptionPolicy.EXCLUDE_MANUAL.value),
        ],
        value=ManualCaptionPolicy.KEEP_MANUAL.value,
        label="手動編集済み画像の扱い",
    )
    with gr.Row():
        make_preview = gr.Button("適用前プレビュー")
        apply_preview_button = gr.Button("プレビューを適用", variant="primary")
    preview_message = gr.Markdown()
    preview_table = gr.Dataframe(
        headers=[
            "ファイル名",
            "変更前",
            "変更後",
            "追加",
            "削除",
            "トリガー",
            "方針",
            "警告",
        ],
        interactive=False,
        label="画像別キャプション差分",
    )

    gr.Markdown("### 画像単位のキャプション編集")
    image_choices = gr.Dropdown(choices=[], label="画像")
    source_tags = gr.Textbox(label="元タグ", interactive=False)
    caption_text = gr.Textbox(label="現在の最終キャプション", lines=4)
    with gr.Row():
        save_caption = gr.Button("保存")
        restore_source = gr.Button("元タグから復元")
        restore_previous = gr.Button("直前リビジョンへ戻す")
    history = gr.Dataframe(
        headers=["revision", "変更前", "変更後", "日時"], interactive=False
    )

    def configure(
        adapter_name: str,
        model_identifier: str,
        model_revision: str,
        model_dir: str,
        selected_device: str,
        selected_batch: float,
        general: float,
        character: float,
        rating: bool,
        character_save: bool,
        general_save: bool,
        underscore_space: bool,
    ) -> None:
        service.settings.tagger_adapter_name = adapter_name.strip()
        service.settings.tagger_model_identifier = model_identifier.strip()
        service.settings.tagger_model_revision = model_revision.strip()
        from pathlib import Path

        service.settings.tagger_model_dir = Path(model_dir).expanduser()
        service.settings.tagger_device = cast(
            Literal["auto", "cuda", "cpu"], selected_device
        )
        service.settings.tagger_batch_size = max(1, int(selected_batch))
        service.settings.tagger_general_threshold = max(0.0, min(1.0, general))
        service.settings.tagger_character_threshold = max(0.0, min(1.0, character))
        service.settings.tagger_save_rating = rating
        service.settings.tagger_save_character = character_save
        service.settings.tagger_save_general = general_save
        service.settings.tagger_underscore_to_space = underscore_space

    def run_start(project: str | None, selected_mode: str, *values: Any) -> str:
        if not project:
            return "エラー: プロジェクトを選択してください。"
        try:
            configure(*values)
            run = controller.start(UUID(project), TaggerRunMode(selected_mode))
            return f"TaggerRunを開始しました: {run.id}"
        except Exception as exc:
            return f"エラー: {exc}"

    def refresh_runs(project: str | None) -> tuple[list[list[str | int]], Any]:
        if not project:
            return [], gr.update(choices=[], value=None)
        runs = controller.runs(UUID(project))
        choices = [(f"{run.id} ({run.status.value})", str(run.id)) for run in runs]
        return tagger_run_rows(runs), gr.update(
            choices=choices, value=choices[0][1] if choices else None
        )

    def refresh_frequency(
        project: str | None,
        run: str | None,
        query: str,
        selected_category: str,
        rate: float,
        page: float,
        draft: dict[str, bool],
    ) -> tuple[list[list[str | int | float]], Any, list[str], dict[str, bool], str]:
        if not project:
            return [], gr.update(choices=[], value=[]), [], {}, ""
        result = controller.frequencies(
            UUID(project),
            UUID(run) if run else None,
            query,
            selected_category,
            rate / 100,
            int(page),
        )
        updated = dict(draft)
        for item in result.items:
            updated.setdefault(item.tag_name_normalized, item.keep)
        choices = [item.tag_name_normalized for item in result.items]
        selected = [name for name in choices if updated.get(name, True)]
        return (
            tag_frequency_rows(result),
            gr.update(choices=choices, value=selected),
            choices,
            updated,
            f"{result.total}タグ / 対象画像{result.target_image_count}枚",
        )

    def update_checks(
        selected: list[str] | None, names: list[str], draft: dict[str, bool]
    ) -> dict[str, bool]:
        result = dict(draft)
        selected_set = set(selected or [])
        for name in names:
            result[name] = name in selected_set
        return result

    def update_rate(
        project: str | None,
        run: str | None,
        draft: dict[str, bool],
        threshold: float,
        keep_above: bool,
    ) -> dict[str, bool]:
        if not project:
            return draft
        page = controller.frequency.list_frequencies(
            UUID(project), run_id=UUID(run) if run else None, page=1, page_size=100_000
        )
        result = dict(draft)
        for item in page.items:
            if (item.occurrence_rate * 100 >= threshold) == keep_above:
                result[item.tag_name_normalized] = keep_above
        return result

    def update_filtered(
        project: str | None,
        run: str | None,
        draft: dict[str, bool],
        query: str,
        selected_category: str,
        keep: bool,
    ) -> dict[str, bool]:
        if not project:
            return draft
        categories = (
            None if selected_category == "all" else {TagCategory(selected_category)}
        )
        page = controller.frequency.list_frequencies(
            UUID(project),
            run_id=UUID(run) if run else None,
            search=query,
            categories=categories,
            page=1,
            page_size=100_000,
        )
        return TagFrequencyService.set_visible(
            draft, (item.tag_name_normalized for item in page.items), keep
        )

    def update_all(
        project: str | None, run: str | None, draft: dict[str, bool], keep: bool
    ) -> dict[str, bool]:
        return update_filtered(project, run, draft, "", "all", keep)

    def make_preview_action(
        project: str | None,
        run: str | None,
        draft: dict[str, bool],
        triggers: str,
        selected_policy: str,
    ) -> tuple[str, list[list[str]], Any]:
        if not project or not run:
            return "エラー: プロジェクトとTaggerRunを選択してください。", [], None
        try:
            result = controller.preview(
                UUID(project),
                UUID(run),
                draft,
                triggers,
                ManualCaptionPolicy(selected_policy),
            )
            return preview_summary(result), preview_rows(result), result
        except Exception as exc:
            return f"エラー: {exc}", [], None

    def apply_action(preview: Any) -> str:
        if preview is None:
            return "エラー: 先にプレビューを生成してください。"
        try:
            controller.apply(preview)
            return "プレビューを適用しました。"
        except Exception as exc:
            return f"エラー: {exc}"

    def cancel_action(run: str | None) -> str:
        if not run:
            return "エラー: TaggerRunを選択してください。"
        try:
            controller.cancel(UUID(run))
            return "キャンセルを要求しました。"
        except Exception as exc:
            return f"エラー: {exc}"

    def load_images(project: str | None) -> Any:
        if not project:
            return gr.update(choices=[], value=None)
        values, _ = images.list_images(
            UUID(project), state=SelectionState.ACCEPTED, page=1, page_size=100
        )
        choices = [
            (f"{item.original_filename} ({str(item.id)[:8]})", str(item.id))
            for item in values
        ]
        return gr.update(choices=choices, value=choices[0][1] if choices else None)

    def load_caption(
        project: str | None, image: str | None
    ) -> tuple[str, str, list[list[str]]]:
        if not project or not image:
            return "", "", []
        try:
            caption = controller.captions.get_caption(UUID(project), UUID(image))
            raw = controller.captions.source_tags_text(UUID(project), UUID(image))
            if caption:
                text = caption.caption_text
            else:
                text = ""
            rows = [
                [
                    str(item.new_revision),
                    item.before_text,
                    item.after_text,
                    str(item.created_at),
                ]
                for item in controller.captions.history(UUID(project), UUID(image))
            ]
            return raw, text, rows
        except Exception as exc:
            return f"エラー: {exc}", "", []

    def save_caption_action(project: str | None, image: str | None, text: str) -> str:
        if not project or not image:
            return "エラー: プロジェクトと画像を選択してください。"
        try:
            controller.captions.save_image_caption(UUID(project), UUID(image), text)
            return "キャプションを保存しました。"
        except (ValueError, UserFacingError) as exc:
            return f"エラー: {exc}"

    validate.click(lambda: controller.validate(), outputs=[message])
    start.click(
        run_start,
        inputs=[
            selected_id,
            mode,
            adapter,
            model,
            revision,
            model_path,
            device,
            batch_size,
            general_threshold,
            character_threshold,
            save_rating,
            save_character,
            save_general,
            underscore,
        ],
        outputs=[message],
    ).then(refresh_runs, inputs=[selected_id], outputs=[run_table, run_choices])
    cancel.click(cancel_action, inputs=[run_choices], outputs=[message])
    reload_runs.click(
        refresh_runs, inputs=[selected_id], outputs=[run_table, run_choices]
    )
    run_choices.change(
        refresh_frequency,
        inputs=[
            selected_id,
            run_choices,
            search,
            category,
            minimum_rate,
            frequency_page,
            draft_state,
        ],
        outputs=[frequency_table, tag_checks, visible_names, draft_state, message],
    )
    for component in (search, category, minimum_rate, frequency_page):
        component.change(
            refresh_frequency,
            inputs=[
                selected_id,
                run_choices,
                search,
                category,
                minimum_rate,
                frequency_page,
                draft_state,
            ],
            outputs=[frequency_table, tag_checks, visible_names, draft_state, message],
        )
    tag_checks.change(
        update_checks,
        inputs=[tag_checks, visible_names, draft_state],
        outputs=[draft_state],
    )
    visible_keep.click(
        lambda names, draft: TagFrequencyService.set_visible(draft, names, True),
        inputs=[visible_names, draft_state],
        outputs=[draft_state],
    )
    visible_remove.click(
        lambda names, draft: TagFrequencyService.set_visible(draft, names, False),
        inputs=[visible_names, draft_state],
        outputs=[draft_state],
    )
    all_keep.click(
        lambda project, run, draft: update_all(project, run, draft, True),
        inputs=[selected_id, run_choices, draft_state],
        outputs=[draft_state],
    )
    all_remove.click(
        lambda project, run, draft: update_all(project, run, draft, False),
        inputs=[selected_id, run_choices, draft_state],
        outputs=[draft_state],
    )
    rate_keep.click(
        lambda project, run, draft, threshold: update_rate(
            project, run, draft, threshold, True
        ),
        inputs=[selected_id, run_choices, draft_state, minimum_rate],
        outputs=[draft_state],
    )
    rate_remove.click(
        lambda project, run, draft, threshold: update_rate(
            project, run, draft, threshold, False
        ),
        inputs=[selected_id, run_choices, draft_state, minimum_rate],
        outputs=[draft_state],
    )
    category_keep.click(
        lambda project, run, draft, selected: update_filtered(
            project, run, draft, "", selected, True
        ),
        inputs=[selected_id, run_choices, draft_state, category],
        outputs=[draft_state],
    )
    category_remove.click(
        lambda project, run, draft, selected: update_filtered(
            project, run, draft, "", selected, False
        ),
        inputs=[selected_id, run_choices, draft_state, category],
        outputs=[draft_state],
    )
    search_keep.click(
        lambda project, run, draft, query: update_filtered(
            project, run, draft, query, "all", True
        ),
        inputs=[selected_id, run_choices, draft_state, search],
        outputs=[draft_state],
    )
    search_remove.click(
        lambda project, run, draft, query: update_filtered(
            project, run, draft, query, "all", False
        ),
        inputs=[selected_id, run_choices, draft_state, search],
        outputs=[draft_state],
    )
    reset_draft.click(
        lambda project: controller.frequency.rules(UUID(project)) if project else {},
        inputs=[selected_id],
        outputs=[draft_state],
    )
    make_preview.click(
        make_preview_action,
        inputs=[selected_id, run_choices, draft_state, trigger_words, policy],
        outputs=[preview_message, preview_table, preview_state],
    )
    apply_preview_button.click(
        apply_action, inputs=[preview_state], outputs=[preview_message]
    )
    selected_id.change(load_images, inputs=[selected_id], outputs=[image_choices])
    image_choices.change(
        load_caption,
        inputs=[selected_id, image_choices],
        outputs=[source_tags, caption_text, history],
    )
    save_caption.click(
        save_caption_action,
        inputs=[selected_id, image_choices, caption_text],
        outputs=[message],
    ).then(
        load_caption,
        inputs=[selected_id, image_choices],
        outputs=[source_tags, caption_text, history],
    )
    restore_source.click(
        lambda project, image: (
            controller.restore_source(UUID(project), UUID(image))
            if project and image
            else ""
        ),
        inputs=[selected_id, image_choices],
        outputs=[caption_text],
    )
    restore_previous.click(
        lambda project, image: (
            controller.restore_previous(UUID(project), UUID(image))
            if project and image
            else ""
        ),
        inputs=[selected_id, image_choices],
        outputs=[caption_text],
    )
