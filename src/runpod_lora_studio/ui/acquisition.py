from __future__ import annotations

from typing import Any
from uuid import UUID

import gradio as gr

from runpod_lora_studio.domain.acquisition_models import (
    DanbooruSearchCriteria,
    ImageRating,
    ImageSearchSort,
)
from runpod_lora_studio.services.acquisition_download_service import (
    AcquisitionDownloadError,
    ImageAcquisitionDownloadService,
)
from runpod_lora_studio.services.acquisition_service import (
    AcquisitionValidationError,
    ImageAcquisitionService,
    SearchCandidateView,
)

RATING_CHOICES = [
    ("一般 (general)", ImageRating.GENERAL.value),
    ("センシティブ (sensitive)", ImageRating.SENSITIVE.value),
    ("問題あり (questionable)", ImageRating.QUESTIONABLE.value),
    ("成人向け (explicit)", ImageRating.EXPLICIT.value),
]
EXENSION_CHOICES = [".jpg", ".jpeg", ".png", ".webp"]
EXCLUSION_LABELS = {
    "MISSING_FILE_URL": "ファイルURLがありません",
    "INVALID_FILE_URL": "ファイルURLを検証できません",
    "INVALID_FILE_HOST": "許可されていないファイルホストです",
    "UNSUPPORTED_FILE_TYPE": "対応していない拡張子です",
    "RATING_NOT_ALLOWED": "ratingが許可条件外です",
    "SCORE_BELOW_MINIMUM": "最低スコア未満です",
    "WIDTH_BELOW_MINIMUM": "最低幅未満です",
    "HEIGHT_BELOW_MINIMUM": "最低高さ未満です",
    "PIXEL_COUNT_BELOW_MINIMUM": "最低画素数未満です",
    "POST_DELETED": "削除済み投稿です",
    "POST_PENDING": "保留中の投稿です",
    "POST_FLAGGED": "フラグ付き投稿です",
    "ALREADY_IMPORTED": "既に取り込み済みです",
    "ALREADY_PLANNED": "既に取得計画済みです",
    "INVALID_METADATA": "メタデータが不正です",
}


def _tags(value: str | None) -> tuple[str, ...]:
    return tuple(item for item in (value or "").replace(",", " ").split() if item)


def _candidate_rows(items: list[SearchCandidateView]) -> list[list[str]]:
    return [
        [
            str(item.result_id)[:8],
            item.external_post_id,
            item.rating,
            item.score,
            item.resolution,
            item.extension,
            item.tags,
            "、".join(
                EXCLUSION_LABELS.get(reason, "候補を利用できません")
                for reason in item.exclusion_reasons
            )
            if item.exclusion_reasons
            else "利用可能",
            "選択" if item.selected else "未選択",
        ]
        for item in items
    ]


def _status_text(service: ImageAcquisitionService, search_id: str | None) -> str:
    if not search_id:
        return "検索条件を入力して検索を開始してください。"
    record = service.get_search(UUID(search_id))
    if record is None:
        return "検索結果が見つかりません。"
    detail = (
        f"状態: {record.status} / API {record.api_request_count}回 / "
        f"候補 {record.returned_post_count}件"
    )
    if record.error_code:
        detail += f" / エラー: {record.error_code}"
    return detail


def build_acquisition_tab(
    service: ImageAcquisitionService,
    selected_project: gr.State,
    download_service: ImageAcquisitionDownloadService | None = None,
) -> None:
    """Build the Phase 8A metadata search and immutable plan UI."""

    search_id = gr.State(value=None)
    plan_state = gr.State(value=None)
    with gr.Row():
        include_tags = gr.Textbox(label="必須タグ", placeholder="1girl, solo")
        exclude_tags = gr.Textbox(label="除外タグ", placeholder="gore, text")
    with gr.Row():
        ratings = gr.CheckboxGroup(
            choices=RATING_CHOICES, label="許可するrating", value=[]
        )
        extensions = gr.CheckboxGroup(
            choices=EXENSION_CHOICES,
            label="許可する拡張子",
            value=list(EXENSION_CHOICES),
        )
        sort_rule = gr.Dropdown(
            choices=[
                ("スコア順", ImageSearchSort.SCORE.value),
                ("ID順", ImageSearchSort.ID.value),
            ],
            value=ImageSearchSort.SCORE.value,
            label="並び順",
        )
    with gr.Row():
        minimum_score = gr.Number(label="最低スコア", value=None, precision=0)
        minimum_width = gr.Number(label="最低幅", value=None, precision=0)
        minimum_height = gr.Number(label="最低高さ", value=None, precision=0)
        minimum_pixels = gr.Number(label="最低画素数", value=None, precision=0)
    with gr.Row():
        maximum_candidates = gr.Number(label="最大候補数", value=100, precision=0)
        page_size = gr.Number(label="1ページ件数", value=20, precision=0)
        search_button = gr.Button("検索", variant="primary")
        cancel_button = gr.Button("検索をキャンセル")
    status = gr.Markdown()
    candidate_table = gr.Dataframe(
        headers=[
            "候補ID",
            "投稿ID",
            "rating",
            "score",
            "解像度",
            "拡張子",
            "タグ",
            "除外理由",
            "選択",
        ],
        interactive=False,
        label="取得候補（画像本体はPhase 8Aでは表示・取得しません）",
    )
    candidate_choices = gr.CheckboxGroup(choices=[], label="取得計画に含める候補")
    with gr.Row():
        refresh_button = gr.Button("候補を更新")
        select_all_button = gr.Button("利用可能候補を全選択")
        clear_selection_button = gr.Button("選択解除")
        preview_button = gr.Button("取得計画をプレビュー")
        confirm_button = gr.Button("取得計画を確定", variant="primary")
    plan_message = gr.Markdown()
    download_plan_state = gr.State(value=None)
    download_job_state = gr.State(value=None)

    download_status = None
    download_items = None
    download_plan_id = None
    if download_service is not None:
        gr.Markdown("### 確定済み計画の画像取得")
        download_plan_id = gr.Textbox(
            label="確定済み計画ID（確認用）",
            interactive=False,
        )
        with gr.Row():
            download_start = gr.Button("画像取得を開始", variant="primary")
            download_cancel = gr.Button("取得をキャンセル")
            download_resume = gr.Button("停止・失敗項目を再開")
            download_refresh = gr.Button("取得状態を更新")
        download_status = gr.Markdown()
        download_items = gr.Dataframe(
            headers=[
                "項目ID",
                "投稿ID",
                "状態",
                "試行回数",
                "受信bytes",
                "予定bytes",
                "形式",
                "寸法",
                "SHA-256",
                "失敗コード",
                "再試行可",
            ],
            interactive=False,
            label="取得項目（URL・保存先は表示しません）",
        )

    def query_values(
        project_id: str | None,
        include: str,
        exclude: str,
        selected_ratings: list[str] | None,
        selected_extensions: list[str] | None,
        selected_sort: str,
        score: float | None,
        width: float | None,
        height: float | None,
        pixels: float | None,
        maximum: float,
        page: float,
    ) -> DanbooruSearchCriteria:
        if not project_id:
            raise AcquisitionValidationError("プロジェクトを選択してください")
        return DanbooruSearchCriteria(
            project_id=UUID(project_id),
            include_tags=_tags(include),
            exclude_tags=_tags(exclude),
            ratings=tuple(ImageRating(value) for value in (selected_ratings or [])),
            required_extensions=tuple(selected_extensions or []),
            minimum_score=int(score) if score is not None else None,
            minimum_width=int(width) if width is not None else None,
            minimum_height=int(height) if height is not None else None,
            minimum_pixel_count=int(pixels) if pixels is not None else None,
            maximum_candidate_count=int(maximum),
            page_size=int(page),
            sort_rule=ImageSearchSort(selected_sort),
        )

    def start(
        project_id: str | None,
        include: str,
        exclude: str,
        selected_ratings: list[str] | None,
        selected_extensions: list[str] | None,
        selected_sort: str,
        score: float | None,
        width: float | None,
        height: float | None,
        pixels: float | None,
        maximum: float,
        page: float,
    ) -> tuple[str, Any, Any, Any]:
        try:
            search = service.start_search(
                query_values(
                    project_id,
                    include,
                    exclude,
                    selected_ratings,
                    selected_extensions,
                    selected_sort,
                    score,
                    width,
                    height,
                    pixels,
                    maximum,
                    page,
                )
            )
            return (
                _status_text(service, str(search)),
                str(search),
                [],
                gr.update(choices=[], value=[]),
            )
        except (AcquisitionValidationError, ValueError) as exc:
            return f"エラー: {exc}", None, [], gr.update(choices=[], value=[])

    def refresh(current_search: str | None) -> tuple[str, list[list[str]], Any]:
        if not current_search:
            return _status_text(service, None), [], gr.update(choices=[], value=[])
        items = service.list_candidates(UUID(current_search))
        choices = [
            (f"投稿 {item.external_post_id} ({item.rating})", str(item.result_id))
            for item in items
            if not item.exclusion_reasons
        ]
        selected = [str(item.result_id) for item in items if item.selected]
        return (
            _status_text(service, current_search),
            _candidate_rows(items),
            gr.update(choices=choices, value=selected),
        )

    def cancel(current_search: str | None) -> str:
        if current_search:
            service.cancel_search(UUID(current_search))
        return _status_text(service, current_search)

    def set_all(current_search: str | None, selected: list[str] | None) -> list[str]:
        if not current_search:
            return []
        service.select_all_available(UUID(current_search), True)
        return [
            str(item.result_id)
            for item in service.list_candidates(UUID(current_search))
            if not item.exclusion_reasons
        ]

    def clear_all(current_search: str | None) -> list[str]:
        if current_search:
            service.select_all_available(UUID(current_search), False)
        return []

    def selection_changed(current: list[str] | None) -> None:
        service.set_selection([UUID(item) for item in (current or [])], True)

    def preview(
        current_search: str | None, selected: list[str] | None
    ) -> tuple[str, Any]:
        if not current_search:
            return "エラー: 検索結果を選択してください。", None
        try:
            value = service.preview_plan(
                UUID(current_search), [UUID(item) for item in (selected or [])]
            )
            return (
                "取得計画プレビュー: "
                f"{len(value.items)}件（確定すると計画だけを保存します）",
                value,
            )
        except (AcquisitionValidationError, ValueError) as exc:
            return f"エラー: {exc}", None

    def confirm(value: Any) -> str | tuple[str, str | None, str | None]:
        if value is None:
            message = "エラー: 先に取得計画をプレビューしてください。"
            return (message, None, None) if download_service else message
        try:
            plan_id = service.confirm_plan(value)
        except (AcquisitionValidationError, ValueError) as exc:
            message = f"エラー: {exc}"
            return (message, None, None) if download_service else message
        message = f"確定済み計画: {str(plan_id)[:8]} / 画像本体の取得を開始できます。"
        return (message, str(plan_id), str(plan_id)) if download_service else message

    def _download_rows(current_job: UUID) -> list[list[str]]:
        if download_service is None:
            return []
        return [
            [
                str(item.id)[:8],
                item.external_post_id,
                item.status.value,
                str(item.attempt_count),
                str(item.received_bytes),
                str(item.expected_file_size or ""),
                item.detected_format or "",
                f"{item.detected_width or '?'}x{item.detected_height or '?'}",
                item.sha256_prefix or "",
                item.failure_code or "",
                "yes" if item.retryable else "no",
            ]
            for item in download_service.list_items(current_job)
        ]

    def _download_status(current_job: str | None) -> tuple[str, list[list[str]]]:
        if download_service is None or not current_job:
            return "取得ジョブはまだ開始されていません。", []
        try:
            job_id = UUID(current_job)
        except ValueError:
            return "取得ジョブIDが不正です。", []
        job = download_service.get_job(job_id)
        if job is None:
            return "取得ジョブが見つかりません。", []
        detail = (
            f"状態: `{job.status.value}` / 成功: {job.imported_count} / 既存link: "
            f"{job.linked_existing_count} / skip: {job.skipped_count} / "
            f"失敗: {job.failed_count} / "
            f"受信: {job.received_bytes} bytes"
        )
        if job.error_code:
            detail += f" / failure: `{job.error_code}`"
        return detail, _download_rows(job_id)

    def start_download(
        plan_value: str | None,
    ) -> tuple[str, list[list[str]], str | None]:
        if download_service is None or not plan_value:
            return "確定済み計画を先に作成してください。", [], None
        try:
            job_id = download_service.start_job(UUID(plan_value))
            status_text, rows = _download_status(str(job_id))
            return status_text, rows, str(job_id)
        except (AcquisitionDownloadError, ValueError) as exc:
            code = getattr(exc, "code", None)
            return (
                f"取得を開始できません: `{getattr(code, 'value', str(exc))}`",
                [],
                None,
            )

    def cancel_download(job_value: str | None) -> tuple[str, list[list[str]]]:
        if download_service is not None and job_value:
            download_service.cancel_job(UUID(job_value))
        return _download_status(job_value)

    def resume_download(job_value: str | None) -> tuple[str, list[list[str]]]:
        if download_service is None or not job_value:
            return _download_status(job_value)
        try:
            download_service.resume_job(UUID(job_value))
        except (AcquisitionDownloadError, ValueError) as exc:
            code = getattr(exc, "code", None)
            return f"取得を再開できません: `{getattr(code, 'value', str(exc))}`", []
        return _download_status(job_value)

    search_button.click(
        start,
        inputs=[
            selected_project,
            include_tags,
            exclude_tags,
            ratings,
            extensions,
            sort_rule,
            minimum_score,
            minimum_width,
            minimum_height,
            minimum_pixels,
            maximum_candidates,
            page_size,
        ],
        outputs=[status, search_id, candidate_table, candidate_choices],
    )
    refresh_button.click(
        refresh,
        inputs=[search_id],
        outputs=[status, candidate_table, candidate_choices],
    )
    cancel_button.click(cancel, inputs=[search_id], outputs=[status])
    select_all_button.click(
        set_all, inputs=[search_id, candidate_choices], outputs=[candidate_choices]
    )
    clear_selection_button.click(
        clear_all, inputs=[search_id], outputs=[candidate_choices]
    )
    candidate_choices.change(selection_changed, inputs=[candidate_choices], outputs=[])
    preview_button.click(
        preview,
        inputs=[search_id, candidate_choices],
        outputs=[plan_message, plan_state],
    )
    confirm_button.click(
        confirm,
        inputs=[plan_state],
        outputs=[plan_message, download_plan_state, download_plan_id]
        if download_service is not None
        else [plan_message],
    )
    if download_service is not None:
        download_start.click(
            start_download,
            inputs=[download_plan_state],
            outputs=[download_status, download_items, download_job_state],
        )
        download_cancel.click(
            cancel_download,
            inputs=[download_job_state],
            outputs=[download_status, download_items],
        )
        download_resume.click(
            resume_download,
            inputs=[download_job_state],
            outputs=[download_status, download_items],
        )
        download_refresh.click(
            _download_status,
            inputs=[download_job_state],
            outputs=[download_status, download_items],
        )

    def clear_on_project_change(_: str | None) -> tuple[str, Any, Any, Any]:
        return (
            "プロジェクトが変わったため、検索結果をクリアしました。",
            None,
            [],
            gr.update(choices=[], value=[]),
        )

    selected_project.change(
        clear_on_project_change,
        inputs=[selected_project],
        outputs=[status, search_id, candidate_table, candidate_choices],
    )
