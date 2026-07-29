from __future__ import annotations

from typing import Any
from uuid import UUID

import gradio as gr

from runpod_lora_studio.domain.recommendation_models import (
    QualityProfile,
    RecommendationInput,
    SpeedProfile,
)
from runpod_lora_studio.domain.training_performance_models import (
    GpuCalibrationExclusionReason,
)
from runpod_lora_studio.services.dataset_statistics_service import (
    DatasetStatisticsService,
)
from runpod_lora_studio.services.environment_diagnostic_service import (
    ComputeEnvironmentService,
    TrainingEnvironmentService,
)
from runpod_lora_studio.services.project_service import UserFacingError
from runpod_lora_studio.services.recommendation_application_service import (
    RecommendationApplicationService,
    RecommendationStaleError,
)
from runpod_lora_studio.services.recommendation_calibration_service import (
    RecommendationHistoryService,
    TrainingCalibrationService,
)
from runpod_lora_studio.services.recommendation_persistence_service import (
    RecommendationPersistenceService,
)
from runpod_lora_studio.services.training_recommendation_engine import (
    RuleBasedRecommendationEngine,
)
from runpod_lora_studio.services.training_service import TrainingService
from runpod_lora_studio.ui.training_controller import TrainingController


def clear_recommendation_state(*_: object) -> tuple[None, str, str, object]:
    return None, "", "manual", gr.update(interactive=False)


def mark_recommendation_edited(mode: str | None) -> str:
    return (
        "recommended_edited"
        if mode in {"recommended", "recommended_edited"}
        else "manual"
    )


_EXCLUSION_REASON_LABELS = {
    GpuCalibrationExclusionReason.GPU_CHANGED_DURING_JOB.value: (
        "学習中にGPUが変更されました"
    ),
    GpuCalibrationExclusionReason.GPU_IDENTITY_UNVERIFIED.value: (
        "GPU identityを検証できませんでした"
    ),
    GpuCalibrationExclusionReason.PHYSICAL_GPU_NOT_FOUND.value: (
        "対応する物理GPUを取得できませんでした"
    ),
    GpuCalibrationExclusionReason.AMBIGUOUS_GPU_SELECTION.value: (
        "使用GPUを一意に特定できませんでした"
    ),
    GpuCalibrationExclusionReason.TARGET_PROCESS_GPU_NOT_FOUND.value: (
        "学習プロセスが使用したGPUを確認できませんでした"
    ),
    GpuCalibrationExclusionReason.SELECTED_GPU_MEMORY_MISMATCH.value: (
        "実行GPUとメモリ測定GPUが一致しません"
    ),
    "job_not_succeeded": "学習が正常終了しませんでした",
    "steps_missing": "学習ステップを確認できませんでした",
    "steps_invalid": "学習ステップが不正です",
    "elapsed_missing": "経過時間を確認できませんでした",
    "speed_measurement_missing": "速度を測定できませんでした",
    "resume_offset_ambiguous": "再開位置を特定できませんでした",
    "progress_parser_warning": "進捗を正確に解析できませんでした",
    "gpu_environment_changed_since_recommendation": "推奨時からGPU環境が変化しました",
}


def _format_exclusion_reasons(reasons: tuple[str, ...]) -> str:
    return (
        "、".join(
            _EXCLUSION_REASON_LABELS.get(reason, "校正対象外") for reason in reasons
        )
        or "なし"
    )


def build_training_tab(service: TrainingService, selected_project: gr.State) -> None:
    controller = TrainingController(service)
    compute_diagnostics = ComputeEnvironmentService(service.settings)
    training_diagnostics = TrainingEnvironmentService(service.settings)
    dataset_statistics = DatasetStatisticsService(service.settings)
    recommendation_service = RecommendationPersistenceService(service.settings)
    history_service = RecommendationHistoryService(service.settings)
    calibration_service = TrainingCalibrationService(service.settings)
    application_service = RecommendationApplicationService(
        service.settings,
        training_service=service,
        persistence=recommendation_service,
        compute_service=compute_diagnostics,
        training_environment_service=training_diagnostics,
        dataset_statistics=dataset_statistics,
    )
    recommendation_engine = RuleBasedRecommendationEngine()
    gr.Markdown("### Phase 7A 実行環境診断・推奨設定")
    with gr.Row():
        concept_type = gr.Dropdown(
            choices=[
                "character",
                "style",
                "outfit",
                "object",
                "pose",
                "general_concept",
            ],
            value="character",
            label="concept type",
        )
        quality_profile = gr.Dropdown(
            choices=["conservative", "balanced", "detail_focused"],
            value="balanced",
            label="quality profile",
        )
        speed_profile = gr.Dropdown(
            choices=["memory_saver", "balanced", "speed_priority"],
            value="balanced",
            label="speed profile",
        )
        diagnose = gr.Button("環境を診断")
    diagnostic_view = gr.Markdown()
    recommend = gr.Button("学習設定を推奨")
    recommendation_view = gr.Markdown()
    recommendation_state = gr.State(value=None)
    recommendation_mode = gr.State(value="manual")

    def recommend_action(
        project_id: str | None,
        snapshot_choice: str | None,
        model_choice: str | None,
        concept: str,
        quality: str,
        speed: str,
        current_resolution: float,
    ) -> tuple[str, dict[str, object] | None, str, object]:
        if project_id is None or snapshot_choice is None or model_choice is None:
            return (
                "project, snapshot, and model are required",
                None,
                "manual",
                gr.update(interactive=False),
            )
        try:
            snapshot_id = UUID(snapshot_choice.split("|", 1)[0].strip())
            model_id = UUID(model_choice.split("|", 1)[0].strip())
            _, model_sha256, model_hash_verified = recommendation_service.model_context(
                model_id
            )
            compute = compute_diagnostics.detect()
            training = training_diagnostics.detect()
            environment_snapshot_id = compute_diagnostics.snapshot(UUID(project_id))
            training_diagnostics.snapshot(
                UUID(project_id), compute_snapshot_id=environment_snapshot_id
            )
            dataset = dataset_statistics.calculate(snapshot_id)
            data = RecommendationInput(
                project_id=UUID(project_id),
                dataset_snapshot_id=snapshot_id,
                model_id=model_id,
                environment_snapshot_id=environment_snapshot_id,
                environment=compute,
                training_environment=training,
                dataset=dataset,
                concept_type=concept,
                quality_profile=QualityProfile(quality),
                speed_profile=SpeedProfile(speed),
                user_constraints={
                    "model_sha256": model_sha256,
                    "model_hash_verified": model_hash_verified,
                },
                current_config={"resolution": int(current_resolution)},
            )
            recommendation = recommendation_engine.recommend(data)[0]
            request = recommendation_service.save(data, (recommendation,))
            blocking_codes = [
                warning.code
                for warning in recommendation.warnings
                if warning.severity.value == "blocking"
            ]
            state: dict[str, object] = {
                "recommendation_id": str(recommendation.id),
                "request_id": str(request.id),
                "input_fingerprint": request.input_fingerprint,
                "engine_version": recommendation.engine_version,
                "warning_codes": [warning.code for warning in recommendation.warnings],
                "blocking_warning_codes": blocking_codes,
            }
            lines = [
                "#### 推奨結果（適用しても学習は開始しません）",
                f"- batch: `{recommendation.batch_size}` / "
                f"epochs: `{recommendation.epochs}`",
                f"- dim/alpha: `{recommendation.network_dim}/"
                f"{recommendation.network_alpha}`",
                f"- precision: `{recommendation.mixed_precision}` / "
                f"optimizer: `{recommendation.optimizer}`",
                f"- 推定総step: `{recommendation.estimated_total_steps}` / "
                f"VRAM: `{recommendation.estimated_vram_bytes}` bytes",
            ]
            lines.extend(
                f"- {warning.severity.value}: {warning.code} — {warning.message}"
                for warning in recommendation.warnings
            )
            return (
                "\n".join(lines),
                state,
                "recommended",
                gr.update(interactive=not blocking_codes),
            )
        except (OSError, ValueError) as exc:
            return (
                f"recommendation error: {exc}",
                None,
                "manual",
                gr.update(interactive=False),
            )

    apply_recommendation = gr.Button("推奨値を入力欄へ反映")
    manual_mode = gr.Button("推奨を解除して手動設定")

    def apply_action(
        state: dict[str, object] | None,
        mode: str | None,
    ) -> tuple[object, ...]:
        if (
            mode not in {"recommended", "recommended_edited"}
            or not state
            or not state.get("recommendation_id")
        ):
            return (gr.update(),) * 11 + ("recommendation result is missing",)
        try:
            recommendation = application_service.preview(
                UUID(str(state["recommendation_id"])),
                input_fingerprint_value=str(state["input_fingerprint"]),
            )
        except (ValueError, RecommendationStaleError) as exc:
            return (gr.update(),) * 11 + (f"recommendation cannot be applied: {exc}",)
        return (
            recommendation.resolution,
            recommendation.batch_size,
            recommendation.epochs,
            recommendation.learning_rate,
            recommendation.optimizer,
            recommendation.scheduler,
            recommendation.network_module,
            recommendation.network_dim,
            recommendation.network_alpha,
            recommendation.mixed_precision,
            "推奨値を入力欄へ反映しました。内容を確認してから設定を保存してください。",
        )

    def diagnose_action(
        project_id: str | None, concept: str, quality: str, speed: str
    ) -> str:
        del concept, quality, speed
        try:
            compute = compute_diagnostics.detect()
            training = training_diagnostics.detect()
            if project_id:
                compute_diagnostics.snapshot(UUID(project_id))
                training_diagnostics.snapshot(UUID(project_id))
            gpu = compute.gpu_devices[0] if compute.gpu_devices else None
            gpu_text = (
                "未検出"
                if gpu is None
                else f"{gpu.name} / VRAM {gpu.total_vram_bytes or 0} bytes"
            )
            return "\n".join(
                [
                    "#### 診断結果",
                    f"- GPU: {gpu_text}",
                    f"- CUDA: {'利用可能' if compute.cuda_available else '利用不可'}",
                    f"- bf16: {compute.bf16_supported}",
                    f"- sd-scripts: {'利用可能' if not training.errors else '要確認'}",
                    f"- 警告: {len(compute.warnings) + len(training.warnings)} / "
                    f"エラー: {len(compute.errors) + len(training.errors)}",
                    "推奨設定の適用後も、既存のTrainingConfig検証を通過させてからジョブ作成してください。",
                ]
            )
        except (OSError, ValueError) as exc:
            return f"診断エラー: {exc}"

    diagnose_event = diagnose.click(
        diagnose_action,
        inputs=[selected_project, concept_type, quality_profile, speed_profile],
        outputs=[diagnostic_view],
    )
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
    recommend.click(
        recommend_action,
        inputs=[
            selected_project,
            snapshot,
            model,
            concept_type,
            quality_profile,
            speed_profile,
            resolution,
        ],
        outputs=[
            recommendation_view,
            recommendation_state,
            recommendation_mode,
            apply_recommendation,
        ],
    )
    apply_recommendation.click(
        apply_action,
        inputs=[recommendation_state, recommendation_mode],
        outputs=[
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
            message,
        ],
    )
    manual_mode.click(
        lambda: (None, "", "manual", gr.update(interactive=False)),
        outputs=[
            recommendation_state,
            recommendation_view,
            recommendation_mode,
            apply_recommendation,
        ],
    )
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

    def save_action(
        project_id: str | None,
        state: dict[str, object] | None,
        mode: str | None,
        *values: Any,
    ) -> str | None:
        if not project_id:
            return "プロジェクトを選択してください"
        try:
            base = controller.config_input(project_id, values)
            if mode in {"recommended", "recommended_edited"}:
                if not state or not state.get("recommendation_id"):
                    return (
                        "recommendation state is missing; generate a new recommendation"
                    )
                config = application_service.apply(
                    UUID(str(state["recommendation_id"])),
                    base,
                    input_fingerprint_value=str(state["input_fingerprint"]),
                )
                return str(config.id)
            if state:
                return "recommendation state and mode are inconsistent"
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

    performance_view = gr.Markdown(label="Phase 7B performance")
    history_table = gr.Dataframe(
        headers=[
            "job",
            "status",
            "GPU",
            "resolution",
            "steps/sec",
            "peak VRAM",
            "OOM",
            "included",
            "exclusion",
        ],
        value=[],
        interactive=False,
        label="Phase 7B learning performance history",
    )
    with gr.Row():
        recollect_performance = gr.Button("performanceを再収集")
        rebuild_calibration = gr.Button("calibrationを再構築")
        refresh_history = gr.Button("performance履歴を更新")

    def performance_action(current_job: str | None) -> str:
        if not current_job:
            return "jobを選択してください"
        try:
            summary = service.performance_collector.collect(
                UUID(current_job.strip()), force=True
            )
            return "\n".join(
                [
                    "#### empirical training performance",
                    f"- status: `{summary.job_result_status}` / "
                    f"failure: `{summary.failure_category.value}`",
                    f"- steps/sec: `{summary.measured_steps_per_second}` / "
                    f"elapsed: `{summary.elapsed_seconds}`",
                    f"- peak VRAM: `{summary.peak_reserved_vram_bytes}` bytes / "
                    f"samples: `{summary.memory_sample_count}`",
                    f"- OOM: `{summary.oom_detected}` / "
                    f"speed calibration: `{summary.usable_for_speed_calibration}`",
                    f"- 校正除外理由: "
                    f"{_format_exclusion_reasons(summary.exclusion_reasons)}",
                ]
            )
        except (OSError, ValueError) as exc:
            return f"performance collection error: {exc}"

    def history_action(project_id: str | None) -> list[list[str]]:
        summaries = history_service.list(UUID(project_id) if project_id else None)
        return [
            [
                str(item.training_job_id),
                item.job_result_status,
                item.gpu_identity_fingerprint or "unknown",
                str(item.resolution or ""),
                str(item.measured_steps_per_second or ""),
                str(item.peak_reserved_vram_bytes or ""),
                str(item.oom_detected),
                str(item.calibration_included),
                _format_exclusion_reasons(item.exclusion_reasons),
            ]
            for item in summaries
        ]

    def rebuild_calibration_action(
        project_id: str | None, current_job: str | None
    ) -> str:
        if not project_id or not current_job:
            return "projectとjobを選択してください"
        try:
            summary = service.performance_collector.collect(
                UUID(current_job.strip()), force=True
            )
            if not summary.gpu_identity_fingerprint:
                return "GPU identityが取得できないためbaselineを維持します"
            snapshot = calibration_service.build_for_project(
                UUID(project_id),
                gpu_identity_fingerprint=summary.gpu_identity_fingerprint,
                gpu_total_vram_bytes=summary.gpu_total_vram_bytes,
                resolution=summary.resolution,
                optimizer=summary.optimizer,
                mixed_precision=summary.mixed_precision,
                cache_latents=summary.cache_latents,
                gradient_checkpointing=summary.gradient_checkpointing,
                gpu_architecture=summary.gpu_architecture,
                batch_size=summary.batch_size,
                gradient_accumulation_steps=summary.gradient_accumulation_steps,
                effective_batch_size=summary.effective_batch_size,
                network_module=summary.network_module,
                network_dim=summary.network_dim,
                network_alpha=summary.network_alpha,
                world_size=summary.world_size,
                sd_scripts_version=summary.sd_scripts_version,
                xformers_available=summary.xformers_available,
            )
            return (
                f"calibration rebuilt: confidence=`{snapshot.confidence.value}`, "
                f"samples=`{snapshot.sample_count}`, "
                f"OOM=`{snapshot.oom_sample_count}`; "
                "再構築だけでは学習を開始しません"
            )
        except (OSError, ValueError) as exc:
            return f"calibration error: {exc}"

    project_change = selected_project.change(
        choices, inputs=[selected_project], outputs=[snapshot, model]
    )
    project_change.then(
        clear_recommendation_state,
        outputs=[
            recommendation_state,
            recommendation_view,
            recommendation_mode,
            apply_recommendation,
        ],
    )

    for recommendation_input in (
        snapshot,
        model,
        concept_type,
        quality_profile,
        speed_profile,
        resolution,
    ):
        recommendation_input.input(
            clear_recommendation_state,
            inputs=[recommendation_input],
            outputs=[
                recommendation_state,
                recommendation_view,
                recommendation_mode,
                apply_recommendation,
            ],
        )
    diagnose_event.then(
        clear_recommendation_state,
        outputs=[
            recommendation_state,
            recommendation_view,
            recommendation_mode,
            apply_recommendation,
        ],
    )
    for recommendation_parameter in (
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
    ):
        recommendation_parameter.input(
            mark_recommendation_edited,
            inputs=[recommendation_mode],
            outputs=[recommendation_mode],
        )
    save_inputs = [
        selected_project,
        recommendation_state,
        recommendation_mode,
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
    recollect_performance.click(
        performance_action, inputs=[job_id], outputs=[performance_view]
    )
    refresh_history.click(
        history_action, inputs=[selected_project], outputs=[history_table]
    )
    rebuild_calibration.click(
        rebuild_calibration_action,
        inputs=[selected_project, job_id],
        outputs=[performance_view],
    )

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
                            f"state epoch/step: `{preview.state_epoch}` / "
                            f"`{preview.state_step}`"
                        ),
                        (
                            f"state source: `{preview.state_epoch_source}` / "
                            f"`{preview.state_step_source}`"
                        ),
                        (
                            f"offset epoch/step: `{preview.progress_epoch_offset}` / "
                            f"`{preview.progress_step_offset}`"
                        ),
                        (
                            f"epoch/step: `{preview.current_epoch}` / "
                            f"`{preview.current_step}`"
                        ),
                        f"compatibility: `{preview.compatibility.status.value}`",
                        f"不一致: {issues}",
                        f"position warning: {preview.position_warning or 'なし'}",
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
