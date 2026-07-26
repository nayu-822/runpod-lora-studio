from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from runpod_lora_studio.domain.recommendation_models import TrainingRecommendation
from runpod_lora_studio.domain.training_models import TrainingConfigInput, TrainingJob
from runpod_lora_studio.domain.training_resume_models import TrainingResumePreview
from runpod_lora_studio.services.training_service import TrainingService


@dataclass(frozen=True, slots=True)
class TrainingController:
    service: TrainingService

    def snapshot_choices(self, project_id: str | None) -> list[str]:
        if not project_id:
            return []
        return [
            f"{item_id} | {name}"
            for item_id, name in self.service.list_completed_snapshots(UUID(project_id))
        ]

    def model_choices(self, project_id: str | None) -> list[str]:
        if not project_id:
            return []
        return [
            f"{item_id} | {name}"
            for item_id, name in self.service.list_available_models(UUID(project_id))
        ]

    def save_config(
        self,
        project_id: str,
        values: tuple[Any, ...],
        recommendation: TrainingRecommendation | None = None,
    ) -> str:
        (
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
        ) = values
        data = TrainingConfigInput(
            project_id=UUID(project_id),
            dataset_snapshot_id=_choice_id(str(snapshot)),
            managed_model_id=_choice_id(str(model)),
            name=str(name),
            output_name=str(output_name),
            output_directory=Path(str(output_directory)),
            sd_scripts_root=Path(str(sd_scripts_root)),
            resolution=int(resolution),
            batch_size=int(batch_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            optimizer=str(optimizer),
            scheduler=str(scheduler),
            network_module=str(network_module),
            network_dim=int(network_dim),
            network_alpha=int(network_alpha),
            mixed_precision=str(mixed_precision),
            seed=int(seed),
            save_every_n_epochs=int(save_every),
            cache_latents=bool(cache_latents),
            gradient_checkpointing=bool(gradient_checkpointing),
            recommendation_id=recommendation.id if recommendation else None,
            recommendation_engine_version=(
                recommendation.engine_version if recommendation else None
            ),
            recommendation_change_diff=(
                {
                    "batch_size": recommendation.batch_size,
                    "epochs": recommendation.epochs,
                    "learning_rate": recommendation.learning_rate,
                    "optimizer": recommendation.optimizer,
                    "scheduler": recommendation.scheduler,
                    "network_dim": recommendation.network_dim,
                    "network_alpha": recommendation.network_alpha,
                    "mixed_precision": recommendation.mixed_precision,
                }
                if recommendation
                else {}
            ),
        )
        return str(self.service.create_config(data).id)

    def create_job(self, config_id: str) -> str:
        return str(self.service.create_job(UUID(config_id)))

    def resumable_job_choices(self, project_id: str | None) -> list[str]:
        if not project_id:
            return []
        return [
            f"{job.id} | {job.status.value}"
            for job in self.service.list_resumable_jobs(UUID(project_id))
        ]

    def config_choices(self, project_id: str | None) -> list[str]:
        if not project_id:
            return []
        return [
            f"{config.id} | {config.name}"
            for config in self.service.list_configs(UUID(project_id))
        ]

    def resume_state_choices(self, job_id: str | None) -> list[str]:
        if not job_id:
            return []
        return [
            f"{item['id']} | {item['filename']}"
            for item in self.service.list_resume_states(UUID(job_id))
            if item["validation_status"] == "valid"
        ]

    def preview_resume(
        self, job_id: str, artifact_choice: str, config_choice: str | None = None
    ) -> TrainingResumePreview:
        return self.service.preview_resume(
            UUID(_choice_id_text(job_id)),
            UUID(_choice_id_text(artifact_choice)),
            _optional_choice_id(config_choice),
        )

    def create_resume_job(
        self,
        job_id: str,
        artifact_choice: str,
        signature: str,
        config_choice: str | None = None,
    ) -> str:
        return str(
            self.service.create_resume_job(
                UUID(_choice_id_text(job_id)),
                UUID(_choice_id_text(artifact_choice)),
                target_config_id=_optional_choice_id(config_choice),
                preview_signature=signature,
            )
        )

    def start_resume_job(self, job_id: str) -> str:
        return self.start_job(job_id)

    def start_job(self, job_id: str) -> str:
        return str(self.service.start_job(UUID(job_id)))

    def cancel_job(self, job_id: str) -> str:
        return self.service.request_cancel(UUID(job_id))

    def job_rows(self, project_id: str | None) -> list[list[str]]:
        jobs = self.service.list_jobs(UUID(project_id) if project_id else None)
        return [self.job_row(job) for job in jobs]

    def progress_row(self, job_id: str | None) -> list[str]:
        if not job_id:
            return ["", "", "", "", "", "", "", "", "", "", "", "", ""]
        progress = self.service.get_progress(UUID(job_id))
        if progress is None:
            return ["unknown"] + [""] * 12
        return [
            progress.parse_status.value,
            _display_pair(progress.current_epoch, progress.total_epochs),
            _display_pair(progress.current_step, progress.total_steps),
            _display_ratio(progress.progress_ratio),
            _display_number(progress.latest_loss),
            _display_number(progress.smoothed_loss),
            _display_number(progress.learning_rate),
            _display_number(progress.steps_per_second),
            _display_seconds(progress.elapsed_seconds),
            _display_seconds(progress.estimated_remaining_seconds),
            progress.latest_log_at.isoformat() if progress.latest_log_at else "unknown",
            progress.parse_warning or "",
            progress.progress_source.value,
        ]

    def metric_rows(self, job_id: str | None, limit: int = 500) -> list[list[object]]:
        if not job_id:
            return []
        return [
            [step, value, epoch if epoch is not None else ""]
            for step, value, epoch in self.service.list_metrics(
                UUID(job_id), limit=limit
            )
        ]

    def artifact_rows(self, job_id: str | None, limit: int = 500) -> list[list[str]]:
        if not job_id:
            return []
        return [
            [
                artifact.filename,
                artifact.artifact_type.value,
                str(artifact.epoch or ""),
                str(artifact.step or ""),
                str(artifact.file_size),
                (artifact.sha256 or "")[:12],
                artifact.validation_status.value,
                artifact.validation_message or "",
                artifact.modified_at.isoformat() if artifact.modified_at else "",
            ]
            for artifact in self.service.list_artifacts(UUID(job_id), limit)
        ]

    @staticmethod
    def job_row(job: TrainingJob) -> list[str]:
        return [
            str(job.id),
            job.status.value,
            str(job.pid or ""),
            job.started_at.isoformat() if job.started_at else "",
            job.finished_at.isoformat() if job.finished_at else "",
            str(job.exit_code if job.exit_code is not None else ""),
            job.worker_heartbeat.isoformat() if job.worker_heartbeat else "",
            job.failure_message or "",
        ]


def _choice_id(value: str) -> UUID:
    return UUID(value.split(" | ", 1)[0].strip())


def _choice_id_text(value: str) -> str:
    return value.split(" | ", 1)[0].strip()


def _optional_choice_id(value: str | None) -> UUID | None:
    return UUID(_choice_id_text(value)) if value else None


def _display_pair(current: int | None, total: int | None) -> str:
    return (
        f"{current if current is not None else '?'} / "
        f"{total if total is not None else '?'}"
    )


def _display_ratio(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "unknown"


def _display_number(value: float | None) -> str:
    return f"{value:.6g}" if value is not None else "unknown"


def _display_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{max(0.0, value):.1f}s"
