from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from runpod_lora_studio.domain.training_models import TrainingConfigInput, TrainingJob
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

    def save_config(self, project_id: str, values: tuple[Any, ...]) -> str:
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
        )
        return str(self.service.create_config(data).id)

    def create_job(self, config_id: str) -> str:
        return str(self.service.create_job(UUID(config_id)))

    def start_job(self, job_id: str) -> str:
        return str(self.service.start_job(UUID(job_id)))

    def cancel_job(self, job_id: str) -> str:
        return self.service.request_cancel(UUID(job_id))

    def job_rows(self, project_id: str | None) -> list[list[str]]:
        jobs = self.service.list_jobs(UUID(project_id) if project_id else None)
        return [self.job_row(job) for job in jobs]

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
