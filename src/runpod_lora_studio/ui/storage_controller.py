from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from runpod_lora_studio.domain.storage_models import (
    ManagedModel,
    StorageTransferJob,
    TransferPlan,
)
from runpod_lora_studio.services.storage_service import StorageService


@dataclass(frozen=True, slots=True)
class StorageController:
    service: StorageService

    def environment_rows(self) -> list[list[str]]:
        result = self.service.validate_environment()
        return [
            [check.name, "OK" if check.ok else "ERROR", check.message]
            for check in result.checks
        ]

    def models(
        self, query: str = "", extension: str | None = None, page: int = 1
    ) -> list[ManagedModel]:
        return self.service.list_models(
            query=query, extension=extension or None, page=page
        )

    def model_rows(self, models: list[ManagedModel]) -> list[list[str]]:
        return [
            [
                str(model.id),
                model.display_name,
                model.model_type.value,
                model.remote_relative_path,
                str(model.remote_size_bytes),
                model.status.value,
                str(model.local_path or ""),
                model.local_sha256 or "",
                model.remote_hash_type or "",
                model.remote_hash_value or "",
                model.verified_at.isoformat() if model.verified_at else "",
            ]
            for model in models
        ]

    def dry_run_model(self, model_id: UUID) -> TransferPlan:
        return self.service.dry_run_model_download(model_id)

    def download_model(self, model_id: UUID, token: str | None = None) -> str:
        return str(self.service.start_model_download(model_id, plan_token=token))

    def verify_model(self, model_id: UUID) -> str:
        return "検証成功" if self.service.verify_model(model_id) else "検証失敗"

    def dry_run_snapshot(self, snapshot_id: UUID) -> TransferPlan:
        return self.service.dry_run_snapshot_upload(snapshot_id)

    def upload_snapshot(self, snapshot_id: UUID, token: str | None = None) -> str:
        return str(self.service.start_snapshot_upload(snapshot_id, plan_token=token))

    def jobs(self, project_id: UUID | None = None) -> list[StorageTransferJob]:
        return self.service.list_jobs(project_id)

    @staticmethod
    def plan_message(plan: TransferPlan) -> str:
        return "\n".join(
            [
                f"- 対象: {len(plan.items)} files",
                f"- コピー: {plan.copy_count}",
                f"- スキップ: {plan.skip_count}",
                f"- 衝突: {plan.conflict_count}",
                f"- bytes: {plan.total_bytes}",
                f"- token: `{plan.token}`",
                *plan.errors,
            ]
        )
