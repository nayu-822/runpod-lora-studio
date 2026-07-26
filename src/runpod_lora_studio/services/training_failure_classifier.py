from __future__ import annotations

import re

from runpod_lora_studio.domain.training_performance_models import (
    TrainingFailureCategory,
    TrainingFailureClassification,
)


class TrainingFailureClassifier:
    """Classify terminal jobs using a small, allowlisted set of log markers.

    Log text is treated as data.  The classifier never evaluates or executes it,
    and callers should persist only the returned evidence codes.
    """

    version = "phase7b-failure-v1"
    _patterns: tuple[tuple[TrainingFailureCategory, str, re.Pattern[str]], ...] = (
        (
            TrainingFailureCategory.CUDA_OUT_OF_MEMORY,
            "cuda_oom",
            re.compile(
                r"cuda out of memory|torch\.cuda\.outofmemoryerror|"
                r"cublas_status_alloc_failed|outofmemoryerror",
                re.IGNORECASE,
            ),
        ),
        (
            TrainingFailureCategory.SYSTEM_OUT_OF_MEMORY,
            "cpu_oom",
            re.compile(
                r"defaultcpuallocator[^\n]*can't allocate memory|"
                r"cannot allocate memory|out of memory|(?:^|\n)killed(?:\r?$)",
                re.IGNORECASE | re.MULTILINE,
            ),
        ),
        (
            TrainingFailureCategory.DISK_FULL,
            "disk_full",
            re.compile(
                r"no space left on device|disk full|not enough disk space",
                re.IGNORECASE,
            ),
        ),
        (
            TrainingFailureCategory.MODEL_LOAD_FAILURE,
            "model_load_failure",
            re.compile(
                r"error loading model|failed to load.*model|checkpoint.*not found",
                re.IGNORECASE,
            ),
        ),
        (
            TrainingFailureCategory.DATASET_FAILURE,
            "dataset_failure",
            re.compile(
                r"dataset.*(not found|failed|empty)|"
                r"failed to read.*image|caption.*not found",
                re.IGNORECASE,
            ),
        ),
        (
            TrainingFailureCategory.INVALID_CONFIGURATION,
            "invalid_configuration",
            re.compile(
                r"invalid (argument|configuration)|unknown argument|"
                r"unrecognized arguments",
                re.IGNORECASE,
            ),
        ),
        (
            TrainingFailureCategory.DEPENDENCY_FAILURE,
            "dependency_failure",
            re.compile(
                r"modulenotfounderror|no module named|importerror|bitsandbytes.*error",
                re.IGNORECASE,
            ),
        ),
    )

    def classify(
        self,
        *,
        status: str,
        exit_code: int | None,
        stdout: str = "",
        stderr: str = "",
        cancel_requested: bool = False,
        stale: bool = False,
    ) -> TrainingFailureClassification:
        normalized_status = status.lower()
        if normalized_status == "canceled" or cancel_requested:
            return self._result(
                TrainingFailureCategory.USER_CANCELED, ("user_canceled",)
            )
        if normalized_status == "stale":
            return self._result(
                TrainingFailureCategory.STALE_PROCESS, ("stale_process",)
            )
        if exit_code == 0 and normalized_status == "succeeded":
            return self._result(TrainingFailureCategory.NONE)

        combined = f"{stderr}\n{stdout}"
        for category, code, pattern in self._patterns:
            if pattern.search(combined):
                return self._result(category, (code,))
        if exit_code is not None and exit_code < 0:
            return self._result(
                TrainingFailureCategory.PROCESS_KILLED, ("signal_exit",)
            )
        if stale:
            return self._result(
                TrainingFailureCategory.STALE_PROCESS, ("stale_process",)
            )
        return self._result(
            TrainingFailureCategory.UNKNOWN_FAILURE, ("unclassified_failure",)
        )

    def _result(
        self,
        category: TrainingFailureCategory,
        evidence_codes: tuple[str, ...] = (),
    ) -> TrainingFailureClassification:
        return TrainingFailureClassification(
            category=category,
            evidence_codes=evidence_codes,
            classifier_version=self.version,
        )
