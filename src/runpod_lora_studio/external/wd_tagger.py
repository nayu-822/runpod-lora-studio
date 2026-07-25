from __future__ import annotations

import csv
import importlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from runpod_lora_studio.config.settings import AppSettings
from runpod_lora_studio.domain.models import (
    TagCategory,
    TaggerInferenceSettings,
    TaggerModelIdentity,
    TaggingResult,
    TagPrediction,
)
from runpod_lora_studio.external.tagger import (
    TaggerBackend,
    TaggerEnvironmentError,
    TaggerInferenceError,
    ValidationResult,
    preprocess_image,
)


class OnnxRuntimeWDBackend:
    """Small optional WD-compatible ONNX backend.

    The ONNX Runtime and NumPy packages remain optional so normal development
    and unit tests do not pull a GPU stack or a model from the network.
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._numpy: Any = None
        self._input_name = ""
        self._labels: tuple[tuple[str, TagCategory], ...] = ()

    def validate_environment(self, model_path: Path) -> ValidationResult:
        missing = [
            str(path)
            for path in (model_path / "model.onnx", model_path / "selected_tags.csv")
            if not path.is_file()
        ]
        if missing:
            return ValidationResult(
                False,
                "WD Taggerの必要ファイルがありません: " + ", ".join(missing),
                "",
            )
        if importlib.util.find_spec("onnxruntime") is None:
            return ValidationResult(
                False,
                "onnxruntimeがインストールされていません。",
                "",
            )
        if importlib.util.find_spec("numpy") is None:
            return ValidationResult(False, "numpyがインストールされていません。", "")
        return ValidationResult(True, "WD ONNXバックエンドを利用できます。", "")

    def load(self, model_path: Path, settings: TaggerInferenceSettings) -> None:
        validation = self.validate_environment(model_path)
        if not validation.ok:
            raise TaggerEnvironmentError(validation.message)
        runtime = importlib.import_module("onnxruntime")
        self._numpy = importlib.import_module("numpy")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if settings.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = runtime.InferenceSession(
            str(model_path / "model.onnx"), providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._labels = self._read_labels(model_path / "selected_tags.csv")

    def predict(
        self, image: Any, settings: TaggerInferenceSettings
    ) -> tuple[TagPrediction, ...]:
        if self._session is None or self._numpy is None:
            raise TaggerInferenceError("WD ONNXモデルがロードされていません。")
        array = self._numpy.asarray(image.image, dtype=self._numpy.float32) / 255.0
        # WD 1.4 ONNX exports commonly expect NHWC BGR tensors. The shared
        # preprocessor keeps RGB explicit; the model-specific conversion is
        # kept inside this backend.
        tensor = array[:, :, ::-1][self._numpy.newaxis, ...]
        outputs = self._session.run(None, {self._input_name: tensor})
        scores = self._numpy.asarray(outputs[0]).reshape(-1)
        predictions: list[TagPrediction] = []
        for order, ((name, category), score) in enumerate(
            zip(self._labels, scores, strict=False)
        ):
            predictions.append(
                TagPrediction(
                    tag_name_raw=name,
                    tag_name_normalized=name,
                    category=category,
                    confidence=float(score),
                    original_order=order,
                )
            )
        return tuple(predictions)

    def unload(self) -> None:
        self._session = None
        self._numpy = None
        self._labels = ()

    @staticmethod
    def _read_labels(path: Path) -> tuple[tuple[str, TagCategory], ...]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            labels: list[tuple[str, TagCategory]] = []
            for row in rows:
                name = (row.get("name") or row.get("tag") or "").strip()
                if not name:
                    continue
                try:
                    category_id = int(row.get("category", "-1"))
                except ValueError:
                    category_id = -1
                category = {
                    0: TagCategory.RATING,
                    1: TagCategory.GENERAL,
                    4: TagCategory.CHARACTER,
                }.get(category_id, TagCategory.UNKNOWN)
                labels.append((name, category))
            return tuple(labels)


class WDTaggerAdapter:
    """WD 1.4-compatible adapter boundary.

    The inference backend is intentionally injected. A RunPod deployment can
    provide an ONNX backend without coupling model-specific code to services or
    Gradio. Unit tests use FakeTaggerAdapter and never download a model.
    """

    adapter_name = "wd14"
    implementation_version = "wd14-adapter-v1"

    def __init__(
        self,
        settings: AppSettings,
        backend: TaggerBackend | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend or OnnxRuntimeWDBackend()
        self._resolved_device = "cpu"
        self._loaded = False

    def model_identity(self) -> TaggerModelIdentity:
        return TaggerModelIdentity(
            adapter_name=self.settings.tagger_adapter_name,
            model_identifier=self.settings.tagger_model_identifier,
            model_revision=self.settings.tagger_model_revision,
            model_path=str(self.settings.tagger_model_dir),
            implementation_version=self.implementation_version,
        )

    def validate_environment(self) -> ValidationResult:
        device = self._resolve_device()
        if device == "unavailable":
            return ValidationResult(False, "指定されたデバイスを利用できません。", "")
        model_dir = self._safe_model_dir()
        if not model_dir.is_dir():
            if self.settings.tagger_allow_model_download and self._hub_available():
                return ValidationResult(
                    True,
                    "WD Taggerモデルは未配置です。開始時に一時領域へ取得します。",
                    device,
                )
            return ValidationResult(
                False,
                f"WD Taggerモデルがありません。配置先: {model_dir}",
                device,
            )
        if isinstance(self.backend, OnnxRuntimeWDBackend):
            backend_validation = self.backend.validate_environment(model_dir)
            if not backend_validation.ok:
                if self.settings.tagger_allow_model_download and self._hub_available():
                    return ValidationResult(
                        True,
                        "WD Taggerモデルを開始時に一時領域へ取得します。",
                        device,
                    )
                return ValidationResult(False, backend_validation.message, device)
        return ValidationResult(True, "WD Tagger環境を利用できます。", device)

    def load(self) -> None:
        validation = self.validate_environment()
        if not validation.ok:
            raise TaggerEnvironmentError(validation.message)
        self._resolved_device = validation.resolved_device
        self._ensure_model_available()
        self.backend.load(
            self._safe_model_dir(),
            self._inference_settings(self._resolved_device),
        )
        self._loaded = True

    def tag_image(
        self, image_path: Path, settings: TaggerInferenceSettings
    ) -> TaggingResult:
        if not self._loaded:
            raise TaggerEnvironmentError("WD Taggerモデルがロードされていません。")
        prepared = preprocess_image(image_path)
        try:
            predictions = tuple(self.backend.predict(prepared, settings))
        except Exception as exc:
            raise TaggerInferenceError("WD Taggerの推論に失敗しました。") from exc
        return TaggingResult(tags=predictions, raw_output=None)

    def unload(self) -> None:
        if self._loaded:
            self.backend.unload()
        self._loaded = False

    def _inference_settings(self, device: str) -> TaggerInferenceSettings:
        return TaggerInferenceSettings(
            device=device,
            batch_size=self.settings.tagger_batch_size,
            general_threshold=self.settings.tagger_general_threshold,
            character_threshold=self.settings.tagger_character_threshold,
            save_rating=self.settings.tagger_save_rating,
            save_character=self.settings.tagger_save_character,
            save_general=self.settings.tagger_save_general,
            underscore_to_space=self.settings.tagger_underscore_to_space,
            escape_mode=self.settings.tagger_escape_mode,
            max_workers=self.settings.tagger_max_workers,
            allow_model_download=self.settings.tagger_allow_model_download,
        )

    def _safe_model_dir(self) -> Path:
        model_dir = self.settings.tagger_model_dir.expanduser().resolve()
        models_root = self.settings.models_dir.expanduser().resolve()
        try:
            model_dir.relative_to(models_root)
        except ValueError as exc:
            raise TaggerEnvironmentError(
                "モデル保存先が許可された範囲外です。"
            ) from exc
        return model_dir

    def _ensure_model_available(self) -> None:
        model_dir = self._safe_model_dir()
        if model_dir.is_dir() and (model_dir / "model.onnx").is_file():
            return
        if not self.settings.tagger_allow_model_download:
            raise TaggerEnvironmentError(
                f"WD Taggerモデルがありません。想定保存先: {model_dir}"
            )
        if not self._hub_available():
            raise TaggerEnvironmentError(
                "モデル自動取得にはhuggingface_hubが必要です。"
            )
        staging = self.settings.temp_dir / f"tagger-model-{uuid4()}"
        try:
            hub = importlib.import_module("huggingface_hub")
            hub.snapshot_download(
                repo_id=self.settings.tagger_model_identifier,
                revision=self.settings.tagger_model_revision,
                local_dir=str(staging),
                allow_patterns=["*.onnx", "selected_tags.csv", "*.json"],
            )
            if (
                not (staging / "model.onnx").is_file()
                or not (staging / "selected_tags.csv").is_file()
            ):
                raise TaggerEnvironmentError(
                    "取得したWD Taggerモデルに必要ファイルがありません。"
                )
            if model_dir.exists():
                raise TaggerEnvironmentError(
                    "モデル保存先に不完全なディレクトリが残っています。"
                )
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(model_dir)
        except TaggerEnvironmentError:
            raise
        except Exception as exc:
            raise TaggerEnvironmentError(
                "WD Taggerモデルの取得に失敗しました。"
            ) from exc
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _hub_available() -> bool:
        return importlib.util.find_spec("huggingface_hub") is not None

    def _resolve_device(self) -> str:
        requested = self.settings.tagger_device
        if requested == "cpu":
            return "cpu"
        has_cuda = False
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                has_cuda = bool(torch.cuda.is_available())
            except Exception:
                has_cuda = False
        if requested == "cuda":
            return "cuda" if has_cuda else "unavailable"
        return "cuda" if has_cuda else "cpu"
