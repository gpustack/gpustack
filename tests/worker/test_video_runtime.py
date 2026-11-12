from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from gpustack.schemas.models import (
    is_image_or_video_model,
    is_omni_model,
    is_video_model,
)
from gpustack.utils.model_instance_workers import sglang_smg_mode
from gpustack.utils.sglang_port_layout import sglang_port_layout
from gpustack.worker.backends.sglang import SGLangServer
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker.model_meta import get_meta_from_running_instance
from gpustack.worker.runtime_metrics_aggregator import RuntimeMetricsAggregator
from gpustack.worker.serve_manager import _get_inference_endpoint_and_payload
from tests.utils.model import new_model


@pytest.mark.parametrize("category", ["image", "video"])
def test_media_generation_keeps_diffusion_startup_ports_and_probes(
    monkeypatch, category
):
    model = new_model(
        1,
        "custom",
        huggingface_repo_id="org/custom",
        categories=[category],
        backend="SGLang",
        backend_parameters=["--dp=2", "--enable-dp-attention"],
    )
    assert is_image_or_video_model(model)
    assert is_omni_model(model)
    assert is_video_model(model) == (category == "video")
    assert not sglang_smg_mode(model)
    layout = sglang_port_layout(model, None, [2])
    assert not layout.engine
    assert not layout.components
    assert layout.ranks == 1
    assert layout.internal_count == 0
    assert _get_inference_endpoint_and_payload(model) is None

    backend = SGLangServer.__new__(SGLangServer)
    backend._model = model
    launch = Mock()
    monkeypatch.setattr(backend, "_start", launch)
    backend.start()
    launch.assert_called_once_with()
    assert backend.is_diffusion

    monkeypatch.setattr(
        "gpustack.worker.model_meta.requests.get",
        Mock(side_effect=AssertionError("No LLM metadata probe")),
    )
    assert get_meta_from_running_instance(None, "SGLang", model) == {}
    assert RuntimeMetricsAggregator.__new__(
        RuntimeMetricsAggregator
    )._should_skip_endpoint(model, None, {})


def test_video_understanding_retains_chat_probes_and_engine_layout():
    model = new_model(
        1,
        "vision",
        huggingface_repo_id="org/vision",
        categories=["llm"],
        backend="SGLang",
        backend_parameters=["--dp=2"],
    )
    assert not is_image_or_video_model(model)
    assert not is_video_model(model)
    assert sglang_smg_mode(model)
    assert sglang_port_layout(model, None, [2]).engine
    assert _get_inference_endpoint_and_payload(model)[0] == "/v1/chat/completions"


@pytest.mark.parametrize("category", ["image", "video"])
def test_vllm_adds_omni_once_for_media_generation(category):
    model = new_model(
        1,
        "media",
        huggingface_repo_id="org/media",
        backend="vLLM",
        categories=[category],
    )
    backend = VLLMServer.__new__(VLLMServer)
    backend._model = model
    ctx = SimpleNamespace(is_omni=is_omni_model(model))
    assert backend._build_omni_arguments(ctx) == ["--omni"]
    model.backend_parameters = ["--omni"]
    assert backend._build_omni_arguments(ctx) == []
