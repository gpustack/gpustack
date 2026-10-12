from types import SimpleNamespace
from unittest.mock import patch

from gpustack.schemas.models import BackendEnum, ModelInstanceStateEnum
from gpustack.worker.serve_manager import ServeManager
from tests.utils.model import new_model, new_model_instance
from tests.worker.test_serve_manager import _build_serve_manager


def _manager():
    # Bypass __init__: _resolve_voice_cloning_support only touches the cache.
    manager = ServeManager.__new__(ServeManager)
    manager._voice_cloning_support = {}
    return manager


def _tts_model():
    model = new_model(
        1,
        "voxtral-tts",
        huggingface_repo_id="mistralai/Voxtral-4B-TTS-2603",
        categories=["text_to_speech"],
        backend="vLLM",
    )
    model.backend_version = "0.8.0"
    return model


def test_unsupported_tts_is_flagged_and_detected_once():
    manager = _manager()
    instance = new_model_instance(10, "i", model_id=1)

    def resolve():
        return manager._resolve_voice_cloning_support(
            instance, _tts_model(), BackendEnum.VLLM
        )

    with patch(
        "gpustack.worker.serve_manager.detect_voice_cloning_support",
        return_value=False,
    ) as detect:
        assert resolve() is False
        # Cached: a second resolve must not re-inspect the checkpoint.
        assert resolve() is False

    assert detect.call_count == 1
    assert manager._voice_cloning_support[10] is False


def test_unknown_support_is_not_cached_and_retries():
    manager = _manager()
    instance = new_model_instance(11, "i", model_id=1)

    def resolve():
        return manager._resolve_voice_cloning_support(
            instance, _tts_model(), BackendEnum.VLLM
        )

    with patch(
        "gpustack.worker.serve_manager.detect_voice_cloning_support",
        return_value=None,
    ) as detect:
        assert resolve() is None
        # A None verdict (transient failure / unknown) must not be cached, so a
        # later RUNNING transition retries instead of never reporting the flag.
        assert 11 not in manager._voice_cloning_support
        assert resolve() is None
        assert detect.call_count == 2


def test_non_tts_model_skips_detection():
    manager = _manager()
    instance = new_model_instance(12, "i", model_id=1)
    llm = new_model(1, "llm", huggingface_repo_id="org/llm", backend="vLLM")

    with patch(
        "gpustack.worker.serve_manager.detect_voice_cloning_support",
        return_value=False,
    ) as detect:
        assert (
            manager._resolve_voice_cloning_support(instance, llm, BackendEnum.VLLM)
            is None
        )

    detect.assert_not_called()


def test_voice_cloning_flag_is_folded_into_the_meta_write_back():
    manager, clientset = _build_serve_manager()
    instance = new_model_instance(
        1, "voxtral", 1, worker_id=1, state=ModelInstanceStateEnum.STARTING
    )
    instance.worker_ip = "127.0.0.1"
    instance.port = 8000
    clientset.model_instances.list.return_value = SimpleNamespace(items=[instance])

    model = _tts_model()
    model.meta = {"max_model_len": 32768}

    with (
        patch(
            "gpustack.worker.serve_manager.get_workload",
            return_value=SimpleNamespace(state="running"),
        ),
        patch("gpustack.worker.serve_manager.is_ready", return_value=True),
        patch(
            "gpustack.worker.serve_manager.get_meta_from_running_instance",
            return_value=None,
        ),
        patch(
            "gpustack.worker.serve_manager.detect_voice_cloning_support",
            return_value=False,
        ),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_get_model", return_value=model),
        patch.object(manager, "_update_model_instance"),
        patch.object(manager, "_update_model") as update_model,
    ):
        manager.sync_model_instances_state()

    # The backend meta can be None; folding the flag in must survive that, and
    # the meta already on the model is preserved rather than overwritten.
    update_model.assert_called_once_with(
        model.id, meta={"max_model_len": 32768, "voice_cloning": False}
    )
