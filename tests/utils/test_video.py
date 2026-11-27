import pytest

from gpustack.utils.video import get_video_capabilities, update_video_metadata
from tests.utils.model import new_model


@pytest.mark.parametrize(
    "backend,flag", [("SGLang", "model-variant"), ("vLLM", "task-type")]
)
@pytest.mark.parametrize(
    "partition,tasks", [("fl2va", ["t2va", "fl2va"]), ("ref2va", ["ref2va"])]
)
def test_h3_tasks_follow_the_loaded_partition_and_ignore_deployment_alias(
    backend, flag, partition, tasks
):
    model = new_model(
        1,
        "arbitrary-alias",
        huggingface_repo_id="MiniMaxAI/MiniMax-H3",
        backend=backend,
        backend_parameters=[f"--{flag}=fl2va", f"--{flag}={partition}"],
    )
    capabilities = get_video_capabilities(model)
    assert capabilities["tasks"] == tasks
    assert capabilities["playground_tasks"] == tasks
    assert capabilities["default_task"] == tasks[0]
    assert capabilities["media_inputs"][partition]["max_items"] == (
        2 if partition == "fl2va" else 12
    )
    assert capabilities["request_profile"] == (
        "sglang_h3" if backend == "SGLang" else "vllm_omni_h3"
    )


def test_h3_artifact_metadata_is_merged_and_bound_to_its_source():
    model = new_model(
        1,
        "local-alias",
        huggingface_repo_id="org/custom",
        backend="SGLang",
        meta={"voices": ["existing"], "video": "custom-value"},
    )
    assert get_video_capabilities(model) is None
    assert update_video_metadata(model, {"_class_name": "MiniMaxH3ModularPipeline"})
    assert model.meta["voices"] == ["existing"]
    assert get_video_capabilities(model)["family"] == "minimax_h3"
    assert not update_video_metadata(model)
    model.huggingface_repo_id = "org/another-model"
    assert get_video_capabilities(model) is None
    assert update_video_metadata(model)
    assert model.meta == {"voices": ["existing"]}


def test_video_named_language_model_does_not_get_h3_parameters():
    model = new_model(
        1, "MiniMax-H3", huggingface_repo_id="org/language-model", backend="vLLM"
    )
    assert get_video_capabilities(model) is None
