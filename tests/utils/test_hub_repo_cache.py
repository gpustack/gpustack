from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.scheduler.evaluator import evaluate_model_metadata
from gpustack.schemas.model_sets import ModelSpec
from gpustack.schemas.models import SourceEnum
from gpustack.utils.hub import is_repo_cached, list_repo


def _list_repo(repo_id, root_dir_only):
    fs = MagicMock()
    fs.ls.return_value = [{"name": f"{repo_id}/config.json", "size": 1}]
    with patch("gpustack.utils.hub.HfFileSystem", return_value=fs):
        list_repo(repo_id, SourceEnum.HUGGING_FACE, root_dir_only=root_dir_only)


@pytest.mark.parametrize("root_dir_only", [False, True])
def test_a_listed_repo_counts_as_cached(config, root_dir_only):
    repo_id = f"org/listed-{root_dir_only}"
    _list_repo(repo_id, root_dir_only)

    assert is_repo_cached(repo_id, SourceEnum.HUGGING_FACE)


def test_an_unlisted_repo_does_not_count_as_cached(config):
    _list_repo("org/listed", False)

    assert not is_repo_cached("org/unlisted", SourceEnum.HUGGING_FACE)


@pytest.mark.asyncio
async def test_evaluation_of_a_cached_repo_does_not_need_the_hub(config):
    repo_id = "org/air-gapped"
    _list_repo(repo_id, True)
    spec = ModelSpec(
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id=repo_id,
        backend="vLLM",
        replicas=1,
    )

    with (
        patch(
            "gpustack.scheduler.evaluator.auth_check",
            side_effect=ConnectionError("huggingface.co is unreachable"),
        ) as auth_check,
        patch(
            "gpustack.scheduler.evaluator.scheduler.evaluate_pretrained_config",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await evaluate_model_metadata(config, spec, [])

    assert result == (True, [])
    auth_check.assert_not_called()
