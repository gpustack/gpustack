import os
import time
import pytest
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.schemas.models import SourceEnum
from gpustack.server.catalog import get_model_set_specs, init_model_catalog
from gpustack.utils.hub import match_hugging_face_files, match_model_scope_file_paths
from gpustack.utils.compat_importlib import pkg_resources
from huggingface_hub import HfApi
from modelscope.hub.api import HubApi


@pytest.mark.skipif(
    os.getenv("HF_TOKEN") is None,
    reason="Skipped by default unless HF_TOKEN is set. Unauthed requests are rate limited.",
)
def test_model_catalog():
    init_model_catalog()

    model_set_specs = get_model_set_specs()

    Hfapi = HfApi()

    model_name_filter = os.getenv("TEST_CATALOG_MODEL_NAME_FILTER")
    for model_set_id, model_specs in model_set_specs.items():
        assert model_set_id is not None
        assert len(model_specs) > 0
        for model_spec in model_specs:
            assert (
                model_spec.source == SourceEnum.HUGGING_FACE
            ), f"Expected huggingface source but got: {model_spec.source}"

            if (
                model_name_filter is not None
                and model_name_filter not in model_spec.huggingface_repo_id
            ):
                continue

            time.sleep(0.01)  # mitigate rate limit

            print(model_spec.huggingface_repo_id, model_spec.huggingface_filename)
            if model_spec.huggingface_filename is None:
                model_info = Hfapi.model_info(model_spec.huggingface_repo_id)
                assert model_info is not None
            else:
                match_files = match_hugging_face_files(
                    model_spec.huggingface_repo_id, model_spec.huggingface_filename
                )
                assert (
                    len(match_files) > 0
                ), f"Failed to find model files: {model_spec.huggingface_repo_id}, {model_spec.huggingface_filename}"


@pytest.mark.skipif(
    os.getenv("HF_TOKEN") is None,
    reason="Skipped by default unless HF_TOKEN is set. Unauthed requests are rate limited.",
)
def test_model_catalog_modelscope():
    modelscope_catalog_file = pkg_resources.files("gpustack.assets").joinpath(
        "model-catalog-modelscope.yaml"
    )

    init_model_catalog(str(modelscope_catalog_file))

    model_set_specs = get_model_set_specs()

    Msapi = HubApi()

    model_name_filter = os.getenv("TEST_CATALOG_MODEL_NAME_FILTER")
    for model_set_id, model_specs in model_set_specs.items():
        assert model_set_id is not None
        assert len(model_specs) > 0
        for model_spec in model_specs:
            assert (
                model_spec.source == SourceEnum.MODEL_SCOPE
            ), f"Expected modelscope source but got: {model_spec.source}"

            if (
                model_name_filter is not None
                and model_name_filter not in model_spec.model_scope_model_id
            ):
                continue

            print(model_spec.model_scope_model_id, model_spec.model_scope_file_path)
            if model_spec.model_scope_file_path is None:
                model_info = Msapi.get_model(model_spec.model_scope_model_id)
                assert model_info is not None
            else:
                match_files = match_model_scope_file_paths_with_retry(
                    model_spec.model_scope_model_id,
                    model_spec.model_scope_file_path,
                )
                assert (
                    len(match_files) > 0
                ), f"Failed to find model files: {model_spec.model_scope_model_id}, {model_spec.model_scope_file_path}"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def match_model_scope_file_paths_with_retry(
    model_scope_model_id, model_scope_file_path
):
    return match_model_scope_file_paths(model_scope_model_id, model_scope_file_path)
