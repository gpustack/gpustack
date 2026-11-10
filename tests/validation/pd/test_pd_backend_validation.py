"""Which engines may be disaggregated at all.

Distinct from the per-mode engine check, which asks whether a recipe can be
injected into a role. This one is about the feature: prefill/decode splits a
prompt KV cache across two engines, and an engine that has no such cache to
split cannot be disaggregated no matter who writes the connection parameters.

`custom` mode is the case worth pinning. It injects nothing, so
the `custom` recipe declares no `backends` and the per-mode loop skips it
entirely — which made it a way past every engine check there was.
"""

from contextlib import contextmanager

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.validation.pd.roles import validate_roles
from gpustack.schemas.models import (
    BackendEnum,
    DisaggregationSpec,
    ModelCreate,
    PDModeEnum,
    RoleSpec,
    SourceEnum,
)


@contextmanager
def rejects(fragment):
    """The API's HTTPException carries its text on ``.message``, not on
    ``str()``, so ``pytest.raises(match=...)`` would match the empty string."""
    with pytest.raises(BadRequestException) as excinfo:
        yield
    assert fragment in excinfo.value.message, excinfo.value.message


def _model_in(backend, mode=PDModeEnum.VLLM_NIXL, role_backends=None, roles=True):
    role_backends = role_backends or {}
    return ModelCreate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        backend=backend,
        roles=(
            [
                RoleSpec(name=name, replicas=1, backend=role_backends.get(name))
                for name in ("prefill", "decode", "router")
            ]
            if roles
            else None
        ),
        disaggregation=DisaggregationSpec(mode=mode) if roles else None,
    )


@pytest.mark.parametrize(
    "backend, mode",
    [
        (BackendEnum.VLLM.value, PDModeEnum.VLLM_NIXL),
        (BackendEnum.SGLANG.value, PDModeEnum.SGLANG_MOONCAKE),
        # A user-supplied engine image, which is how a BYO engine runs PD here
        # -- and it can only run it through `custom`, since no recipe targets
        # an engine the platform has never seen.
        (BackendEnum.CUSTOM.value, PDModeEnum.CUSTOM),
    ],
)
def test_an_engine_with_a_kv_cache_to_split_is_admitted(backend, mode):
    validate_roles(_model_in(backend, mode=mode))


@pytest.mark.parametrize(
    "backend",
    [
        # Speech models: there is no prompt KV to hand across, so
        # prefill/decode does not name anything this engine does.
        BackendEnum.VOX_BOX.value,
        # Has disaggregation of its own on Ascend, and is still refused: no
        # recipe ships for it, so the only way in would be `custom` with every
        # connection parameter hand-written.
        BackendEnum.ASCEND_MINDIE.value,
    ],
)
def test_an_engine_pd_does_not_apply_to_is_refused(backend):
    with rejects("cannot be disaggregated"):
        validate_roles(_model_in(backend))


@pytest.mark.parametrize(
    "backend",
    [BackendEnum.VOX_BOX.value, BackendEnum.ASCEND_MINDIE.value],
)
def test_custom_mode_is_not_a_way_around_the_engine_gate(backend):
    """`custom` permits any engine *mix*, which is a statement about recipes,
    not a licence for an engine that cannot be disaggregated. The per-mode
    check reads an empty permit list here and skips, so this gate is the only
    thing standing in the way."""
    with rejects("cannot be disaggregated"):
        validate_roles(_model_in(backend, mode=PDModeEnum.CUSTOM))


def test_a_role_overriding_the_engine_is_judged_on_its_own_backend():
    """A group is only as disaggregable as the engine each member runs, so the
    model-level engine passing says nothing about a role that overrode it."""
    with rejects("MindIE"):
        validate_roles(
            _model_in(
                BackendEnum.VLLM.value,
                mode=PDModeEnum.CUSTOM,
                role_backends={"decode": BackendEnum.ASCEND_MINDIE.value},
            )
        )


@pytest.mark.parametrize("backend", ["SGLang", "vox-box", "my-router"])
def test_router_backend_selection_is_independent_of_engine_compatibility(backend):
    validate_roles(_model_in("vLLM", role_backends={"router": backend}))


@pytest.mark.parametrize("name", ["prefill", "decode", "router"])
def test_a_custom_role_can_supply_its_launch_in_a_builtin_pd_mode(name):
    model = _model_in("vLLM")
    role = next(role for role in model.roles if role.name == name)
    role.backend = BackendEnum.CUSTOM.value
    role.image_name = "my/image:1"
    role.run_command = "my-server --port {{port}}"
    validate_roles(model)


@pytest.mark.parametrize("missing", ["image_name", "run_command"])
def test_an_independent_custom_role_requires_image_and_command(missing):
    model = _model_in("vLLM", mode=PDModeEnum.CUSTOM)
    role = model.roles[-1]
    role.backend = BackendEnum.CUSTOM.value
    role.image_name = "my/router:1"
    role.run_command = "router"
    setattr(role, missing, None)
    with rejects("image and command"):
        validate_roles(model)


@pytest.mark.parametrize(
    "mode",
    [
        PDModeEnum.VLLM_NIXL,
        PDModeEnum.VLLM_MOONCAKE,
        PDModeEnum.VLLM_ASCEND_MOONCAKE,
        PDModeEnum.SGLANG_NIXL,
        PDModeEnum.SGLANG_MOONCAKE,
    ],
)
def test_builtin_pd_accepts_a_custom_router_image_without_a_command(mode):
    model = _model_in(
        "SGLang" if mode.value.startswith("sglang-") else "vLLM", mode=mode
    )
    router = model.roles[-1]
    router.backend = BackendEnum.CUSTOM
    router.image_name = "upstream/router:compatible"
    assert router.run_command is None
    validate_roles(model)


@pytest.mark.parametrize("name", ["prefill", "decode"])
def test_custom_engine_roles_still_require_a_command(name):
    model = _model_in("vLLM")
    role = next(role for role in model.roles if role.name == name)
    role.backend = BackendEnum.CUSTOM
    role.image_name = "my/engine:1"
    with rejects("image and command"):
        validate_roles(model)


def test_an_automatic_custom_router_still_requires_an_image():
    model = _model_in("vLLM", role_backends={"router": BackendEnum.CUSTOM})
    with rejects("requires image"):
        validate_roles(model)


@pytest.mark.parametrize("name", ["prefill", "decode", "router"])
def test_roles_cannot_pin_an_engine_version_beside_a_custom_image(name):
    model = _model_in("vLLM")
    role = next(role for role in model.roles if role.name == name)
    role.backend = "vLLM"
    role.backend_version = "0.30.0"
    role.image_name = "custom:patched"
    with rejects("custom image and a backend version"):
        validate_roles(model)


@pytest.mark.parametrize(
    "backend",
    [BackendEnum.VOX_BOX.value, BackendEnum.ASCEND_MINDIE.value],
)
def test_an_aggregated_deployment_on_the_same_engine_is_untouched(backend):
    """The gate is on disaggregation, not on the engine: refusing these
    deployments outright would take away the way they are meant to run."""
    validate_roles(_model_in(backend, roles=False))
