"""The draft model a member downloads, and the role that asked for it.

`speculative_config` is a per-role override, and deliberately so: prefill and
decode need *different* values, because for MTP-style speculation the draft head
is part of the model and a prefill that does not load it fails the connector's
structure check.

Two of the places that act on it read the Model's value and not the role's, and
both failures are silent.

* `_build_instance_create` decided `draft_model_source` — the weights to put on
  the machine — from the Model while deciding `backend` from the role, on the
  adjacent line. A group that configured a draft model on decode alone was
  admitted, started, and ran decode with the engine argument naming the draft
  model and no draft weights anywhere on disk. MTP hid it, its head travelling
  inside the main weights; an eagle3 or an external draft model does not.
* `_pinned_draft_names` decided whether a catalog entry is still in use from the
  Model's value, so a name pinned only by a role read as unpinned and the
  reconcile deleted the entry underneath a running deployment.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.catalog_source import _pinned_draft_names
from gpustack.schemas.models import (
    Model,
    RoleSpec,
    SourceEnum,
    SpeculativeAlgorithmEnum,
    SpeculativeConfig,
)
from gpustack.server.controllers import _build_instance_create


def _eagle3(draft_model):
    return SpeculativeConfig(
        enabled=True,
        algorithm=SpeculativeAlgorithmEnum.EAGLE3,
        draft_model=draft_model,
        num_draft_tokens=3,
    )


def _model(speculative_config=None, roles=None) -> Model:
    return Model(
        id=1,
        name="m",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/base",
        owner_principal_id=1,
        cluster_id=1,
        speculative_config=speculative_config,
        roles=roles,
    )


async def _create_for(model, role_name):
    role = next(role for role in model.roles if role.name == role_name)
    # The catalog is empty, so `get_draft_model_source` falls through to reading
    # the name against the main model's source — which is what makes the
    # returned repository id a direct readout of the config it was given.
    with patch(
        "gpustack.utils.model_source.get_catalog_draft_models",
        AsyncMock(return_value=[]),
    ):
        return await _build_instance_create(
            MagicMock(), model, role, group_id="g1", digest="d1"
        )


# --- the weights that reach the machine ------------------------------------ #


@pytest.mark.asyncio
async def test_a_draft_model_configured_on_decode_alone_reaches_decode():
    model = _model(
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(
                name="decode", replicas=1, speculative_config=_eagle3("org/draft")
            ),
        ]
    )

    decode = await _create_for(model, "decode")

    assert decode.draft_model_source is not None
    assert decode.draft_model_source.huggingface_repo_id == "org/draft"


@pytest.mark.asyncio
async def test_and_does_not_reach_prefill():
    """The other half of the same claim: a per-role override that every member
    received would not be an override."""
    model = _model(
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(
                name="decode", replicas=1, speculative_config=_eagle3("org/draft")
            ),
        ]
    )

    prefill = await _create_for(model, "prefill")

    assert prefill.draft_model_source is None


@pytest.mark.asyncio
async def test_a_role_overriding_the_models_draft_model_wins():
    model = _model(
        speculative_config=_eagle3("org/model-level"),
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(
                name="decode", replicas=1, speculative_config=_eagle3("org/role-level")
            ),
        ],
    )

    prefill = await _create_for(model, "prefill")
    decode = await _create_for(model, "decode")

    assert prefill.draft_model_source.huggingface_repo_id == "org/model-level"
    assert decode.draft_model_source.huggingface_repo_id == "org/role-level"


@pytest.mark.asyncio
async def test_a_model_level_draft_model_still_reaches_every_role():
    """Absent on the role means inherit, so the shape that worked before the
    override existed has to keep working."""
    model = _model(
        speculative_config=_eagle3("org/draft"),
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="decode", replicas=1),
        ],
    )

    for role_name in ("prefill", "decode"):
        created = await _create_for(model, role_name)
        assert created.draft_model_source.huggingface_repo_id == "org/draft", role_name


# --- and the catalog entry those weights are resolved through -------------- #


@pytest.mark.asyncio
async def test_a_draft_model_pinned_only_by_a_role_counts_as_pinned():
    model = _model(
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="decode", replicas=1, speculative_config=_eagle3("drafty")),
        ]
    )
    with patch.object(Model, "all", AsyncMock(return_value=[model])):
        pinned = await _pinned_draft_names(SimpleNamespace())

    assert pinned == {"drafty": ["m"]}


@pytest.mark.asyncio
async def test_a_deployment_is_named_once_however_many_roles_pin_the_name():
    """The list is read out into a sentence telling an operator which
    deployments to migrate, so a name repeated per role would be noise."""
    model = _model(
        speculative_config=_eagle3("drafty"),
        roles=[
            RoleSpec(name="prefill", replicas=1, speculative_config=_eagle3("drafty")),
            RoleSpec(name="decode", replicas=1, speculative_config=_eagle3("drafty")),
        ],
    )
    with patch.object(Model, "all", AsyncMock(return_value=[model])):
        pinned = await _pinned_draft_names(SimpleNamespace())

    assert pinned == {"drafty": ["m"]}


@pytest.mark.asyncio
async def test_a_stopped_deployment_pins_nothing_from_its_roles_either():
    """`replicas` on a role-bearing model is the deployment's on/off switch, so
    a group switched off releases its roles' draft models along with its own."""
    model = _model(
        roles=[
            RoleSpec(name="decode", replicas=1, speculative_config=_eagle3("drafty"))
        ]
    )
    model.replicas = 0
    with patch.object(Model, "all", AsyncMock(return_value=[model])):
        pinned = await _pinned_draft_names(SimpleNamespace())

    assert pinned == {}
