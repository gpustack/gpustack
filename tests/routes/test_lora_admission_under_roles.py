"""LoRA adapters against a role-bearing deployment.

Two refusals, one wall behind both.

A group is addressed through its router, and the router's worker registry is
indexed by served-model name. `membership_api.body` in `pd-modes.yaml` registers
each member with `model_id: "{{model_name}}"` — the base name, once — so nothing
in that registry answers to an adapter's name.

So the route for `<base>:<adapter>` is created and reports a ready target, but
a chat against the adapter name answers 503 `No available workers (all circuits
open or unhealthy)` while the same chat against the base name answers 200. The
engines are innocent — prefill and decode each list the adapter in their own
`GET /v1/models` and each answer a direct chat on it, and the same adapter on
the same worker deployed as a plain single instance serves fine.

So `lora_list` beside `disaggregation` is refused at admission, and
`roles[].lora_list` is refused wherever it appears.
"""

from contextlib import contextmanager

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.models import validate_roles
from gpustack.schemas.models import (
    DisaggregationSpec,
    LoraListEntry,
    Model,
    ModelCreate,
    ModelUpdate,
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


def _adapters(*names):
    return [LoraListEntry(lora_name=name) for name in names]


def _pd_roles(**overrides):
    return [
        RoleSpec(name="prefill", replicas=1, **overrides.get("prefill", {})),
        RoleSpec(name="decode", replicas=1, **overrides.get("decode", {})),
        RoleSpec(name="router", replicas=1, **overrides.get("router", {})),
    ]


def _model_in(cls=ModelCreate, roles=None, disaggregation=True, **kwargs):
    return cls(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        backend="vLLM",
        roles=_pd_roles() if roles is None else roles,
        disaggregation=(
            DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL) if disaggregation else None
        ),
        **kwargs,
    )


def _stored(**kwargs) -> Model:
    return Model(
        id=1,
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        backend="vLLM",
        owner_principal_id=1,
        cluster_id=1,
        **kwargs,
    )


# --- creation -------------------------------------------------------------- #


def test_a_group_created_with_adapters_is_refused():
    with rejects("cannot serve LoRA adapters"):
        validate_roles(_model_in(lora_list=_adapters("m:tldr")))


def test_the_refusal_says_where_the_request_stops_and_what_to_do_instead():
    """The text is shown to the user and is the only account they get of a
    failure whose symptom — a 503 from a group that reports itself healthy —
    points at nothing."""
    with pytest.raises(BadRequestException) as excinfo:
        validate_roles(_model_in(lora_list=_adapters("m:tldr")))

    message = excinfo.value.message
    assert "router" in message
    assert "served-model name" in message
    assert "non-disaggregated" in message


def test_a_group_without_adapters_is_untouched():
    validate_roles(_model_in())


def test_adapters_on_a_model_with_no_disaggregation_are_untouched():
    """Plain multi-role orchestration is not a group behind a router in the
    sense this rule is about, and a role-less model is the ordinary LoRA case
    that measurably works."""
    validate_roles(_model_in(disaggregation=False, lora_list=_adapters("m:tldr")))
    validate_roles(
        ModelCreate(
            name="m",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            lora_list=_adapters("m:tldr"),
        )
    )


def test_an_empty_adapter_list_is_not_a_declaration():
    validate_roles(_model_in(lora_list=[]))


# --- both directions of an update ------------------------------------------ #


def test_adding_adapters_to_a_group_that_already_disaggregates_is_refused():
    """The update carries no `disaggregation` — it is not what is changing — so
    the rule has to read the stored row to see the combination at all."""
    stored = _stored(
        roles=_pd_roles(), disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL)
    )
    model_in = ModelUpdate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        lora_list=_adapters("m:tldr"),
    )
    with rejects("cannot serve LoRA adapters"):
        validate_roles(model_in, stored=stored)


def test_adding_disaggregation_to_a_model_that_already_carries_adapters_is_refused():
    """The mirror image, and it has to be caught by the same rule: an update
    that sends only `roles` and `disaggregation` never mentions the adapters it
    is about to break."""
    stored = _stored(lora_list=_adapters("m:tldr"))
    with rejects("cannot serve LoRA adapters"):
        validate_roles(_model_in(cls=ModelUpdate), stored=stored)


def test_clearing_the_adapters_in_the_same_request_is_the_way_out():
    """The refusal must leave an exit: a deployment that already holds both is
    fixed by sending the change that removes one of them, and an explicit empty
    list is that change."""
    stored = _stored(lora_list=_adapters("m:tldr"))
    validate_roles(_model_in(cls=ModelUpdate, lora_list=[]), stored=stored)


def test_dropping_the_disaggregation_is_the_other_way_out():
    stored = _stored(
        lora_list=_adapters("m:tldr"),
        roles=_pd_roles(),
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    )
    model_in = ModelUpdate(
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        lora_list=_adapters("m:tldr"),
        roles=None,
        disaggregation=None,
    )
    validate_roles(model_in, stored=stored)


# --- the role level ------------------------------------------------------- #


def test_adapters_declared_on_a_role_are_refused():
    """Without an explicit `RoleSpec.lora_list` field the adapters would be
    dropped by pydantic's `extra="ignore"` and the request would answer 200:
    no error, no warning, and nothing in the read-back to distinguish it from a
    request that never carried them."""
    roles = _pd_roles(decode={"lora_list": _adapters("m:tldr")})
    with rejects("cannot declare LoRA adapters"):
        validate_roles(_model_in(roles=roles))


def test_the_role_level_refusal_names_the_role():
    roles = _pd_roles(prefill={"lora_list": _adapters("m:tldr")})
    with rejects("Role 'prefill'"):
        validate_roles(_model_in(roles=roles))


def test_a_role_declaring_adapters_is_refused_without_disaggregation_too():
    """Plain multi-role orchestration is reached through a router as well, so
    the wall is the same one and the rule cannot live inside the disaggregation
    block."""
    roles = _pd_roles(decode={"lora_list": _adapters("m:tldr")})
    with rejects("cannot declare LoRA adapters"):
        validate_roles(_model_in(roles=roles, disaggregation=False))


def test_a_role_carrying_an_empty_list_declares_nothing():
    roles = _pd_roles(decode={"lora_list": []})
    validate_roles(_model_in(roles=roles))


def test_the_field_exists_so_the_refusal_can_see_it():
    """The refusal is only possible because the field is declared: `RoleSpec`
    takes pydantic's default `extra="ignore"`, so an undeclared key is gone
    before any validation runs."""
    role = RoleSpec(name="decode", lora_list=_adapters("m:tldr"))
    assert role.lora_list is not None
    assert role.lora_list[0].lora_name == "m:tldr"
