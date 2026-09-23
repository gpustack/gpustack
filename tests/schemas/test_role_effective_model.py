"""Projecting a role's overrides onto a Model.

A RoleSpec field left as None inherits the Model field of the same name.
Nothing downstream performs that merge — the worker's start path and the
scheduler's filters/selectors/scorers all read Model-level fields directly —
so it happens once, at the two points where a Model is handed to those
readers. These tests pin the three things that make that safe: the merge is
correct, a role-less model is untouched, and the projection can never reach
the database.
"""

from datetime import datetime

import pytest

from gpustack.schemas.models import (
    role_takes_no_accelerator,
    ExtendedKVCacheConfig,
    GPUSelector,
    KVCacheModeEnum,
    Model,
    ModelPublic,
    RoleEffectiveModel,
    RoleSpec,
    find_role,
    role_effective_model,
)


def _model(**kwargs):
    base = dict(
        id=7,
        name="llama-70b-pd",
        source="local_path",
        local_path="/models/llama",
        backend="vLLM",
        backend_version="0.11.0",
        backend_parameters=["--model-level"],
        env={"MODEL_LEVEL": "1"},
        image_name="model/image",
        replicas=1,
        cluster_id=3,
    )
    base.update(kwargs)
    return Model(**base)


def _pd_roles():
    return [
        RoleSpec(
            name="prefill",
            replicas=3,
            backend_parameters=["--enforce-eager", "--max-num-batched-tokens=8192"],
            env={"HCCL_CONNECT_TIMEOUT": "120", "HCCL_BUFFSIZE": "2560"},
        ),
        RoleSpec(
            name="decode",
            replicas=1,
            backend_parameters=["--max-num-batched-tokens=120"],
            env={"HCCL_CONNECT_TIMEOUT": "1200", "HCCL_BUFFSIZE": "1024"},
        ),
        RoleSpec(name="router", dependencies=["prefill", "decode"]),
    ]


# --- the no-op baseline -------------------------------------------------


def test_a_model_without_roles_is_returned_unchanged():
    model = _model()
    assert role_effective_model(model, "prefill") is model
    assert role_effective_model(model, None) is model


def test_a_role_less_instance_of_a_role_bearing_model_is_unchanged():
    # An instance created before roles were added carries role=None.
    model = _model(roles=_pd_roles())
    assert role_effective_model(model, None) is model


def test_an_unknown_role_name_is_returned_unchanged():
    model = _model(roles=_pd_roles())
    assert role_effective_model(model, "encoder") is model


def test_find_role():
    model = _model(roles=_pd_roles())
    assert find_role(model, "decode").replicas == 1
    assert find_role(model, "encoder") is None
    assert find_role(model, None) is None
    assert find_role(_model(), "prefill") is None


# --- the merge ----------------------------------------------------------


def test_declared_fields_override_and_the_rest_inherit():
    model = _model(roles=_pd_roles())
    prefill = role_effective_model(model, "prefill")

    # Overridden.
    assert prefill.backend_parameters == [
        "--enforce-eager",
        "--max-num-batched-tokens=8192",
    ]
    assert prefill.env == {"HCCL_CONNECT_TIMEOUT": "120", "HCCL_BUFFSIZE": "2560"}
    # Inherited: the role declares neither.
    assert prefill.backend == "vLLM"
    assert prefill.backend_version == "0.11.0"
    assert prefill.image_name == "model/image"
    # Identity carried through, so downstream readers still know what this is.
    assert (prefill.id, prefill.name, prefill.cluster_id) == (7, "llama-70b-pd", 3)


def test_two_roles_of_one_model_get_different_values():
    # The whole point: on real hardware prefill and decode differ in nearly
    # every performance-related parameter.
    model = _model(roles=_pd_roles())
    prefill = role_effective_model(model, "prefill")
    decode = role_effective_model(model, "decode")

    assert prefill.env["HCCL_CONNECT_TIMEOUT"] == "120"
    assert decode.env["HCCL_CONNECT_TIMEOUT"] == "1200"
    assert prefill.backend_parameters != decode.backend_parameters


def test_a_role_declaring_nothing_inherits_everything():
    model = _model(roles=_pd_roles())
    router = role_effective_model(model, "router")

    assert router.backend_parameters == ["--model-level"]
    assert router.env == {"MODEL_LEVEL": "1"}
    assert router.image_name == "model/image"


def test_replicas_is_projected_from_the_role():
    # roles[].replicas is the only scaling truth; Model.replicas is a 0/1
    # deployment switch. Inside the read paths the role's count is what
    # decides GPUs-per-replica and the multi-replica overcommit rule.
    model = _model(roles=_pd_roles(), replicas=1)
    assert role_effective_model(model, "prefill").replicas == 3
    assert role_effective_model(model, "decode").replicas == 1
    assert model.replicas == 1


def test_a_stopped_group_still_projects_the_role_count():
    model = _model(roles=_pd_roles(), replicas=0)
    assert role_effective_model(model, "prefill").replicas == 3


def test_nested_override_objects_are_projected():
    roles = [
        RoleSpec(
            name="prefill",
            gpu_selector=GPUSelector(gpus_per_replica=2),
            extended_kv_cache=ExtendedKVCacheConfig(
                enabled=True, mode=KVCacheModeEnum.LOCAL
            ),
        ),
        RoleSpec(name="decode"),
    ]
    model = _model(roles=roles)
    prefill = role_effective_model(model, "prefill")

    assert prefill.gpu_selector.gpus_per_replica == 2
    assert prefill.extended_kv_cache.enabled is True
    # The sibling inherits the Model's (absent) values rather than prefill's.
    assert role_effective_model(model, "decode").gpu_selector is None


def test_every_role_override_field_is_projected():
    # Derived, not listed, so adding an override to RoleSpec cannot silently
    # fail to be projected. This test is what makes that claim true.
    #
    # Reads the production set rather than restating it: a copy here has to be
    # edited every time a role-own field is added, and the edit is indis-
    # tinguishable from the mistake this guards against. Importing it keeps the
    # coverage — a new field nobody declared as role-own still arrives in the
    # override set and still has to find a Model field — while dropping the
    # duplication that made the test fail for a correct change.
    from gpustack.schemas.models import _ROLE_OWN_FIELDS

    role_fields = set(RoleSpec.model_fields)
    model_fields = set(Model.model_fields)
    for field in role_fields - _ROLE_OWN_FIELDS:
        assert field in model_fields, f"RoleSpec.{field} has no Model field to override"


# --- the projection must never reach the database -----------------------


def test_the_projection_is_not_an_orm_object():
    model = _model(roles=_pd_roles())
    projected = role_effective_model(model, "prefill")

    assert isinstance(projected, RoleEffectiveModel)
    assert not isinstance(projected, Model)
    # A Model carries SQLAlchemy identity state; a projection must not, or a
    # session flush could write a role's overrides onto the Model row.
    assert hasattr(model, "_sa_instance_state")
    assert not hasattr(projected, "_sa_instance_state")


def test_model_copy_would_have_shared_the_orm_identity():
    # Pins the reason RoleEffectiveModel exists: the obvious implementation is
    # unsafe, and silently so.
    model = _model(roles=_pd_roles())
    shallow = model.model_copy()
    assert shallow._sa_instance_state is model._sa_instance_state


def test_projecting_does_not_mutate_the_source():
    model = _model(roles=_pd_roles())
    before_params = list(model.backend_parameters)
    before_env = dict(model.env)

    role_effective_model(model, "prefill")

    assert model.backend_parameters == before_params
    assert model.env == before_env
    assert model.replicas == 1


def test_mutating_a_projected_list_does_not_reach_the_role():
    # The worker substitutes {data_dir} into backend_parameters in place.
    model = _model(roles=_pd_roles())
    projected = role_effective_model(model, "prefill")

    projected.backend_parameters[0] = "--rewritten"

    assert find_role(model, "prefill").backend_parameters[0] == "--enforce-eager"


def test_the_projection_carries_no_aggregate_status():
    # state / role_status / degradations are model-wide aggregates. A worker or
    # a scheduling pass acting on them would be reading the wrong thing.
    for field in ("state", "role_status", "degradations", "stale"):
        assert field in Model.model_fields
        assert field not in RoleEffectiveModel.model_fields


# --- the worker's view --------------------------------------------------


@pytest.mark.parametrize("role_name", ["prefill", "decode", "router"])
def test_a_model_public_projects_the_same_way(role_name):
    # The worker reads the model over the API, so it holds a ModelPublic
    # rather than a Model.
    model = _model(roles=_pd_roles())
    now = datetime(2026, 8, 21, 10, 0, 0)
    public = ModelPublic.model_validate(
        model, update={"created_at": now, "updated_at": now}
    )
    projected = role_effective_model(public, role_name)
    expected = role_effective_model(model, role_name)

    assert projected.backend_parameters == expected.backend_parameters
    assert projected.env == expected.env
    assert projected.replicas == expected.replicas


def test_the_projection_is_unhashable_just_like_a_model():
    # Not an oversight: SQLModel sets __hash__ = None on a table class, so
    # `Model` is unhashable too (ModelInstance overrides it to go into a
    # queue). A projection behaving the same way means a reader that starts
    # hashing models fails for both, not only for role-bearing deployments.
    model = _model(roles=_pd_roles())
    projected = role_effective_model(model, "prefill")

    assert Model.__hash__ is None
    with pytest.raises(TypeError):
        {projected}
    with pytest.raises(TypeError):
        {model}


# --- which roles take no accelerator --------------------------------------- #


def test_a_managed_router_takes_no_accelerator_without_being_told():
    """It is a proxy: it forwards to the members holding the weights and loads
    none itself. Relying on the `cpu_only` flag alone was measured to leave the
    router unschedulable on a two-card host whose cards its own prefill and
    decode had just filled — and the deploy form only registers that flag on
    the hand-written branch, so it is False in every group the UI produces."""
    model = _model(
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="router", replicas=1),
        ]
    )

    assert role_takes_no_accelerator(model, "router") is True
    assert role_takes_no_accelerator(model, "prefill") is False


def test_a_user_supplied_router_takes_no_accelerator_either():
    """A router cannot ask for a card, however it was supplied. The only sizing
    reachable on that path is `estimate_model_vram`, which returns the MODEL'S
    WEIGHTS — so «a custom router with a GPU» would book a proxy at 164 GiB for
    a 72B model, with no per-role way to override it
    (`GPUSTACK_MODEL_VRAM_CLAIM` is model-level and would mis-size prefill and
    decode too).

    So the answer is the role's name, image and command or not. A GPU-bearing
    role that is neither prefill nor decode comes back as a new role NAME."""
    model = _model(
        roles=[
            RoleSpec(
                name="router",
                replicas=1,
                image_name="me/router:1",
                run_command="my-router",
            )
        ]
    )
    assert role_takes_no_accelerator(model, "router") is True


def test_a_stale_cpu_only_on_a_gpu_role_is_read_straight_past():
    """`RoleSpec` takes pydantic's default `extra="ignore"`, so an old row or
    an old client still sending the flag loads without error — and no longer
    has any effect. Worth pinning because the flag was accepted on ANY role:
    `cpu_only: true` on prefill was never refused at admission, and it
    produced a prefill placed with no card. That footgun closes here."""
    model = _model(
        roles=[
            RoleSpec.model_validate(
                {"name": "prefill", "replicas": 1, "cpu_only": True}
            )
        ]
    )
    assert role_takes_no_accelerator(model, "prefill") is False


def test_a_role_less_model_takes_accelerators():
    model = _model()
    assert role_takes_no_accelerator(model, None) is False
    assert role_takes_no_accelerator(model, "router") is False
