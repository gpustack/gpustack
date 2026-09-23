"""The model-level override of the cluster's gather default.

Two things are worth pinning, and neither is about the happy path. The pair
`(strategy, layer)` is only meaningful in two of its four shapes, and a
half-declared pair is silently stood down by the solver rather than refused —
so the edge has to refuse it. And `gather` must stay *out* of the spec digest,
because a deployment form that promises "running groups are not moved" cannot
be backed by a field that restarts every member.
"""

import pytest

from gpustack.schemas.clusters import GatherStrategyEnum
from gpustack.schemas.models import GatherSpec, Model, ModelSpecBase
from gpustack.server.controllers import _DIGEST_EXCLUDED_SPEC_FIELDS
from gpustack.topology.tree import NODE_LAYER
from tests.utils.topology_layers import layer_dict, lid


def test_no_gather_at_all_means_inherit():
    """NULL is the correct value, not a missing one: absent means the model
    inherits the cluster's default."""
    assert Model(name="m1").gather is None


def test_prefer_gather_needs_no_layer():
    """`PreferGather` widens to the cluster root, so there is no layer for it
    to stop at."""
    spec = GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER)
    assert spec.layer is None


def test_must_gather_carries_the_layer_it_stops_at():
    spec = GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer="Rack")
    assert spec.strategy == GatherStrategyEnum.MUST_GATHER
    assert spec.layer == "Rack"


def test_must_gather_without_a_layer_is_refused():
    """The failure mode this prevents is not a crash, it is a promise that was
    never in force: the solver's `_enforced_gather` stands an unnamed
    requirement down, so the deployment would be accepted under a constraint
    nothing enforces."""
    with pytest.raises(ValueError, match="requires a layer"):
        GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER)


def test_a_layer_without_a_strategy_is_refused():
    """A layer alone says where but not whether. Accepting it would store an
    intent no code reads."""
    with pytest.raises(ValueError, match="nothing to apply it to"):
        GatherSpec(layer="Rack")


def test_any_rung_is_named_the_same_way_including_the_accelerator_domain():
    """There is no `accelerator_domain` special value any more — the name is
    accepted here exactly like `Rack` is, as a plain layer id, and whether the
    cluster actually has a rung by that name is checked against the cluster
    (`routes.models.validate_gather_layer`). The schema cannot do it: it has no
    cluster in hand."""
    spec = GatherSpec(
        strategy=GatherStrategyEnum.MUST_GATHER, layer="accelerator_domain"
    )
    assert spec.layer == "accelerator_domain"
    assert not hasattr(spec, "chain")


def test_the_wire_form_accepts_the_enum_values_verbatim():
    """The two spellings the form sends."""
    assert (
        GatherSpec.model_validate({"strategy": "PreferGather"}).strategy
        == GatherStrategyEnum.PREFER_GATHER
    )
    assert (
        GatherSpec.model_validate({"strategy": "MustGather", "layer": "Rack"}).strategy
        == GatherStrategyEnum.MUST_GATHER
    )


def test_gather_round_trips_on_the_model():
    model = Model(
        name="m1",
        gather=GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer="Rack"),
    )
    assert model.gather.layer == "Rack"


# ---------------------------------------------------------------------------
# The layer has to name a rung the cluster actually has.
# ---------------------------------------------------------------------------


async def _validate_gather(layer, topology):
    """Run the route-level check against a stubbed cluster."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from gpustack.routes.models import validate_gather_layer

    cluster = SimpleNamespace(id=1, topology=topology)
    with patch(
        "gpustack.schemas.clusters.Cluster.one_by_id",
        new=AsyncMock(return_value=cluster),
    ):
        await validate_gather_layer(
            None,
            Model(
                name="m1",
                gather=GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=layer),
            ),
            cluster_id=1,
        )


@pytest.mark.asyncio
async def test_a_gather_layer_the_cluster_does_not_have_is_refused():
    """The silence this closes. The solver's `_enforced_gather` *ignores* an
    unknown layer and places the group anyway — right at schedule time, wrong
    at submit time, where it would accept a `MustGather` under a promise
    nothing enforces. `accelerator_domain` is the name operators most often
    expect to be valid everywhere, so it is the one worth naming in the
    test."""
    from gpustack.api.exceptions import BadRequestException
    from gpustack.schemas.clusters import ClusterTopology

    with pytest.raises(BadRequestException) as refused:
        await _validate_gather(lid("accelerator_domain"), None)
    assert "not a layer of this cluster" in refused.value.message

    # Declared, and it is accepted — the rung exists now.
    declared = ClusterTopology.model_validate(
        {
            "layers": [
                layer_dict(
                    "accelerator_domain", ["topology.gpustack.ai/accelerator-domain"]
                )
            ]
        }
    )
    await _validate_gather(lid("accelerator_domain"), declared)


@pytest.mark.asyncio
async def test_the_host_is_accepted_without_the_cluster_declaring_anything():
    """The leaf is built in, so the tightest choice must never need a lookup."""
    await _validate_gather(NODE_LAYER, None)


@pytest.mark.asyncio
async def test_a_builtin_rung_is_accepted_by_every_cluster():
    await _validate_gather(lid("rack"), None)


def test_gather_is_excluded_from_the_spec_digest():
    """The regression this file exists for.

    `model_spec_digest` walks *every* field of `ModelSpecBase` minus the
    exclusion set, so adding a field opts it in by default. Gather is a
    preference for the next scheduling decision and nothing re-places a
    running group — folding it in would restart every member to relocate none
    of them, and would make the form's own sentence ("only affects later
    scheduling; running groups are not moved") false.
    """
    assert "gather" in ModelSpecBase.model_fields
    assert "gather" in _DIGEST_EXCLUDED_SPEC_FIELDS


# ---------------------------------------------------------------------------
# Speculative decoding is per role, and the reason is not the obvious one.
# ---------------------------------------------------------------------------


def test_speculative_config_is_overridable_per_role():
    """The regression this pins, and the reasoning that got it wrong once.

    The first reading was "prefill does not decode, so a draft model is pure
    waste there — switch it off". That is backwards. What the NIXL handshake
    hashes is the *model* (`model`, `num_hidden_layers`, `num_kv_heads`,
    `head_size`), and for MTP-style speculation the draft head is part of the
    model — so a prefill that does not load it produces a different structure
    and fails the compatibility check. Upstream's recipes say it in numbers:
    prefill runs 1 draft token, decode runs 3 or more.

    So one model-level value cannot serve both roles, which is exactly what
    `RoleSpec` carrying the field fixes.
    """
    from gpustack.schemas.models import (
        RoleSpec,
        SpeculativeConfig,
        _ROLE_OVERRIDE_FIELDS,
        role_effective_model,
    )

    assert "speculative_config" in _ROLE_OVERRIDE_FIELDS

    model = Model(
        name="pd",
        source="huggingface",
        huggingface_repo_id="x/y",
        # The model-level value is what a role without its own opinion runs.
        speculative_config=SpeculativeConfig(enabled=True, num_draft_tokens=1),
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(
                name="decode",
                replicas=2,
                speculative_config=SpeculativeConfig(enabled=True, num_draft_tokens=3),
            ),
        ],
    )

    assert (
        role_effective_model(model, "prefill").speculative_config.num_draft_tokens == 1
    )
    assert (
        role_effective_model(model, "decode").speculative_config.num_draft_tokens == 3
    )


def test_a_role_without_its_own_speculative_config_inherits():
    """Absent means inherit, not "off". A non-MTP draft model is the case
    where prefill genuinely gains nothing — the field makes the split
    possible, it does not force it."""
    from gpustack.schemas.models import (
        RoleSpec,
        SpeculativeConfig,
        role_effective_model,
    )

    model = Model(
        name="pd",
        source="huggingface",
        huggingface_repo_id="x/y",
        speculative_config=SpeculativeConfig(enabled=True, num_draft_tokens=2),
        roles=[RoleSpec(name="prefill", replicas=1)],
    )
    projected = role_effective_model(model, "prefill")
    assert projected.speculative_config.num_draft_tokens == 2


def test_a_role_speculative_override_changes_the_generation():
    """It is container shape, so it belongs in the spec digest — unlike
    `gather`, changing it means the member must restart to take effect."""
    from gpustack.schemas.models import RoleSpec, SpeculativeConfig
    from gpustack.server.controllers import _role_digest_payload

    plain = _role_digest_payload(RoleSpec(name="decode", replicas=1))
    speculating = _role_digest_payload(
        RoleSpec(
            name="decode",
            replicas=1,
            speculative_config=SpeculativeConfig(enabled=True, num_draft_tokens=3),
        )
    )
    assert plain != speculating


# ---------------------------------------------------------------------------
# «Every member on one machine» against a member that needs two.
# ---------------------------------------------------------------------------


async def _validate_host_floor(
    params, workers, strategy=GatherStrategyEnum.MUST_GATHER, backend="vLLM"
):
    """Run the route-level check for a floor at the built-in host rung."""
    from unittest.mock import AsyncMock, patch

    from gpustack.routes.models import validate_gather_layer
    from gpustack.schemas.models import RoleSpec
    from gpustack.topology.tree import NODE_LAYER

    model = Model(
        name="m1",
        source="huggingface",
        huggingface_repo_id="x/y",
        backend=backend,
        backend_parameters=params,
        gather=GatherSpec(strategy=strategy, layer=NODE_LAYER),
    )
    model.cluster_id = 1
    model.roles = [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1),
    ]
    with patch(
        "gpustack.schemas.workers.Worker.all_by_field",
        new=AsyncMock(return_value=workers),
    ):
        await validate_gather_layer(None, model, cluster_id=1)


def _host(cards):
    from types import SimpleNamespace

    return SimpleNamespace(
        id=1,
        status=SimpleNamespace(
            gpu_devices=[SimpleNamespace(index=i) for i in range(cards)]
        ),
    )


@pytest.mark.asyncio
async def test_a_host_floor_and_a_member_that_needs_two_hosts_is_refused():
    """The one contradiction cross-machine members introduce, and it is two
    things the operator asked for rather than a capacity shortfall.

    Left to the solver it surfaces as a group that never leaves PENDING beside
    a sentence about capacity — on a cluster that is not short of any. Caught
    here it is one sentence naming the two settings that disagree, while both
    are still on screen."""
    from gpustack.api.exceptions import BadRequestException

    with pytest.raises(BadRequestException) as refused:
        await _validate_host_floor(["--tensor-parallel-size=16"], [_host(8)])

    assert "needs 16 GPUs" in refused.value.message
    assert "widest worker" in refused.value.message


@pytest.mark.asyncio
async def test_a_member_that_fits_one_host_is_accepted():
    await _validate_host_floor(["--tensor-parallel-size=8"], [_host(8)])


@pytest.mark.asyncio
async def test_prefer_gather_at_the_host_rung_is_a_target_not_a_promise():
    """It is allowed to be missed, and says so afterwards. Refusing here would
    turn "aim for this" into "refuse below this", which is the other option and
    the one the operator did not pick."""
    await _validate_host_floor(
        ["--tensor-parallel-size=16"],
        [_host(8)],
        strategy=GatherStrategyEnum.PREFER_GATHER,
    )


@pytest.mark.asyncio
async def test_a_width_the_engine_was_never_told_is_left_alone():
    """A refusal is only worth issuing when the contradiction is certain."""
    await _validate_host_floor([], [_host(8)])


@pytest.mark.asyncio
async def test_a_cluster_with_no_readable_workers_is_left_alone():
    await _validate_host_floor(["--tensor-parallel-size=16"], [])


@pytest.mark.asyncio
async def test_sglang_spells_the_same_width_differently():
    """`--tp-size` rather than `--tensor-parallel-size`, which is why the width
    is asked of the selectors instead of re-read here."""
    from gpustack.api.exceptions import BadRequestException

    with pytest.raises(BadRequestException):
        await _validate_host_floor(["--tp-size=16"], [_host(8)], backend="SGLang")
