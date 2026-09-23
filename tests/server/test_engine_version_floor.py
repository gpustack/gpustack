"""A PD recipe's declared engine range, and what it binds.

`backend_versions` in `pd-modes.yaml` is checked rather than informational: a
model pinned to an engine below its recipe's floor is reported, not silently
accepted.

The `>=0.5.7` on the SGLang recipes is a correctness floor. A member's id
became a registry-minted UUID at that version (2a098200), so an older build
answers 400 to `DELETE /workers/{url}` — a scaled-down member stays in the
router's registry and keeps taking traffic while GPUStack reports it gone.

Reported and not refused, which is the deliberate difference from the cache
provider's `versions` next door in `create_model`: that one rejects with a 400
because the engine would not start at all, whereas here the group runs and the
number may belong to a self-built image whose private version carries the fix.
"""

from types import SimpleNamespace

from gpustack.schemas.models import (
    DegradationReasonEnum,
    DisaggregationSpec,
    PDModeEnum,
    RoleSpec,
)
from gpustack.server.controllers import _engine_version_below_recipe_floor


def _model(mode, backend_version=None, roles=None):
    # A real `DisaggregationSpec`, because the mode name is what the catalog is
    # looked up by — a stub that stringifies differently would pass by finding
    # no recipe at all, which is the same answer this function gives when it is
    # working.
    return SimpleNamespace(
        disaggregation=(DisaggregationSpec(mode=mode) if mode is not None else None),
        backend_version=backend_version,
        roles=roles,
    )


SGLANG = PDModeEnum.SGLANG_MOONCAKE
VLLM = PDModeEnum.VLLM_NIXL


def test_a_version_under_the_floor_is_reported():
    """0.5.5 on SGLang: the build where `DELETE /workers/{id}` does not exist
    yet, so every scale-down leaves a member serving."""
    assert _engine_version_below_recipe_floor(_model(SGLANG, "0.5.5")) is True


def test_a_version_at_the_floor_is_not():
    assert _engine_version_below_recipe_floor(_model(SGLANG, "0.5.7")) is False


def test_a_version_above_the_floor_is_not():
    assert _engine_version_below_recipe_floor(_model(SGLANG, "0.6.1")) is False


def test_an_unpinned_version_sets_nothing():
    """Not "unknown, assume bad". An unpinned deployment takes whatever the
    image ships, which is normally the current one — marking it would put the
    badge on the default way of deploying."""
    assert _engine_version_below_recipe_floor(_model(SGLANG)) is False


def test_an_unparseable_version_fails_open():
    """The same failure mode the cache provider's range takes, and for the same
    reason: a private version string on a self-built image must never be the
    thing that condemns a deployment. `version_in_range` answers None here, and
    only a positive `False` counts.

    The last three are real strings off real images, kept here so the pass the
    local-version and pre-release carve-outs added does not quietly become the
    only way an exotic string survives — these never reach that predicate at
    all, and must keep not reaching it."""
    assert _engine_version_below_recipe_floor(_model(SGLANG, "not-a-version")) is False
    assert (
        _engine_version_below_recipe_floor(
            _model(SGLANG, "0.23.0-ascend-router-custom")
        )
        is False
    )
    assert _engine_version_below_recipe_floor(_model(SGLANG, "latest")) is False
    assert (
        _engine_version_below_recipe_floor(_model(SGLANG, "0.8.0-2.8-cu128")) is False
    )


def test_a_local_version_under_the_floor_is_left_alone():
    """The self-built image the docstring promises not to condemn, in the
    form that parses.

    `0.5.6+ourfix` sorts under 0.5.7, but `+local` means in PEP 440's own
    vocabulary "the official 0.5.6 with something of mine on top" — and backporting the very fix `>=0.5.7` asks for is the usual
    reason to cut one. Passing it is the platform saying it cannot tell, not
    that the build is sound."""
    assert _engine_version_below_recipe_floor(_model(SGLANG, "0.5.6+ourfix")) is False


def test_a_pre_release_of_the_floor_itself_is_left_alone():
    """`0.5.7-rc1` parses to `0.5.7rc1`, which PEP 440 sorts *before* 0.5.7 — so
    a naive comparison reports the rc of the very release the floor names as
    under it. Whoever is running an rc is running the code 0.5.7 became, and the version
    string cannot say which commits made it in."""
    assert _engine_version_below_recipe_floor(_model(SGLANG, "0.5.7-rc1")) is False


def test_a_dev_release_is_a_pre_release_and_is_left_alone_too():
    """`packaging` reports `is_prerelease` True for `0.5.6.dev0`, and that is
    the right answer here and not an accident of the library: a `.dev` build is
    cut off a branch rather than off a release, which is the same thing the
    floor cannot see into."""
    assert _engine_version_below_recipe_floor(_model(SGLANG, "0.5.6.dev0")) is False


def test_a_plain_release_under_the_floor_is_still_reported():
    """The line that must not move. Nothing about `0.5.5` or `0.5.6` claims
    to be anything but the release it names, so the floor ranks them and the
    answer is the one it was built to give."""
    assert _engine_version_below_recipe_floor(_model(SGLANG, "0.5.5")) is True
    assert _engine_version_below_recipe_floor(_model(SGLANG, "0.5.6")) is True


def test_a_role_on_a_local_version_does_not_drag_a_clean_group_down():
    """Per role on the way in, so per role on the way out as well: the decode
    carrying a private build is the member the check would otherwise mark the
    whole group for."""
    roles = [
        RoleSpec(name="prefill"),
        RoleSpec(name="decode", backend_version="0.5.6+ourfix"),
    ]
    model = _model(SGLANG, backend_version="0.6.0", roles=roles)
    assert _engine_version_below_recipe_floor(model) is False


def test_a_nonexistent_but_parseable_version_is_above_the_floor():
    """9.9.9 does not exist, and this deliberately says nothing about that: the
    marker answers "below the declared floor", not "is a real release". Naming
    releases is not something the catalog can do without a network call."""
    assert _engine_version_below_recipe_floor(_model(SGLANG, "9.9.9")) is False


def test_the_vllm_recipe_has_its_own_floor():
    assert _engine_version_below_recipe_floor(_model(VLLM, "0.19.0")) is True
    assert _engine_version_below_recipe_floor(_model(VLLM, "0.20.0")) is False


def test_a_role_pinning_its_own_old_version_is_reported():
    """Per role, not per model. `_build_instance_create` takes
    `role.backend_version or model.backend_version`, so a group whose
    model-level pin is fine can still run one decode on a build that cannot be
    scaled down — and that member is the one that keeps serving after removal,
    so one is enough."""
    roles = [
        RoleSpec(name="prefill"),
        RoleSpec(name="decode", backend_version="0.5.5"),
    ]
    model = _model(SGLANG, backend_version="0.6.0", roles=roles)
    assert _engine_version_below_recipe_floor(model) is True


def test_a_role_pinning_a_good_version_over_a_bad_model_one_still_reports():
    """The model-level pin is what every role without an override inherits, so
    one role climbing out of the hole does not fill it in."""
    roles = [
        RoleSpec(name="prefill", backend_version="0.6.0"),
        RoleSpec(name="decode", backend_version="0.6.0"),
    ]
    model = _model(SGLANG, backend_version="0.5.5", roles=roles)
    assert _engine_version_below_recipe_floor(model) is True


def test_roles_inheriting_a_good_model_version_report_nothing():
    roles = [RoleSpec(name="prefill"), RoleSpec(name="decode")]
    model = _model(SGLANG, backend_version="0.6.0", roles=roles)
    assert _engine_version_below_recipe_floor(model) is False


def test_a_router_pinning_its_own_image_version_is_not_the_engine():
    """`backend_versions` describes the ENGINE, and the router's version is
    not the engine's.

    The router is the one role whose engine is genuinely its own — the deploy
    form keeps an image-and-version section for it for exactly that reason ("a
    `vllm-router` is not the model's engine"), and a hand-written router image
    carries whatever version its author gave it. Compared against `>=0.5.7`
    that number answers a different question, and letting it in would condemn a
    deployment whose engine is in range over a router that was never described
    by the range at all.
    """
    roles = [
        RoleSpec(name="prefill"),
        RoleSpec(name="decode"),
        RoleSpec(name="router", backend_version="0.1.0"),
    ]
    model = _model(SGLANG, backend_version="0.6.0", roles=roles)
    assert _engine_version_below_recipe_floor(model) is False


def test_custom_declares_no_floor_so_there_is_none_to_be_under():
    """`custom` injects nothing, so there is no recipe-owned expectation of the
    engine. No declaration means "no answer", never "compatible"."""
    model = _model(PDModeEnum.CUSTOM, backend_version="0.0.1")
    assert _engine_version_below_recipe_floor(model) is False


def test_a_model_that_is_not_disaggregated_has_no_recipe():
    assert _engine_version_below_recipe_floor(_model(None, "0.5.5")) is False


def test_the_reason_is_its_own_enum_value():
    assert (
        DegradationReasonEnum.ENGINE_VERSION_BELOW_RECIPE_FLOOR.value
        == "engine_version_below_recipe_floor"
    )
