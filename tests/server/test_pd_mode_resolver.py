from gpustack.schemas.models import BackendEnum, PDModeEnum
from gpustack.schemas.pd_mode_resolution import (
    PDModeIneligibleCode,
    PDModeUnresolvedCode,
)
from gpustack.server.pd_mode_resolver import resolve_pd_mode

VLLM = BackendEnum.VLLM.value
SGLANG = BackendEnum.SGLANG.value


def verdict(resolution, name):
    return next(option for option in resolution.options if option.name == name)


# ---- the three cells that ship -------------------------------------------


def test_vllm_on_nvidia_has_one_candidate_and_needs_no_question():
    resolution = resolve_pd_mode(VLLM, {"nvidia"})
    assert resolution.mode == PDModeEnum.VLLM_NIXL.value
    assert resolution.vendor == "nvidia"
    assert resolution.unresolved_reason is None
    assert verdict(resolution, PDModeEnum.VLLM_NIXL.value).recommended is True


def test_vllm_on_ascend_resolves_to_the_ascend_recipe():
    resolution = resolve_pd_mode(VLLM, {"ascend"})
    assert resolution.mode == PDModeEnum.VLLM_ASCEND_MOONCAKE.value
    assert resolution.vendor == "ascend"


def test_sglang_on_nvidia_is_the_only_ambiguous_cell_and_preferred_settles_it():
    """Two recipes fit, so this is the one cell where a tie-break is needed.
    Keeping it in the catalog (`preferred`) rather than here means neither the
    API nor the UI holds a "which one is better" rule."""
    resolution = resolve_pd_mode(SGLANG, {"nvidia"})
    assert resolution.mode == PDModeEnum.SGLANG_MOONCAKE.value
    assert verdict(resolution, PDModeEnum.SGLANG_NIXL.value).eligible is True
    assert verdict(resolution, PDModeEnum.SGLANG_NIXL.value).recommended is False


# ---- the cells it does not ship ------------------------------------------


def test_sglang_on_ascend_has_no_recipe_and_points_at_custom():
    resolution = resolve_pd_mode(SGLANG, {"ascend"})
    assert resolution.mode is None
    assert "custom" in resolution.unresolved_reason
    assert verdict(resolution, PDModeEnum.CUSTOM.value).eligible is True


def test_amd_has_no_recipe_and_points_at_custom():
    """No AMD recipe ships, and the NVIDIA recipes carry `gpu_filters`, so
    none of them is offered here."""
    resolution = resolve_pd_mode(VLLM, {"amd"})
    assert resolution.mode is None
    assert verdict(resolution, PDModeEnum.VLLM_NIXL.value).eligible is False
    assert "nvidia" in verdict(resolution, PDModeEnum.VLLM_NIXL.value).ineligible_reason


def test_custom_is_eligible_on_every_accelerator():
    """The load-bearing invariant: "no built-in recipe" must never become
    "no PD"."""
    for vendors in ({"nvidia"}, {"ascend"}, {"amd"}, {"metax"}, set()):
        for backend in (VLLM, SGLANG, "SomeBYOEngine", None):
            resolution = resolve_pd_mode(backend, vendors)
            assert verdict(resolution, PDModeEnum.CUSTOM.value).eligible is True


def test_custom_is_never_the_derived_answer():
    """It injects nothing, so choosing it is a decision about who supplies the
    connection parameters -- not something the platform can make for you."""
    for vendors in ({"nvidia"}, {"ascend"}, {"amd"}, set()):
        resolution = resolve_pd_mode(VLLM, vendors)
        assert resolution.mode != PDModeEnum.CUSTOM.value


# ---- mixed-vendor clusters ------------------------------------------------


def test_a_mixed_cluster_returns_the_candidates_and_no_answer():
    """A PD group cannot span vendors, so this is a placement question. The
    platform deliberately does not pick the larger partition: the user may
    want the idle one."""
    resolution = resolve_pd_mode(VLLM, {"nvidia", "ascend"})
    assert resolution.mode is None
    assert sorted(resolution.candidate_vendors) == ["ascend", "nvidia"]
    assert "cannot span vendors" in resolution.unresolved_reason


def test_choosing_a_vendor_settles_a_mixed_cluster():
    resolution = resolve_pd_mode(VLLM, {"nvidia", "ascend"}, vendor="ascend")
    assert resolution.mode == PDModeEnum.VLLM_ASCEND_MOONCAKE.value
    assert resolution.vendor == "ascend"
    # The NVIDIA recipe is out relative to the *chosen partition*, not the
    # whole cluster -- that is the difference from the unresolved case above.
    assert verdict(resolution, PDModeEnum.VLLM_NIXL.value).eligible is False


def test_a_mixed_cluster_with_only_one_usable_partition_still_resolves():
    """SGLang has no Ascend recipe, so a mixed cluster is unambiguous for it:
    only the NVIDIA partition can host the group."""
    resolution = resolve_pd_mode(SGLANG, {"nvidia", "ascend"})
    assert resolution.mode == PDModeEnum.SGLANG_MOONCAKE.value
    assert resolution.candidate_vendors == ["nvidia"]


def test_choosing_a_vendor_the_cluster_lacks_is_refused():
    resolution = resolve_pd_mode(VLLM, {"nvidia"}, vendor="ascend")
    assert resolution.mode is None
    assert "no ascend accelerator" in resolution.unresolved_reason


# ---- unknown accelerators -------------------------------------------------


def test_unknown_accelerators_decide_nothing_and_disable_nothing():
    """A cluster whose workers have not reported devices yet reads the same as
    one with none. Neither may be judged unable to run anything -- absence of
    evidence is not a mismatch, so scheduling gets the call."""
    resolution = resolve_pd_mode(VLLM, set())
    assert resolution.mode is None
    assert resolution.cluster_vendors == []
    for option in resolution.options:
        if VLLM in (get_backends(option.name) or [VLLM]):
            assert option.eligible is True


def get_backends(name):
    from gpustack.server.pd_mode_catalog import get_pd_mode

    mode = get_pd_mode(name)
    return mode.backends if mode else None


# ---- engine mismatch is reported, not hidden ------------------------------


def test_engine_mismatch_carries_its_reason():
    resolution = resolve_pd_mode(VLLM, {"nvidia"})
    sglang = verdict(resolution, PDModeEnum.SGLANG_MOONCAKE.value)
    assert sglang.eligible is False
    assert "SGLang" in sglang.ineligible_reason
    assert "custom" in sglang.ineligible_reason


# ---- every unresolved exit is translatable --------------------------------


def test_every_unresolved_exit_carries_a_code_the_ui_can_translate():
    """`unresolved_reason` is English prose assembled here, so a UI that
    rendered it verbatim put an English sentence inside a localized form. The
    code plus `unresolved_params` is the same statement in a shape the client
    looks up in its own catalog; the prose stays as the fallback for a client
    that predates the code.

    Asserted as a set over every exit rather than one case at a time: the
    failure this guards against is a *new* exit added with prose only, which a
    per-case test would not notice.
    """
    Code = PDModeUnresolvedCode
    cases = [
        # (resolution, expected code, params that must be present)
        (resolve_pd_mode(VLLM, set()), Code.VENDORS_UNKNOWN, {}),
        (
            resolve_pd_mode(VLLM, {"nvidia"}, vendor="ascend"),
            Code.VENDOR_NOT_IN_CLUSTER,
            {"vendor": "ascend", "vendors": "nvidia"},
        ),
        (
            resolve_pd_mode(SGLANG, {"ascend"}),
            Code.NO_BUILT_IN_RECIPE,
            {"backend": SGLANG, "vendors": "ascend"},
        ),
        (
            resolve_pd_mode(VLLM, {"ascend", "nvidia"}),
            Code.MULTIPLE_VENDORS,
            {"vendors": "ascend, nvidia"},
        ),
    ]
    for resolution, code, params in cases:
        assert resolution.mode is None
        assert resolution.unresolved_code == code
        # The prose is kept, not replaced: an older client still renders it.
        assert resolution.unresolved_reason
        for key, value in params.items():
            assert (resolution.unresolved_params or {})[key] == value


def test_a_resolved_answer_carries_no_code():
    resolution = resolve_pd_mode(VLLM, {"nvidia"})
    assert resolution.unresolved_code is None
    assert resolution.unresolved_params is None


def test_an_engine_the_request_omits_is_left_for_the_client_to_word():
    """`backend` is optional on the request. The server sends `''` rather than
    its own "this engine": that half-sentence is the client's to word, and a
    server-supplied English one would be the very mixing this code exists to
    end."""
    resolution = resolve_pd_mode(None, {"amd"})
    assert resolution.unresolved_code == PDModeUnresolvedCode.NO_BUILT_IN_RECIPE
    assert resolution.unresolved_params["backend"] == ""


# --- the verdict a client can translate ------------------------------------ #


def _options(resolution):
    return {o.name: o for o in resolution.options}


def test_an_engine_mismatch_carries_a_code_and_joined_params():
    """The prose is English and assembled here, so a client that renders it
    verbatim puts an English sentence inside a localized form. The code plus
    params is the same statement in a shape the client can look up."""
    resolution = resolve_pd_mode(backend="SGLang", cluster_vendors={"nvidia"})

    entry = _options(resolution)["vllm-nixl"]

    assert entry.eligible is False
    assert entry.ineligible_code == PDModeIneligibleCode.BACKEND_MISMATCH
    # Pre-joined: how a list of engines reads belongs here, not in every client.
    assert entry.ineligible_params["targets"] == "vLLM"
    assert entry.ineligible_params["backend"] == "SGLang"
    # The prose stays, for a client older than the code and for logs.
    assert "Requires vLLM" in entry.ineligible_reason


def test_an_accelerator_mismatch_says_which_side_the_statement_is_about():
    """`scope` is the half-sentence the client has to choose between, and it
    is not cosmetic: on a single-vendor cluster the vendor is derived, so the
    sentence is about the partition that was settled on. It only speaks about
    the cluster while no partition has been settled."""
    settled = _options(resolve_pd_mode(backend="vLLM", cluster_vendors={"ascend"}))[
        "vllm-nixl"
    ]

    assert settled.ineligible_code == PDModeIneligibleCode.VENDOR_MISMATCH
    assert settled.ineligible_params["runtime"] == "nvidia"
    assert settled.ineligible_params["vendors"] == "ascend"
    assert settled.ineligible_params["scope"] == "partition"

    # No partition can host this engine at all, so nothing was settled on and
    # the statement goes back to being about the cluster as a whole.
    unsettled = _options(
        resolve_pd_mode(backend="SGLang", cluster_vendors={"ascend", "amd"})
    )["sglang-mooncake"]

    assert unsettled.ineligible_params["scope"] == "cluster"
    assert unsettled.ineligible_params["vendors"] == "amd, ascend"


def test_an_eligible_entry_carries_neither():
    """None rather than an empty string: a client tests the code, and an empty
    one would render an empty note instead of no note."""
    entry = _options(resolve_pd_mode(backend="vLLM", cluster_vendors={"nvidia"}))[
        "vllm-nixl"
    ]

    assert entry.eligible is True
    assert entry.ineligible_code is None
    assert entry.ineligible_params is None
    assert entry.ineligible_reason is None
