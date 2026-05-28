"""
Unit tests for vLLM multi-node topology inference.

Covers:
- world_size formula regression (Issue #5089).
- ``cal_multinode_topology`` resolver across the three deployment shapes
  (``dp_only`` / ``mp_only`` / ``nested``).
- Heterogeneous worker groups (vLLM allows per-node DP-Local).
- Failure paths: TP/PP that cannot fit, mismatched user-supplied DP, etc.
"""

from unittest.mock import MagicMock

import pytest

from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    VLLMResourceFitSelector,
)
from gpustack.utils.vllm_topology import (
    MultinodeUserParallelism,
    ValidatedTopology,
    validate_multinode_topology,
)
from gpustack.worker.backends.vllm import (
    MultinodeTopology,
    cal_multinode_topology,
)


def _model(params):
    m = MagicMock()
    m.backend_parameters = params
    return m


def _instance(g_main, *subordinate_gs):
    mi = MagicMock()
    mi.gpu_indexes = list(range(g_main))
    sub = []
    for g in subordinate_gs:
        s = MagicMock()
        s.gpu_indexes = list(range(g))
        sub.append(s)
    mi.distributed_servers.subordinate_workers = sub
    return mi


def _meta(role, follower_index=None):
    """role: 'leader' or 'follower'. follower_index is 0-based."""
    m = MagicMock()
    m.distributed_leader = role == "leader"
    m.distributed_follower = role == "follower"
    m.distributed_follower_index = follower_index
    return m


# ---------------------------------------------------------------------------
# Issue #5089 regression
# ---------------------------------------------------------------------------


def test_world_size_5089_multi_line():
    """Regression: TP=8 DP=2 DP-Local=1 split across tokens yields world_size=16."""
    params = [
        "--tensor-parallel-size",
        "8",
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "1",
    ]
    world_size, strategies = (
        VLLMResourceFitSelector.get_world_size_from_backend_parameters(_model(params))
    )
    assert world_size == 16
    assert strategies == ["tp", "dp", "dpl"]


def test_world_size_dpl_does_not_multiply():
    """world_size = TP * PP * DP; DP-Local appears only in strategies."""
    params = ["--tp", "4", "--dp", "2", "--dpl", "2"]
    world_size, _ = VLLMResourceFitSelector.get_world_size_from_backend_parameters(
        _model(params)
    )
    assert world_size == 8


def test_world_size_only_tp():
    world_size, strategies = (
        VLLMResourceFitSelector.get_world_size_from_backend_parameters(
            _model(["--tensor-parallel-size", "4"])
        )
    )
    assert world_size == 4
    assert strategies == ["tp"]


def test_world_size_tp_pp_dp():
    world_size, strategies = (
        VLLMResourceFitSelector.get_world_size_from_backend_parameters(
            _model(["--tp", "4", "--pp", "2", "--dp", "3"])
        )
    )
    assert world_size == 24
    assert strategies == ["tp", "pp", "dp"]


def test_world_size_no_parallelism():
    world_size, strategies = (
        VLLMResourceFitSelector.get_world_size_from_backend_parameters(
            _model(["--max-model-len", "8192"])
        )
    )
    assert world_size is None
    assert strategies is None


# ---------------------------------------------------------------------------
# shape == "dp_only" — homogeneous clusters
# ---------------------------------------------------------------------------


def test_dp_only_homogeneous_defaults_leader():
    """2 nodes x 8 GPU, no user args → TP=8, DP=2, DPL=1, leader start_rank=0."""
    out = cal_multinode_topology(_instance(8, 8), _meta("leader"))
    assert out == MultinodeTopology(
        shape="dp_only",
        tp=8,
        pp=1,
        dp=2,
        dpl=1,
        nnodes=2,
        node_rank=0,
        start_rank=0,
        is_follower=False,
    )


def test_dp_only_homogeneous_defaults_follower():
    """2 nodes x 8 GPU, follower index 0 → start_rank=1 (cumulative DPL)."""
    out = cal_multinode_topology(_instance(8, 8), _meta("follower", 0))
    assert out.shape == "dp_only"
    assert out.start_rank == 1
    assert out.node_rank == 1
    assert out.is_follower is True


def test_dp_only_user_tp_4_dpl_2():
    """User TP=4 on 2x8 → per-node DPL=2, DP=4."""
    out = cal_multinode_topology(
        _instance(8, 8), _meta("leader"), MultinodeUserParallelism(tp=4)
    )
    assert out.shape == "dp_only"
    assert (out.tp, out.dp, out.dpl, out.start_rank) == (4, 4, 2, 0)


def test_dp_only_user_tp_4_follower_start_rank():
    """User TP=4 on 2x8, follower index 0 → start_rank = leader's DPL = 2."""
    out = cal_multinode_topology(
        _instance(8, 8), _meta("follower", 0), MultinodeUserParallelism(tp=4)
    )
    assert out.shape == "dp_only"
    assert (out.tp, out.dp, out.dpl, out.start_rank) == (4, 4, 2, 2)


def test_dp_only_single_node():
    out = cal_multinode_topology(_instance(8), _meta("leader"))
    # Single-node still resolves to dp_only — there is no PP/TP cross-node split.
    assert out.shape == "dp_only"
    assert (out.dp, out.dpl, out.start_rank, out.nnodes) == (1, 1, 0, 1)


def test_dp_only_three_nodes_start_ranks():
    """3 nodes x 8 GPU → start_ranks 0,1,2."""
    inst = _instance(8, 8, 8)
    assert cal_multinode_topology(inst, _meta("leader")).start_rank == 0
    assert cal_multinode_topology(inst, _meta("follower", 0)).start_rank == 1
    assert cal_multinode_topology(inst, _meta("follower", 1)).start_rank == 2


# ---------------------------------------------------------------------------
# shape == "dp_only" — heterogeneous clusters
# ---------------------------------------------------------------------------


def test_dp_only_heterogeneous_default_tp_gcd():
    """8 + 4 GPU mix → default TP = gcd(8,4) = 4, per-node DPL = (2, 1)."""
    inst = _instance(8, 4)
    leader = cal_multinode_topology(inst, _meta("leader"))
    assert leader.shape == "dp_only"
    assert (leader.tp, leader.dp, leader.dpl, leader.start_rank) == (4, 3, 2, 0)

    follower = cal_multinode_topology(inst, _meta("follower", 0))
    assert follower.shape == "dp_only"
    assert (follower.dp, follower.dpl, follower.start_rank) == (3, 1, 2)


def test_dp_only_heterogeneous_user_tp_overrides():
    """User TP=2 on 8 + 4 mix → per-node DPL = (4, 2), DP=6."""
    inst = _instance(8, 4)
    leader = cal_multinode_topology(
        inst, _meta("leader"), MultinodeUserParallelism(tp=2)
    )
    follower = cal_multinode_topology(
        inst, _meta("follower", 0), MultinodeUserParallelism(tp=2)
    )
    assert (leader.tp, leader.dp, leader.dpl, leader.start_rank) == (2, 6, 4, 0)
    assert (follower.dp, follower.dpl, follower.start_rank) == (6, 2, 4)


def test_dp_only_heterogeneous_rejects_uniform_dpl():
    with pytest.raises(ValueError, match="heterogeneous worker group"):
        cal_multinode_topology(
            _instance(8, 4), _meta("leader"), MultinodeUserParallelism(dpl=1)
        )


def test_dp_only_heterogeneous_tp_must_divide_every_node():
    with pytest.raises(ValueError, match=r"worker\[1\] 4 GPUs"):
        cal_multinode_topology(
            _instance(8, 4), _meta("leader"), MultinodeUserParallelism(tp=8)
        )


# ---------------------------------------------------------------------------
# shape == "mp_only" — PP/TP spans multiple nodes, dp == 1
# ---------------------------------------------------------------------------


def test_mp_only_pp_across_two_nodes_leader():
    """TP=8 PP=2 on 2x8 GPU → workers_per_dp=16 > g_per_node=8 → mp_only."""
    inst = _instance(8, 8)
    out = cal_multinode_topology(
        inst, _meta("leader"), MultinodeUserParallelism(tp=8, pp=2)
    )
    assert out == MultinodeTopology(
        shape="mp_only",
        tp=8,
        pp=2,
        dp=1,
        dpl=1,
        nnodes=2,
        node_rank=0,
        start_rank=0,
        is_follower=False,
    )


def test_mp_only_pp_across_two_nodes_follower():
    inst = _instance(8, 8)
    out = cal_multinode_topology(
        inst, _meta("follower", 0), MultinodeUserParallelism(tp=8, pp=2)
    )
    assert out.shape == "mp_only"
    assert out.node_rank == 1
    assert out.is_follower is True


def test_mp_only_pp_three_nodes():
    """TP=4 PP=3 on 3x4 GPU."""
    inst = _instance(4, 4, 4)
    out = cal_multinode_topology(
        inst, _meta("follower", 1), MultinodeUserParallelism(tp=4, pp=3)
    )
    assert out.shape == "mp_only"
    assert out.nnodes == 3 and out.node_rank == 2


def test_mp_only_rejects_heterogeneous():
    """mp_only requires every node to contribute the same GPU count.

    Uses TP=4 PP=4 so both nodes pass the TP-divides-GPU check first, then
    the cross-node branch's homogeneity check is what fires.
    """
    inst = _instance(8, 4)
    with pytest.raises(ValueError, match="homogeneous"):
        cal_multinode_topology(
            inst, _meta("leader"), MultinodeUserParallelism(tp=4, pp=4)
        )


def test_mp_only_rejects_dp_gt_1_without_capacity():
    """workers_per_dp=16 > capacity=8 → nnodes_within_dp=2; cluster only has 2 nodes,
    so a single DP rank fills the cluster — dp=2 is impossible to fit."""
    inst = _instance(8, 8)
    with pytest.raises(ValueError, match="does not match"):
        cal_multinode_topology(
            inst,
            _meta("leader"),
            MultinodeUserParallelism(tp=8, pp=2, dp=2),
        )


# ---------------------------------------------------------------------------
# shape == "nested" — DP rank spans multiple nodes, dp > 1
# ---------------------------------------------------------------------------


def test_nested_dp_2_pp_2():
    """TP=4 PP=2 DP=2 on 4x4 GPU → workers_per_dp=8 > 4 → nested."""
    inst = _instance(4, 4, 4, 4)
    leader = cal_multinode_topology(
        inst, _meta("leader"), MultinodeUserParallelism(tp=4, pp=2, dp=2)
    )
    assert leader == MultinodeTopology(
        shape="nested",
        tp=4,
        pp=2,
        dp=2,
        dpl=1,
        nnodes=4,
        node_rank=0,
        start_rank=0,
        is_follower=False,
    )

    follower2 = cal_multinode_topology(
        inst, _meta("follower", 1), MultinodeUserParallelism(tp=4, pp=2, dp=2)
    )
    # node_rank=2, nnodes_within_dp=2 → node_rank_within_dp=0 → DP rank 1 head.
    assert follower2.shape == "nested"
    assert follower2.node_rank == 2
    # vLLM derives DP rank from node_rank // nnodes_within_dp internally;
    # we surface it here for diagnostics but do not emit --data-parallel-start-rank.
    assert follower2.start_rank == 1


def test_nested_rejects_non_divisible_nnodes():
    """3 nodes, workers_per_dp=8, capacity=4 → nnodes_within_dp=2 but 3 % 2 != 0."""
    inst = _instance(4, 4, 4)
    with pytest.raises(ValueError, match="cannot evenly distribute"):
        cal_multinode_topology(
            inst, _meta("leader"), MultinodeUserParallelism(tp=4, pp=2, dp=2)
        )


# ---------------------------------------------------------------------------
# Generic failure paths
# ---------------------------------------------------------------------------


def test_tp_not_divisor():
    with pytest.raises(ValueError, match=r"worker\[0\] 8 GPUs"):
        cal_multinode_topology(
            _instance(8, 8), _meta("leader"), MultinodeUserParallelism(tp=3)
        )


def test_dp_only_dp_mismatch():
    # Single-dp without tp now goes through tp inference (16 % 5 != 0),
    # so the failure is "cannot infer" rather than "does not match".
    with pytest.raises(ValueError, match="cannot infer"):
        cal_multinode_topology(
            _instance(8, 8), _meta("leader"), MultinodeUserParallelism(dp=5)
        )


def test_dp_only_explicit_tp_dp_mismatch():
    # When the user pins tp explicitly, dp inference is skipped and we fall
    # back to the original dp_total consistency check.
    with pytest.raises(ValueError, match="does not match"):
        cal_multinode_topology(
            _instance(8, 8),
            _meta("leader"),
            MultinodeUserParallelism(tp=8, dp=5),
        )


def test_dp_only_user_dpl_consistent():
    out = cal_multinode_topology(
        _instance(8, 8), _meta("leader"), MultinodeUserParallelism(tp=8, dpl=1)
    )
    assert out.shape == "dp_only"
    assert (out.tp, out.dp, out.dpl, out.start_rank) == (8, 2, 1, 0)


def test_follower_index_out_of_bounds():
    with pytest.raises(ValueError, match="out of bounds"):
        cal_multinode_topology(_instance(8, 8), _meta("follower", 5))


# ---------------------------------------------------------------------------
# validate_multinode_topology — pure function exposed to selector
# ---------------------------------------------------------------------------


def test_validate_dp_only_homogeneous():
    out = validate_multinode_topology([8, 8])
    assert out == ValidatedTopology(
        shape="dp_only",
        tp=8,
        pp=1,
        dp=2,
        dpl_per_node=[1, 1],
        nnodes_within_dp=1,
    )


def test_validate_dp_only_heterogeneous_dpl_per_node():
    """Heterogeneous cluster: dpl_per_node should differ per node."""
    out = validate_multinode_topology([8, 4], MultinodeUserParallelism(tp=4))
    assert out.shape == "dp_only"
    assert out.dpl_per_node == [2, 1]
    assert out.dp == 3


def test_validate_mp_only():
    out = validate_multinode_topology([8, 8], MultinodeUserParallelism(tp=8, pp=2))
    assert out == ValidatedTopology(
        shape="mp_only",
        tp=8,
        pp=2,
        dp=1,
        dpl_per_node=[1, 1],
        nnodes_within_dp=2,
    )


def test_validate_nested():
    out = validate_multinode_topology(
        [4, 4, 4, 4], MultinodeUserParallelism(tp=4, pp=2, dp=2)
    )
    assert out == ValidatedTopology(
        shape="nested",
        tp=4,
        pp=2,
        dp=2,
        dpl_per_node=[1, 1, 1, 1],
        nnodes_within_dp=2,
    )


@pytest.mark.parametrize(
    "gpu_per_node,user,expected_match",
    [
        # tp doesn't divide a node's GPU count.
        ([8, 8], MultinodeUserParallelism(tp=3), r"worker\[0\] 8 GPUs"),
        # User pins both tp and dp inconsistently.
        ([8, 8], MultinodeUserParallelism(tp=8, dp=5), "does not match"),
        # Heterogeneous cross-node TP/PP.
        ([8, 4], MultinodeUserParallelism(tp=4, pp=4), "homogeneous"),
        # nnodes_within_dp doesn't divide cluster size.
        (
            [4, 4, 4],
            MultinodeUserParallelism(tp=4, pp=2, dp=2),
            "cannot evenly distribute",
        ),
        # User dpl != 1 in cross-node layout.
        (
            [8, 8],
            MultinodeUserParallelism(tp=8, pp=2, dpl=2),
            "must be 1",
        ),
    ],
)
def test_validate_rejects(gpu_per_node, user, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        validate_multinode_topology(gpu_per_node, user)


def test_validate_default_tp_uses_gcd():
    """No user TP → default to gcd(gpu_per_node)."""
    out = validate_multinode_topology([8, 4])
    assert out.tp == 4
    assert out.dpl_per_node == [2, 1]


def test_validate_returns_tp_for_inference_by_caller():
    """selector can read out.tp to learn what GPUStack inferred."""
    out = validate_multinode_topology([16, 16], MultinodeUserParallelism())
    assert out.tp == 16
    assert out.shape == "dp_only"


# ---------------------------------------------------------------------------
# tp inference from user-supplied dp (vLLM-aligned semantics)
# ---------------------------------------------------------------------------


def test_validate_infers_tp_from_dp_only():
    """Cluster=[8, 8] + --dp 4 → tp=4 (16 GPUs / 4 DP ranks)."""
    out = validate_multinode_topology([8, 8], MultinodeUserParallelism(dp=4))
    assert out.tp == 4
    assert out.dp == 4
    assert out.shape == "dp_only"
    assert out.dpl_per_node == [2, 2]


def test_validate_infers_tp_from_dp_with_pp():
    """Cluster=[8, 8] + --dp 1 --pp 2 → tp=8, mp_only across both nodes."""
    out = validate_multinode_topology([8, 8], MultinodeUserParallelism(dp=1, pp=2))
    assert out.tp == 8
    assert out.pp == 2
    assert out.dp == 1
    assert out.shape == "mp_only"
    assert out.nnodes_within_dp == 2


def test_validate_infers_tp_yields_nested():
    """Cluster=[4, 4, 4, 4] + --dp 2 --pp 2 → tp=4, nested layout."""
    out = validate_multinode_topology(
        [4, 4, 4, 4], MultinodeUserParallelism(dp=2, pp=2)
    )
    assert out.tp == 4
    assert out.shape == "nested"
    assert out.nnodes_within_dp == 2


def test_validate_rejects_dp_not_dividing_total_gpus():
    """Cluster=[8, 8] + --dp 3 → 16 % 3 != 0 → reject before further checks."""
    with pytest.raises(ValueError, match="cannot infer"):
        validate_multinode_topology([8, 8], MultinodeUserParallelism(dp=3))


def test_validate_explicit_tp_overrides_dp_inference():
    """Explicit tp wins; dp is only consistency-checked, not used to infer tp.

    With tp=8 the inferred dp_total is 2; user.dp=2 must agree.
    """
    out = validate_multinode_topology([8, 8], MultinodeUserParallelism(tp=8, dp=2))
    assert out.tp == 8
    assert out.dp == 2


def test_validate_explicit_tp_with_inconsistent_dp_still_rejects():
    """Regression: tp explicit + dp explicit but inconsistent → reject by
    existing dp-mismatch check (not by the new inference branch)."""
    with pytest.raises(ValueError, match="does not match"):
        validate_multinode_topology([8, 8], MultinodeUserParallelism(tp=8, dp=5))


# ---------------------------------------------------------------------------
# selector ``_create_candidate`` topology gating
# ---------------------------------------------------------------------------


def _mock_worker(name, gpu_count, total_memory=80 * 1024**3):
    """Build a worker mock just rich enough for _create_candidate."""
    worker = MagicMock()
    worker.id = hash(name) & 0xFFFF
    worker.name = name
    worker.ip = "10.0.0.1"
    worker.ifname = "eth0"
    gpus = []
    for i in range(gpu_count):
        gpu = MagicMock()
        gpu.index = i
        gpu.memory.total = total_memory
        gpu.type = "nvidia"
        gpus.append(gpu)
    worker.status.gpu_devices = gpus
    return worker


def _mock_model(backend_parameters=None, backend_version="0.18.0"):
    model = MagicMock()
    model.backend = "vLLM"
    model.backend_parameters = backend_parameters
    model.backend_version = backend_version
    model.name = "test-model"
    model.categories = None
    # _create_candidate → get_computed_ram_claim consults extended_kv_cache;
    # leaving it as a MagicMock would trip ``ram_size > 0``.
    model.extended_kv_cache = None
    return model


def test_create_candidate_returns_none_on_topology_failure():
    """Heterogeneous cluster requested for cross-node TP/PP → selector should
    skip the worker combination by returning None plus an explanatory reason."""
    from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
        _create_candidate,
    )

    model = _mock_model(
        ["--distributed-executor-backend", "mp", "--tp", "4", "--pp", "4"]
    )
    workers = [_mock_worker("a", 8), _mock_worker("b", 4)]
    candidate, reason = _create_candidate(model, workers)
    assert candidate is None
    assert reason is not None
    assert "homogeneous" in reason


def test_create_candidate_tp_unfittable_returns_reason():
    """Cluster=[1,1] + tp=2 → vLLM forbids cross-node TP; reason surfaces it.

    Regression: this is the user-reported scenario where the UI used to show
    "no suitable workers" without explaining the parameter is impossible.
    """
    from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
        _create_candidate,
    )

    model = _mock_model(["--distributed-executor-backend", "mp", "--tp", "2"])
    workers = [_mock_worker("a", 1), _mock_worker("b", 1)]
    candidate, reason = _create_candidate(model, workers)
    assert candidate is None
    assert "cannot divide" in reason


def test_create_candidate_returns_candidate_when_topology_fits():
    """Homogeneous cluster with valid cross-node TP/PP → candidate created."""
    from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
        _create_candidate,
    )

    model = _mock_model(
        ["--distributed-executor-backend", "mp", "--tp", "8", "--pp", "2"]
    )
    workers = [_mock_worker("a", 8), _mock_worker("b", 8)]
    candidate, reason = _create_candidate(model, workers)
    assert candidate is not None
    assert reason is None
    assert len(candidate.subordinate_workers) == 1


def test_create_candidate_skips_topology_check_for_ray_backend():
    """Ray-path candidates don't go through MP topology validation."""
    from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
        _create_candidate,
    )

    # Heterogeneous would fail mp topology, but Ray path bypasses the check.
    model = _mock_model(
        ["--distributed-executor-backend", "ray", "--tp", "4", "--pp", "4"]
    )
    workers = [_mock_worker("a", 8), _mock_worker("b", 4)]
    candidate, reason = _create_candidate(model, workers)
    assert candidate is not None
    assert reason is None


def test_create_candidate_skips_topology_check_for_single_worker():
    """Single-worker schedule is not multi-node — no topology validation."""
    from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
        _create_candidate,
    )

    model = _mock_model(["--distributed-executor-backend", "mp"])
    candidate, reason = _create_candidate(model, [_mock_worker("a", 8)])
    assert candidate is not None
    assert reason is None
