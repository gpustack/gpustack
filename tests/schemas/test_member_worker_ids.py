"""Where a member actually is, asked once instead of four times.

Four readers each asked "which machine is this member on" by reading
`worker_id`, and that field is only the machine the row is *filed under*. An
instance that spans machines records the rest on
`distributed_servers.subordinate_workers`. So a member on three machines was
invisible on two of them to the gather floor, to the breach report meant to
catch what the floor lets through, to the pairing-locality sum, and to the
proximity scorer.

Reachable without any cross-node support in the group solver: a scaled-out
member goes down the per-instance path with the whole worker list, and
`distributed_inference_across_workers` defaults to true for vLLM, SGLang and
MindIE.
"""

from types import SimpleNamespace

from gpustack.schemas.models import member_worker_ids


def _subordinate(worker_id):
    return SimpleNamespace(worker_id=worker_id)


def test_a_single_machine_member_is_the_machine_it_is_filed_under():
    """The compatibility case, and it is every member that exists today: one
    element, the same value every caller was already reading."""
    instance = SimpleNamespace(worker_id=7, distributed_servers=None)

    assert member_worker_ids(instance) == [7]


def test_a_spanning_member_reports_every_machine():
    instance = SimpleNamespace(
        worker_id=1,
        distributed_servers=SimpleNamespace(
            subordinate_workers=[_subordinate(2), _subordinate(3)]
        ),
    )

    assert member_worker_ids(instance) == [1, 2, 3]


def test_the_primary_comes_first():
    """Rank 0 lives on the machine the row is filed under, and a caller that
    cares which machine holds it must not have to guess."""
    instance = SimpleNamespace(
        worker_id=9,
        distributed_servers=SimpleNamespace(subordinate_workers=[_subordinate(2)]),
    )

    assert member_worker_ids(instance)[0] == 9


def test_an_unplaced_member_is_nowhere():
    """Not `[None]`: a caller building a set of machines would then hold a
    None and compare it against real ids."""
    instance = SimpleNamespace(worker_id=None, distributed_servers=None)

    assert member_worker_ids(instance) == []


def test_a_subordinate_repeating_the_primary_is_counted_once():
    """Whatever produced it, a machine appearing twice would double a member's
    weight in the locality sum and in every count derived from it."""
    instance = SimpleNamespace(
        worker_id=1,
        distributed_servers=SimpleNamespace(
            subordinate_workers=[_subordinate(1), _subordinate(2)]
        ),
    )

    assert member_worker_ids(instance) == [1, 2]


def test_a_subordinate_without_a_machine_is_skipped():
    instance = SimpleNamespace(
        worker_id=1,
        distributed_servers=SimpleNamespace(
            subordinate_workers=[_subordinate(None), _subordinate(2)]
        ),
    )

    assert member_worker_ids(instance) == [1, 2]


def test_an_instance_with_no_distributed_block_at_all():
    """Rows predating the field, and every role-less deployment."""
    assert member_worker_ids(SimpleNamespace(worker_id=4)) == [4]
