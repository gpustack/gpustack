"""Unit tests for GPUInstance phase predicates and the canonical failed set.

These lock the single source of truth for phase semantics: the ``is_*``
predicates on ``GPUInstance`` and the ``FAILED_PHASES`` gating set.
"""

import pytest

from gpustack.schemas.gpu_instances import (
    FAILED_PHASES,
    INTERRUPTED_PHASES,
    TRANSITIONING_PHASES,
    GPUInstance,
    GPUInstancePhase,
    GPUInstancePort,
    GPUInstanceSpec,
    GPUInstanceStatus,
)


def _gi(phase) -> GPUInstance:
    # ``spec`` must be a GPUInstanceSpec instance (not a raw dict): SQLModel
    # table models skip init validation, and ``is_ready()`` reads ``spec.ports``.
    status = None if phase is _NO_STATUS else GPUInstanceStatus(phase=phase)
    return GPUInstance(
        id=1,
        name="gi-1",
        owner_principal_id=1,
        cluster_id=2,
        spec=GPUInstanceSpec(type_="gpu", image="busybox"),
        status=status,
    )


_NO_STATUS = object()


def test_is_creating_when_phase_none_or_status_missing():
    assert _gi(None).is_creating() is True
    assert _gi(_NO_STATUS).is_creating() is True  # status is None entirely
    assert _gi(GPUInstancePhase.READY).is_creating() is False


@pytest.mark.parametrize(
    "predicate, phase",
    [
        ("is_starting", GPUInstancePhase.STARTING),
        ("is_stopping", GPUInstancePhase.STOPPING),
        ("is_stopped", GPUInstancePhase.STOPPED),
        ("is_deleting", GPUInstancePhase.DELETING),
    ],
)
def test_exact_phase_predicates(predicate, phase):
    assert getattr(_gi(phase), predicate)() is True
    # A different phase must not match.
    assert getattr(_gi(GPUInstancePhase.NOT_READY), predicate)() is False


def test_is_failed_matches_every_known_failure_phase():
    for phase in FAILED_PHASES:
        assert _gi(phase).is_failed() is True


def test_is_failed_is_broad_and_catches_downstream_phases_outside_enum():
    # The worker CR may report failure phases GPUStack does not define; the
    # broad endswith("Failed") check must still treat them as failed, even
    # though they are NOT members of FAILED_PHASES.
    downstream = "SchedulingFailed"
    assert downstream not in FAILED_PHASES
    assert _gi(downstream).is_failed() is True


def test_is_failed_false_for_non_failure_phases():
    assert _gi(GPUInstancePhase.READY).is_failed() is False
    assert _gi(GPUInstancePhase.STOPPED).is_failed() is False
    assert _gi(None).is_failed() is False


def _gi_full(phase, *, spec: GPUInstanceSpec, status_extra=None) -> GPUInstance:
    # ``spec`` must be a GPUInstanceSpec instance: SQLModel table models skip
    # init validation, so a raw dict would not be coerced (in production the
    # column deserializes to GPUInstanceSpec).
    status_kwargs = {"phase": phase}
    status_kwargs.update(status_extra or {})
    return GPUInstance(
        id=1,
        name="gi-1",
        owner_principal_id=1,
        cluster_id=2,
        spec=spec,
        status=GPUInstanceStatus(**status_kwargs),
    )


def test_is_ready_true_when_ready_and_no_extra_fields_required():
    # No exposed ports and no accelerator -> readiness needs no extra fields.
    spec = GPUInstanceSpec(type_="gpu", image="busybox")
    assert _gi_full(GPUInstancePhase.READY, spec=spec).is_ready() is True


def test_is_ready_false_when_not_ready():
    spec = GPUInstanceSpec(type_="gpu", image="busybox")
    assert _gi_full(GPUInstancePhase.NOT_READY, spec=spec).is_ready() is False
    assert _gi_full(GPUInstancePhase.STARTING, spec=spec).is_ready() is False


def test_is_ready_requires_populated_fields_for_exposed_ports():
    spec = GPUInstanceSpec(
        type_="gpu", image="busybox", ports=[GPUInstancePort(port=8080)]
    )
    # Phase Ready but access_addresses / ports not yet reported -> not ready:
    # is_ready() now folds in the full-population check.
    incomplete = _gi_full(GPUInstancePhase.READY, spec=spec)
    assert incomplete.is_ready() is False
    # Ready and fully populated -> ready.
    complete = _gi_full(
        GPUInstancePhase.READY,
        spec=spec,
        status_extra={
            "access_addresses": ["1.2.3.4:8080"],
            "ports": [{"port": 8080}],
        },
    )
    assert complete.is_ready() is True


def test_failed_phases_are_the_five_gpustack_defined_ones():
    assert FAILED_PHASES == frozenset(
        {
            GPUInstancePhase.CREATE_FAILED,
            GPUInstancePhase.SSH_KEY_CREATE_FAILED,
            GPUInstancePhase.PV_TYPE_CREATE_FAILED,
            GPUInstancePhase.PV_CREATE_FAILED,
            GPUInstancePhase.INITIALIZE_FAILED,
        }
    )
    # Every known failure phase ends with "Failed", so is_failed() is a
    # strict superset of FAILED_PHASES membership.
    assert all(p.endswith("Failed") for p in FAILED_PHASES)


def test_phase_category_sets_are_disjoint():
    assert TRANSITIONING_PHASES == frozenset(
        {
            GPUInstancePhase.DELETING,
            GPUInstancePhase.STOPPING,
            GPUInstancePhase.STARTING,
            GPUInstancePhase.NOT_READY,
        }
    )
    # Stopped is the interrupted-but-resumable phase, kept out of the
    # transitioning set so /start can gate on it alone.
    assert INTERRUPTED_PHASES == frozenset({GPUInstancePhase.STOPPED})
    # The three lifecycle-gating categories never overlap.
    assert not (TRANSITIONING_PHASES & INTERRUPTED_PHASES)
    assert not (TRANSITIONING_PHASES & FAILED_PHASES)
    assert not (INTERRUPTED_PHASES & FAILED_PHASES)
