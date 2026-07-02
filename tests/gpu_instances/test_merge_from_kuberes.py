"""Unit tests for ``GPUInstance.merge_from_kuberes()``.

This is the inverse of ``convert_to_kuberes``: it maps a downstream worker CR
dict into a :class:`GPUInstanceStatus`. It is a **pure merge** — no concurrency
guards (DELETING sticky / require_current_phase / count backoff / refresh) and
no mutation of ``self``; those live in the controller's ``_set_status``.
"""

from gpustack.schemas.gpu_instances import (
    GPUInstance,
    GPUInstanceSpec,
    GPUInstanceStatus,
)


def _gi(status=None) -> GPUInstance:
    return GPUInstance(
        id=1,
        name="gi-1",
        owner_principal_id=1,
        cluster_id=2,
        spec=GPUInstanceSpec(type_="gpu", image="busybox"),
        status=status,
    )


def test_merge_from_kuberes_maps_downstream_status_fields():
    inst = _gi()
    downstream = {
        "status": {
            "phase": "Ready",
            "phaseMessage": "all good",
            "accessAddresses": ["1.2.3.4:8080"],
            "ports": [{"port": 8080}],
        }
    }

    status = inst.merge_from_kuberes(downstream)

    assert isinstance(status, GPUInstanceStatus)
    assert status.phase == "Ready"
    assert status.phase_message == "all good"
    assert status.access_addresses == ["1.2.3.4:8080"]
    assert status.ports and status.ports[0].port == 8080


def test_merge_from_kuberes_takes_namespace_from_metadata():
    # The k8s namespace is sourced from the CR's metadata (authoritative) and
    # overrides any namespace echoed in the status body.
    inst = _gi()

    status = inst.merge_from_kuberes(
        {
            "metadata": {"namespace": "gpustack-user-9"},
            "status": {"phase": "Ready", "namespace": "stale"},
        }
    )

    assert status.namespace == "gpustack-user-9"


def test_merge_from_kuberes_without_metadata_leaves_namespace_default():
    # No metadata.namespace and none in the status -> default empty string; the
    # method never invents one.
    inst = _gi()

    status = inst.merge_from_kuberes({"status": {"phase": "Ready"}})

    assert status.namespace == ""


def test_merge_from_kuberes_empty_or_missing_status():
    inst = _gi()

    assert inst.merge_from_kuberes({}).phase is None
    assert inst.merge_from_kuberes({"status": None}).phase is None
    assert inst.merge_from_kuberes({"status": {}}).count == 0


def test_merge_from_kuberes_is_pure_no_guards_no_mutation():
    # A DELETING row must NOT be protected here (that guard lives in the
    # controller); merge returns exactly what the downstream reports, applies no
    # count backoff, and never mutates self.status.
    inst = _gi(status=GPUInstanceStatus(phase="Deleting", count=3))

    status = inst.merge_from_kuberes({"status": {"phase": "Ready"}})

    assert status.phase == "Ready"  # no DELETING sticky
    assert status.count == 0  # no count backoff
    assert inst.status.phase == "Deleting"  # self untouched
    assert inst.status.count == 3
