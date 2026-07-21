"""Golden tests for ``GPUInstance.convert_to_kuberes()``.

The method folds the former ``cluster_apis_util.spec_instance`` transforms into
the schema and now returns the *full* worker CR body (``metadata`` + ``spec``),
with the ``gpustack.ai/instance-id`` label on ``metadata.labels``. These lock:
the spec portion stays byte-for-byte the old ``spec_instance`` output, and the
metadata envelope (name + id label) is added.
"""

from gpustack.schemas.gpu_instance_persistent_volumes import (
    GPUInstancePersistentVolumeSpec,
)
from gpustack.schemas.gpu_instances import (
    KUBERES_INSTANCE_ID_LABEL,
    GPUInstance,
    GPUInstancePort,
    GPUInstanceResources,
    GPUInstanceSSHPublicKeyReference,
    GPUInstanceSpec,
    GPUInstanceVolume,
    GPUInstancePersistentVolumeTemplate,
)


def _gi(*, spec: GPUInstanceSpec, **kwargs) -> GPUInstance:
    # ``spec`` must be a GPUInstanceSpec instance: SQLModel table models skip
    # init validation (in production the column deserializes to GPUInstanceSpec).
    return GPUInstance(
        id=42,
        name="gi-x",
        owner_principal_id=1,
        cluster_id=2,
        spec=spec,
        **kwargs,
    )


def test_convert_to_kuberes_full_body_shape():
    spec = GPUInstanceSpec(
        type_="gpu",
        image="busybox",
        ports=[GPUInstancePort(port=8080)],
        resources=GPUInstanceResources(accelerator="1"),
        volume=GPUInstanceVolume(
            persistent_template=GPUInstancePersistentVolumeTemplate(
                name="pv-1",
                spec=GPUInstancePersistentVolumeSpec(type_="nfs"),
                release_with_instance=True,
            )
        ),
        ssh_public_keys=[GPUInstanceSSHPublicKeyReference(name="k1")],
    )
    inst = _gi(spec=spec, display_name="My GI", description="a desc")

    body = inst.convert_to_kuberes()

    # metadata envelope: name + id label, and nothing the ops layer must add
    # (apiVersion / kind / namespace stay out of the schema).
    assert body["metadata"] == {
        "name": "gi-x",
        "labels": {KUBERES_INSTANCE_ID_LABEL: "42"},
    }
    assert "apiVersion" not in body
    assert "kind" not in body
    assert "namespace" not in body["metadata"]

    # spec portion is byte-for-byte the former spec_instance() output: start
    # from the same dump and apply the same transforms independently.
    expected_spec = spec.model_dump(by_alias=True, exclude_none=True)
    expected_spec["displayName"] = "My GI"
    expected_spec["description"] = "a desc"
    del expected_spec["volume"]["persistentTemplate"]
    expected_spec["volume"]["persistent"] = {"name": "pv-1"}
    del expected_spec["sshPublicKeys"]
    expected_spec["sshPublicKey"] = {"name": "gi-x"}
    assert body["spec"] == expected_spec

    # Spot-check the individual transforms the golden compare depends on.
    assert body["spec"]["displayName"] == "My GI"
    assert body["spec"]["description"] == "a desc"
    assert body["spec"]["volume"] == {"persistent": {"name": "pv-1"}}
    assert "sshPublicKeys" not in body["spec"]
    assert body["spec"]["sshPublicKey"] == {"name": "gi-x"}


def test_convert_to_kuberes_minimal_instance():
    # No display_name/description, no volume, no ssh keys.
    spec = GPUInstanceSpec(type_="gpu", image="busybox")
    inst = _gi(spec=spec)

    body = inst.convert_to_kuberes()

    assert body["metadata"] == {
        "name": "gi-x",
        "labels": {KUBERES_INSTANCE_ID_LABEL: "42"},
    }
    # Hoisted fields are absent when the row has none.
    assert "displayName" not in body["spec"]
    assert "description" not in body["spec"]
    assert "volume" not in body["spec"]
    # sshPublicKey is always set to the instance-named reference, even with no
    # user-supplied keys (matches the former spec_instance behavior).
    assert body["spec"]["sshPublicKey"] == {"name": "gi-x"}
    assert "sshPublicKeys" not in body["spec"]


def test_kuberes_instance_id_label_key():
    assert KUBERES_INSTANCE_ID_LABEL == "gpustack.ai/instance-id"
