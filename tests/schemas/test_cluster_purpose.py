"""The cluster-purpose predicate: one home for "is this a GPU Service cluster".

A cluster carries no purpose column. The presence or absence of
``k8s_options.gpu_instance_options`` *is* the signal — present means GPU
Service, absent means Model Service — and that one bit is read from three
different shapes depending on where the cluster came from:

* a ``K8sOptions`` model, as constructed by the route layer;
* a snake-cased dict, as produced by ``model_dump`` and stored in the JSON
  column (and replayed over the event bus without re-validation);
* a camel-cased dict, as submitted by the API / UI.

These lock all three to the same answer. A disagreement between them would
split the product in half: a cluster the deploy picker calls Model Service
while the GPU Service pages call it GPU Service.
"""

import pytest

from gpustack.schemas.clusters import (
    Cluster,
    GpuInstanceOptions,
    K8sOptions,
    is_gpu_service_cluster,
    is_gpu_service_k8s_options,
)

# The same cluster expressed the three ways it can reach the predicate. All-defaults
# ``GpuInstanceOptions()`` is deliberate: it is what a GPU Service cluster with no knobs
# set persists as, and it must still read as GPU Service.
GPU_SERVICE_FORMS = {
    "model": K8sOptions(gpu_instance_options=GpuInstanceOptions()),
    "snake_dict": {"gpu_instance_options": {}},
    "camel_dict": {"gpuInstanceOptions": {}},
}

MODEL_SERVICE_FORMS = {
    "model": K8sOptions(),
    "snake_dict": {"gpu_instance_options": None},
    "camel_dict": {"gpuInstanceOptions": None},
}


def test_model_form_reads_the_purpose():
    """The K8sOptions branch: a set ``gpu_instance_options`` is the whole signal."""
    assert is_gpu_service_k8s_options(
        K8sOptions(gpu_instance_options=GpuInstanceOptions())
    )
    assert not is_gpu_service_k8s_options(K8sOptions())


@pytest.mark.parametrize("form", GPU_SERVICE_FORMS.values(), ids=GPU_SERVICE_FORMS)
def test_every_form_of_a_gpu_service_cluster_agrees(form):
    assert is_gpu_service_k8s_options(form) is True


@pytest.mark.parametrize("form", MODEL_SERVICE_FORMS.values(), ids=MODEL_SERVICE_FORMS)
def test_every_form_of_a_model_service_cluster_agrees(form):
    assert is_gpu_service_k8s_options(form) is False


def test_snake_and_camel_dicts_are_both_honoured():
    """A populated payload, not just an empty marker, under either key spelling.

    ``model_dump`` writes snake, the API/UI submits camel, and the JSON column
    replays whichever was written — so both spellings have to be live.
    """
    options = {"gpuInstancesAccessStaticAddress": "10.0.0.1"}
    assert is_gpu_service_k8s_options({"gpu_instance_options": options})
    assert is_gpu_service_k8s_options({"gpuInstanceOptions": options})


def test_absent_options_is_model_service():
    assert not is_gpu_service_k8s_options({})
    assert not is_gpu_service_k8s_options({"namespace": "gpustack-system"})


def test_none_is_model_service():
    """An unset ``k8s_options`` column is a Model Service cluster, not an error."""
    assert not is_gpu_service_k8s_options(None)


@pytest.mark.parametrize("value", ["gpuInstanceOptions", 42, [], object()])
def test_a_shape_we_never_write_is_model_service(value):
    """Fail closed: a shape the predicate cannot read is excluded from GPU Service.

    Nothing writes these — the column is only ever written from a validated
    model — so the branch exists to keep an unreadable row out of the GPU
    Service surfaces rather than to raise on a list/watch tick.
    """
    assert is_gpu_service_k8s_options(value) is False


@pytest.mark.parametrize("form", GPU_SERVICE_FORMS.values(), ids=GPU_SERVICE_FORMS)
def test_cluster_delegates_to_its_k8s_options(form):
    """Real ``Cluster`` rows, not mocks: SQLModel keeps a dict assignment as a raw
    dict, which is exactly the shape the event bus and the JSON column hand back.
    """
    assert is_gpu_service_cluster(Cluster(name="gpu-service", k8s_options=form))


def test_a_cluster_without_options_is_model_service():
    assert not is_gpu_service_cluster(Cluster(name="model-service", k8s_options=None))
    assert not is_gpu_service_cluster(
        Cluster(name="model-service", k8s_options=K8sOptions())
    )
