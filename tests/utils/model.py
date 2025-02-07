from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    PlacementStrategyEnum,
    SourceEnum,
)


def new_model_instance(
    id,
    name,
    model_id,
    worker_id=None,
    state=ModelInstanceStateEnum.PENDING,
    gpu_indexes=None,
    computed_resource_claim=None,
) -> ModelInstance:
    return ModelInstance(
        id=id,
        name=name,
        worker_id=worker_id,
        model_id=model_id,
        model_name="test",
        state=state,
        gpu_indexes=gpu_indexes,
        computed_resource_claim=computed_resource_claim,
    )


def new_model(
    id,
    name,
    replicas=1,
    ollama_library_model_name=None,
    huggingface_repo_id=None,
    placement_strategy=PlacementStrategyEnum.BINPACK,
    distributable=True,
    **kargs,
) -> Model:
    worker_selector = kargs.get("worker_selector")
    cpu_offloading = kargs.get("cpu_offloading", True)
    distributed_inference_across_workers = kargs.get(
        "distributed_inference_across_workers", True
    )
    gpu_selector = kargs.get("gpu_selector", None)

    source = SourceEnum.OLLAMA_LIBRARY
    huggingface_filename = None
    if huggingface_repo_id is not None:
        source = SourceEnum.HUGGING_FACE
        huggingface_filename = kargs.get("huggingface_filename")

    return Model(
        id=id,
        name=name,
        replicas=replicas,
        ready_replicas=0,
        source=source,
        ollama_library_model_name=ollama_library_model_name,
        huggingface_repo_id=huggingface_repo_id,
        huggingface_filename=huggingface_filename,
        distributable=distributable,
        placement_strategy=placement_strategy,
        worker_selector=worker_selector,
        cpu_offloading=cpu_offloading,
        distributed_inference_across_workers=distributed_inference_across_workers,
        gpu_selector=gpu_selector,
    )
