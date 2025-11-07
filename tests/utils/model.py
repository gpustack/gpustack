from gpustack.schemas.models import (
    CategoryEnum,
    GPUSelector,
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
    huggingface_repo_id=None,
    model_scope_model_id=None,
    placement_strategy=PlacementStrategyEnum.BINPACK,
    distributable=True,
    extended_kv_cache=None,
    **kargs,
) -> Model:
    worker_selector = kargs.get("worker_selector")
    cpu_offloading = kargs.get("cpu_offloading", True)
    distributed_inference_across_workers = kargs.get(
        "distributed_inference_across_workers", True
    )
    gpu_selector = kargs.get("gpu_selector", None)
    backend_parameters = kargs.get("backend_parameters")
    categories = kargs.get("categories", [CategoryEnum.LLM])

    huggingface_filename = None
    if huggingface_repo_id is not None:
        source = SourceEnum.HUGGING_FACE
        huggingface_filename = kargs.get("huggingface_filename")
    if model_scope_model_id is not None:
        source = SourceEnum.MODEL_SCOPE

    return Model(
        id=id,
        name=name,
        replicas=replicas,
        ready_replicas=0,
        source=source,
        huggingface_repo_id=huggingface_repo_id,
        huggingface_filename=huggingface_filename,
        model_scope_model_id=model_scope_model_id,
        distributable=distributable,
        placement_strategy=placement_strategy,
        worker_selector=worker_selector,
        cpu_offloading=cpu_offloading,
        distributed_inference_across_workers=distributed_inference_across_workers,
        gpu_selector=gpu_selector,
        backend_parameters=backend_parameters,
        categories=categories,
        extended_kv_cache=extended_kv_cache,
    )


def make_model(
    gpus_per_replica=2, gpu_ids=None, repo_id="Qwen/Qwen2.5-7B-Instruct", **kwargs
):
    gpu_selector = None
    if gpu_ids is not None:
        gpu_selector = GPUSelector(
            gpu_ids=gpu_ids,
            gpus_per_replica=gpus_per_replica,
        )

    return new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id=repo_id,
        gpu_selector=gpu_selector,
        **kwargs,
    )
