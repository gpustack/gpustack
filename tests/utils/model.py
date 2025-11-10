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
    categories=[CategoryEnum.LLM],
    cpu_offloading=True,
    distributed_inference_across_workers=True,
    **kargs,
) -> Model:
    if huggingface_repo_id is not None:
        source = SourceEnum.HUGGING_FACE
    if model_scope_model_id is not None:
        source = SourceEnum.MODEL_SCOPE

    return Model(
        id=id,
        name=name,
        replicas=replicas,
        ready_replicas=0,
        source=source,
        huggingface_repo_id=huggingface_repo_id,
        model_scope_model_id=model_scope_model_id,
        distributable=distributable,
        placement_strategy=placement_strategy,
        cpu_offloading=cpu_offloading,
        distributed_inference_across_workers=distributed_inference_across_workers,
        categories=categories,
        extended_kv_cache=extended_kv_cache,
        **kargs,
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
