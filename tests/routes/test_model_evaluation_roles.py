"""The shape the deployment form sends to /model-evaluations.

A role held in FORM shape — override switches, GPU ids still in the cascader's
[worker, gpu] pairs — is rejected outright, and a 422 surfaces as an error
toast over the whole form. The form therefore runs the same transform submit
runs, and what these assert is that the transform's OUTPUT parses: a role that
inherits says so with nulls, and a role that overrides sends the flat wire
shape.
"""

from gpustack.schemas.model_evaluations import ModelEvaluationRequest


def _request(roles):
    return ModelEvaluationRequest.model_validate(
        {
            "cluster_id": 1,
            "model_specs": [
                {
                    "source": "huggingface",
                    "huggingface_repo_id": "x/y",
                    "backend": "vLLM",
                    "replicas": 1,
                    "roles": roles,
                    "disaggregation": {"mode": "vllm-nixl"},
                }
            ],
        }
    )


def test_an_inheriting_role_parses_with_its_nulls():
    """`null` is the wire's word for inherit, and the transform writes one into
    every field of a group left on "same as model". They have to survive
    parsing as None rather than be refused as the wrong type."""
    request = _request(
        [
            {
                "name": "prefill",
                "replicas": 2,
                "backend": None,
                "backend_version": None,
                "image_name": None,
                "run_command": None,
                "backend_parameters": None,
                "env": None,
                "gpu_selector": None,
                "worker_selector": None,
                "gpu_type_selector": None,
                "extended_kv_cache": None,
                "speculative_config": None,
            },
            {"name": "decode", "replicas": 4},
            {"name": "router", "replicas": 1, "resources": None},
        ]
    )

    roles = request.model_specs[0].roles
    assert [(r.name, r.replicas) for r in roles] == [
        ("prefill", 2),
        ("decode", 4),
        ("router", 1),
    ]
    assert roles[0].backend_parameters is None


def test_an_overriding_role_parses_its_flat_gpu_ids_and_parameters():
    """What the form's cascader holds as [worker, gpu] pairs, `generateGPUIds`
    flattens before submit. The API's `List[str]` is what rejected the form
    shape, so the flat one is the half worth pinning."""
    request = _request(
        [
            {
                "name": "decode",
                "replicas": 2,
                "backend_parameters": ["--tensor-parallel-size=4"],
                "env": {"HCCL_BUFFSIZE": "1024"},
                "gpu_selector": {
                    "gpu_ids": ["worker-a:npu:0", "worker-a:npu:1"],
                    "gpus_per_replica": 2,
                },
            },
        ]
    )

    decode = request.model_specs[0].roles[0]
    assert decode.gpu_selector.gpu_ids == ["worker-a:npu:0", "worker-a:npu:1"]
    assert decode.gpu_selector.gpus_per_replica == 2
    assert decode.backend_parameters == ["--tensor-parallel-size=4"]
    assert decode.env == {"HCCL_BUFFSIZE": "1024"}


def test_a_router_carries_its_declared_resources():
    request = _request(
        [{"name": "router", "replicas": 1, "resources": {"memory": 8589934592}}]
    )

    assert request.model_specs[0].roles[0].resources.memory == 8589934592


def test_a_plain_deployment_still_sends_no_roles():
    """The form nulls both fields when PD is off, and that has to stay the
    role-less shape rather than becoming a group with no members."""
    request = ModelEvaluationRequest.model_validate(
        {
            "cluster_id": 1,
            "model_specs": [
                {
                    "source": "huggingface",
                    "huggingface_repo_id": "x/y",
                    "roles": None,
                    "disaggregation": None,
                }
            ],
        }
    )

    assert request.model_specs[0].roles is None
    assert request.model_specs[0].disaggregation is None
