from gpustack.routes.benchmarks import order_benchmark_export_fields


def test_order_benchmark_export_fields_puts_identifying_fields_first():
    benchmark = {
        "description": "benchmark description",
        "dataset_output_tokens": 256,
        "snapshot": {},
        "name": "benchmark-a",
        "request_rate": 10,
        "model_name": "model-a",
        "profile": "Custom",
        "dataset_name": "Random",
        "model_instance_name": "model-a-1",
        "total_requests": 100,
        "dataset_input_tokens": 128,
        "dataset_seed": 42,
    }

    ordered = order_benchmark_export_fields(benchmark)

    assert list(ordered) == [
        "name",
        "model_name",
        "model_instance_name",
        "profile",
        "dataset_name",
        "request_rate",
        "total_requests",
        "dataset_input_tokens",
        "dataset_output_tokens",
        "dataset_seed",
        "description",
        "snapshot",
    ]
