from gpustack.server.metrics_collector import parse_token_metrics, ModelUsageMetrics

# Example Prometheus metrics text
METRICS_TEXT = '''
route_upstream_model_consumer_metric_input_token{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 156
route_upstream_model_consumer_metric_llm_duration_count{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 13
route_upstream_model_consumer_metric_output_token{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 1755
route_upstream_model_consumer_metric_total_token{ai_route="ai-route-model-1",ai_cluster="outbound|80||model-1-1.static",ai_model="qwen3-0.6b",ai_consumer="d720eeb5b57fbe94.gpustack-2"} 1911
'''


def test_parse_token_metrics_basic():
    result = parse_token_metrics(METRICS_TEXT)
    assert len(result) == 1
    metrics = list(result.values())[0]
    assert isinstance(metrics, ModelUsageMetrics)
    assert metrics.model == "qwen3-0.6b"
    assert metrics.input_token == 156
    assert metrics.output_token == 1755
    assert metrics.total_token == 1911
    assert metrics.request_count == 13
    assert metrics.user_id == 2
    assert metrics.access_key == "d720eeb5b57fbe94"


def test_parse_token_metrics_multiple_users():
    metrics_text = (
        METRICS_TEXT
        + '''
route_upstream_model_consumer_metric_input_token{ai_route="ai-route-model-2",ai_cluster="outbound|80||model-2-1.static",ai_model="qwen3-0.7b",ai_consumer="abcdef.gpustack-3"} 200
route_upstream_model_consumer_metric_output_token{ai_route="ai-route-model-2",ai_cluster="outbound|80||model-2-1.static",ai_model="qwen3-0.7b",ai_consumer="abcdef.gpustack-3"} 3000
route_upstream_model_consumer_metric_total_token{ai_route="ai-route-model-2",ai_cluster="outbound|80||model-2-1.static",ai_model="qwen3-0.7b",ai_consumer="abcdef.gpustack-3"} 3200
route_upstream_model_consumer_metric_llm_duration_count{ai_route="ai-route-model-2",ai_cluster="outbound|80||model-2-1.static",ai_model="qwen3-0.7b",ai_consumer="abcdef.gpustack-3"} 21

route_upstream_model_consumer_metric_input_token{ai_route="ai-route-model-3",ai_cluster="outbound|80||model-3-1.static",ai_model="qwen3-0.8b",ai_consumer="xyz123.gpustack-4"} 500
route_upstream_model_consumer_metric_output_token{ai_route="ai-route-model-3",ai_cluster="outbound|80||model-3-1.static",ai_model="qwen3-0.8b",ai_consumer="xyz123.gpustack-4"} 800
route_upstream_model_consumer_metric_total_token{ai_route="ai-route-model-3",ai_cluster="outbound|80||model-3-1.static",ai_model="qwen3-0.8b",ai_consumer="xyz123.gpustack-4"} 1300
route_upstream_model_consumer_metric_llm_duration_count{ai_route="ai-route-model-3",ai_cluster="outbound|80||model-3-1.static",ai_model="qwen3-0.8b",ai_consumer="xyz123.gpustack-4"} 7
'''
    )
    result = parse_token_metrics(metrics_text)
    assert len(result) == 3
    # Sort by model for assertion
    result_sorted = sorted(result.values(), key=lambda m: m.model)
    # Check first user
    m1 = result_sorted[0]
    assert m1.model == "qwen3-0.6b"
    assert m1.input_token == 156
    assert m1.output_token == 1755
    assert m1.total_token == 1911
    assert m1.request_count == 13
    assert m1.user_id == 2
    assert m1.access_key == "d720eeb5b57fbe94"
    # Check second user
    m2 = result_sorted[1]
    assert m2.model == "qwen3-0.7b"
    assert m2.input_token == 200
    assert m2.output_token == 3000
    assert m2.total_token == 3200
    assert m2.request_count == 21
    assert m2.user_id == 3
    assert m2.access_key == "abcdef"
    # Check third user
    m3 = result_sorted[2]
    assert m3.model == "qwen3-0.8b"
    assert m3.input_token == 500
    assert m3.output_token == 800
    assert m3.total_token == 1300
    assert m3.request_count == 7
    assert m3.user_id == 4
    assert m3.access_key == "xyz123"
