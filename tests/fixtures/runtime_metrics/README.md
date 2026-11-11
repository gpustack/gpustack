# Runtime metrics fixtures

`sglang_v0_5_23_timing.prom` uses metric names, types and label dimensions from
[SGLang v0.5.23](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/observability/metrics_collector.py).
Its values are synthetic representative observations, not a production scrape.
TPOT carries `is_streaming`; the ITL histogram does not. The payload contains
only exposition-compatible metadata so it can also be parsed as OpenMetrics
after adding the `# EOF` trailer.

`audited_mappings.json` fixes the independently audited mapping expectations,
native label shapes, source URLs, units and synthetic sample values. The five
`*_mappings.prom` payloads cover vLLM v0.33.0rc3, SGLang v0.5.23, SGLang main
commit `1be97a0a6312ef5bf1e613bb50beb0081ec8ea31`, and verified legacy shapes.
They include optional and component-specific families without claiming those
features coexist in every deployment. Tests read these fixed artifacts rather
than deriving expected names or values from the metrics configuration under test.
