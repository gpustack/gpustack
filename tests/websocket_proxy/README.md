# WebSocket Proxy Tests

This directory contains tests for the WebSocket proxy functionality.

## Running Tests

### Run all tests:
```bash
pytest tests/ -v
```

### Run specific test class:
```bash
pytest tests/test_websocket_bench.py::TestProxyBasicFunctionality -v
pytest tests/test_websocket_bench.py::TestProxyThroughput -v
pytest tests/test_websocket_bench.py::TestProxyLatency -v
```

### Run specific test:
```bash
pytest tests/test_websocket_bench.py::TestProxyBasicFunctionality::test_single_get_request -v
```

## Test Coverage

### test_message.py
Tests the binary message protocol serialization/deserialization.

### test_server_federation.py
Tests server-to-server federation:
- Server connection via WebSocket handshake with header-based registration
- Client registration broadcast to peers
- Client disconnection broadcast to peers
- Peer removal
- Bidirectional peer connections

### test_websocket_bench.py
Tests the proxy over WebSocket tunnel functionality with pytest:

#### TestProxyBasicFunctionality
- `test_single_get_request`: Verifies basic GET request works
- `test_single_post_request`: Verifies POST request with payload works
- `test_concurrent_requests`: Tests concurrent request handling
- Performance targets: Average < 500ms

#### TestProxyThroughput
- `test_small_payload_throughput`: Tests with 512B payload
  - 100 requests, 10 concurrent
  - Target: >= 1.0 MB/s
- `test_large_payload_throughput`: Tests with 8KB payload
  - 50 requests, 5 concurrent
  - Target: >= 0.5 MB/s

#### TestProxyLatency
- `test_request_latency_distribution`: Measures latency distribution
  - 50 requests
  - Targets: Average < 100ms, P95 < 200ms

## Performance Expectations

Based on testing, here are reasonable performance targets:

| Metric | Target | Reason |
|--------|--------|--------|
| GET request (cold) | < 1s | Should complete quickly |
| GET request (warm) | < 500ms | After connection established |
| POST request (small) | < 500ms | 1KB payload |
| POST request (large) | < 1s | 8KB payload |
| Throughput (small) | >= 1 MB/s | 10 concurrent, 512B payload |
| Throughput (large) | >= 0.5 MB/s | 5 concurrent, 8KB payload |
| P50 latency | < 200ms | 95% of requests should be fast |

## Benchmark Script (Deprecated)

The standalone benchmark script at `benchmarks/websocket_bench.py` is deprecated.
For ad-hoc testing, you can still use it, but pytest-based tests are recommended.

```bash
# Old way (still works):
python -m benchmarks.websocket_bench --requests 100 --concurrency 10

# New recommended way:
pytest tests/test_websocket_bench.py -v
```

## Test Infrastructure

The tests use pytest fixtures for proper setup and teardown:

- `proxy_infrastructure`: Sets up complete proxy stack
  - WebSocket server (FastAPI + Uvicorn)
  - HTTP proxy server (HTTPSProxyServer)
  - WebSocket client (MessageClient with registered CIDRs)
  - Automatic cleanup after each test

- `test_http_server`: Simple HTTP server for receiving requests
  - Configurable response size via `response_size` parameter
  - Automatically started/stopped with fixture scope
