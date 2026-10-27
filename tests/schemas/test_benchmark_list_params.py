import pytest

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.benchmark import Benchmark, BenchmarkListParams

# The latency tails and the measured per-interval ITL the benchmark list can
# sort on. Named here rather than read off the whitelist so dropping one is a
# test failure instead of a silently narrower API.
LATENCY_SORT_FIELDS = [
    "time_to_first_token_p95",
    "time_to_first_token_p99",
    "inter_token_latency_p95",
    "inter_token_latency_p99",
    "itl_per_chunk_mean",
    "itl_per_chunk_p95",
    "itl_per_chunk_p99",
]


@pytest.mark.parametrize("field", BenchmarkListParams.sortable_fields)
def test_every_sortable_field_is_a_benchmark_column(field: str):
    """A whitelisted name that is not a column passes validation and then fails
    in the query, so the whitelist has to be checked against the table."""
    assert field in Benchmark.__table__.columns


@pytest.mark.parametrize("field", LATENCY_SORT_FIELDS)
@pytest.mark.parametrize("prefix", ["", "-"])
def test_latency_tails_are_sortable(field: str, prefix: str):
    params = BenchmarkListParams(sort_by=f"{prefix}{field}")

    assert params.order_by == [(field, "desc" if prefix == "-" else "asc")]


def test_a_non_column_field_is_rejected():
    with pytest.raises(InvalidException):
        BenchmarkListParams(sort_by="raw_metrics")
