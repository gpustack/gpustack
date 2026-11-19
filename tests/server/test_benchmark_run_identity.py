from types import SimpleNamespace

import pytest

from gpustack.api.exceptions import ConflictException
from gpustack.server.benchmark_run_identity import assert_benchmark_run


@pytest.mark.parametrize('submitted', [None, 'old-run'])
def test_identified_run_rejects_missing_or_stale_writer(submitted):
    with pytest.raises(ConflictException):
        assert_benchmark_run(SimpleNamespace(run_id='current-run'), submitted)


def test_current_writer_can_update_run():
    assert_benchmark_run(SimpleNamespace(run_id='current-run'), 'current-run')


def test_legacy_unidentified_run_accepts_legacy_writer():
    assert_benchmark_run(SimpleNamespace(run_id=None), None)


def test_user_stop_can_be_unscoped_but_not_explicitly_stale():
    benchmark = SimpleNamespace(run_id='current-run')
    assert_benchmark_run(benchmark, None, allow_unscoped=True)
    with pytest.raises(ConflictException):
        assert_benchmark_run(benchmark, 'old-run', allow_unscoped=True)


def test_legacy_record_cannot_accept_a_token_from_another_execution():
    with pytest.raises(ConflictException):
        assert_benchmark_run(SimpleNamespace(run_id=None), 'old-run')
