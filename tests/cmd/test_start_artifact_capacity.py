"""Capacity seeds can be supplied consistently through startup configuration."""

import argparse
import re

import pytest
from pydantic import ValidationError

from gpustack.config.config import Config
from gpustack.cmd.start import set_server_options, setup_start_cmd


def parse(values):
    parser = argparse.ArgumentParser()
    setup_start_cmd(parser.add_subparsers(dest='command'))
    return parser.parse_args(['start', *values])


@pytest.mark.parametrize('maximum', [0, 2**40])
def test_capacity_flag_reaches_the_settings(maximum, monkeypatch, tmp_path):
    monkeypatch.delenv('GPUSTACK_BENCHMARK_ARCHIVE_MAX_TOTAL_BYTES', raising=False)
    data = {}
    set_server_options(
        parse(['--benchmark-archive-max-total-bytes', str(maximum)]), data
    )
    assert data['benchmark_archive_max_total_bytes'] == maximum
    assert (
        Config(data_dir=str(tmp_path), **data).benchmark_archive_max_total_bytes
        == maximum
    )


def test_capacity_seed_supports_environment_and_unset_does_not_clobber_yaml(
    monkeypatch, tmp_path
):
    monkeypatch.setenv('GPUSTACK_BENCHMARK_ARCHIVE_MAX_TOTAL_BYTES', str(2**40))
    data = {}
    set_server_options(parse([]), data)
    assert (
        Config(data_dir=str(tmp_path), **data).benchmark_archive_max_total_bytes
        == 2**40
    )
    monkeypatch.delenv('GPUSTACK_BENCHMARK_ARCHIVE_MAX_TOTAL_BYTES')
    data = {'benchmark_archive_max_total_bytes': 0}
    set_server_options(parse([]), data)
    assert Config(data_dir=str(tmp_path), **data).benchmark_archive_max_total_bytes == 0


@pytest.mark.parametrize('maximum', [-1, 9007199254740992])
def test_invalid_capacity_seed_is_rejected(maximum, tmp_path):
    with pytest.raises(ValidationError):
        Config(data_dir=str(tmp_path), benchmark_archive_max_total_bytes=maximum)


def test_archive_flags_are_documented_in_server_settings(capsys):
    with pytest.raises(SystemExit) as exc:
        parse(['--help'])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    server_help, worker_help = help_text.split('Server settings:\n', 1)[1].split(
        'Worker settings:\n', 1
    )
    flags = [
        entry
        for entry in re.split(r'\n  (?=--)', server_help)
        if entry.startswith('--benchmark-archive-')
    ]
    assert len(flags) == 8
    assert all(len(flag.split(maxsplit=2)) == 3 for flag in flags)
    normalized_help = ' '.join(server_help.split())
    assert 'first initializes the shared policy' in normalized_help
    assert 'Archive Storage' in normalized_help
    assert '--benchmark-archive-' not in worker_help


def test_archive_source_interval_reaches_settings(tmp_path):
    data = {}
    set_server_options(
        parse(['--benchmark-archive-source-interval-seconds', '30']), data
    )
    assert (
        Config(data_dir=str(tmp_path), **data).benchmark_archive_source_interval_seconds
        == 30
    )
