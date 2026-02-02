import shutil
import pytest

from gpustack.utils.command import (
    is_command_available,
    find_parameter,
    find_bool_parameter,
    get_versioned_command,
)


def test_is_command_available_true(monkeypatch):
    # Simulate that shutil.which finds the command
    monkeypatch.setattr(shutil, 'which', lambda name: f'/usr/bin/{name}')
    assert is_command_available('foo') is True


def test_is_command_available_false(monkeypatch):
    # Simulate that shutil.which does not find the command
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    assert is_command_available('bar') is False


@pytest.mark.parametrize(
    'parameters,param_names,expected',
    [
        (None, ['foo'], None),
        ([], ['foo'], None),
        (['--foo=bar'], ['foo'], 'bar'),
        (['-f=baz'], ['f'], 'baz'),
        (['--foo=bar=baz'], ['foo'], 'bar=baz'),
        (['--foo', 'value'], ['foo'], 'value'),
        (['-f', 'val'], ['f'], 'val'),
        # multiple names: should match first key in parameters
        (['--x=5'], ['y', 'x'], '5'),
        # missing value after flag
        (['--foo'], ['foo'], None),
        # key not present
        (['--bar=1', '--baz', '2'], ['foo'], None),
        # Leading whitespace with = format (most common case)
        ([' --max-model-len=8192'], ['max-model-len'], '8192'),
        (['  --foo=bar'], ['foo'], 'bar'),
        (['  --foo bar'], ['foo'], 'bar'),
        # Trailing whitespace before =
        (['--foo =bar'], ['foo'], 'bar'),
        # Both leading and trailing whitespace
        (['  --foo  =bar'], ['foo'], 'bar'),
        # Multiple spaces around =
        (['--foo  =  bar'], ['foo'], '  bar'),
        (['--foo  ="  bar"'], ['foo'], '"  bar"'),
        (
            [
                '--hf-overrides \'{"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}\''
            ],
            ['hf-overrides'],
            '{"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}',
        ),
        (
            [
                '--hf-overrides={"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
            ],
            ['hf-overrides'],
            '{"rope_scaling": {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}',
        ),
    ],
)
def test_find_parameter(parameters, param_names, expected):
    assert find_parameter(parameters, param_names) == expected


@pytest.mark.parametrize(
    'parameters,param_names,expected',
    [
        # flag present
        (['--foo'], ['foo'], True),
        (['-f'], ['f'], True),
        (['--enable-feature'], ['enable-feature'], True),
        # key=value should not count as boolean flag
        (['--foo=bar'], ['foo'], False),
        # flag not present
        (['--bar'], ['foo'], False),
        ([], ['foo'], False),
        # Leading whitespace
        ([' --foo'], ['foo'], True),
        (['  --enable-feature'], ['enable-feature'], True),
        # Trailing whitespace
        (['--foo  '], ['foo'], True),
        # Both leading and trailing whitespace
        (['  --foo  '], ['foo'], True),
    ],
)
def test_find_bool_parameter(parameters, param_names, expected):
    assert find_bool_parameter(parameters, param_names) is expected


@pytest.mark.parametrize(
    'command_name,version,expected',
    [
        ('mycmd.exe', '1.0', 'mycmd_1.0.exe'),
        ('tool', '2.3.4', 'tool_2.3.4'),
        ('app.bin', 'v2', 'app.bin_v2'),
        ('', 'v', '_v'),
    ],
)
def test_get_versioned_command(command_name, version, expected):
    assert get_versioned_command(command_name, version) == expected
