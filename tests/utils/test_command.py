import shutil
import pytest

from gpustack.utils.command import (
    is_command_available,
    find_parameter,
    find_bool_parameter,
    get_versioned_command,
    extend_args_no_exist,
    flatten_to_argv,
    format_backend_parameters,
    is_parameter_key,
    safe_split,
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
        # Note: cases with whitespace AROUND '=' (e.g. "--foo =bar",
        # "--foo  =  bar") were dropped — under shlex tokenization "=" becomes
        # its own token and the result is no longer a meaningful CLI form.
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
        # Bare flag.
        (['--foo'], ['foo'], True),
        (['-f'], ['f'], True),
        (['--enable-feature'], ['enable-feature'], True),
        # Flag not present.
        (['--bar'], ['foo'], False),
        ([], ['foo'], False),
        # Whitespace around the token.
        ([' --foo'], ['foo'], True),
        (['  --enable-feature'], ['enable-feature'], True),
        (['--foo  '], ['foo'], True),
        (['  --foo  '], ['foo'], True),
        # Truthy values (space-separated).
        (['--foo', 'true'], ['foo'], True),
        (['--foo', '1'], ['foo'], True),
        (['--foo', 'yes'], ['foo'], True),
        (['--foo', 'on'], ['foo'], True),
        (['--foo', 'YES'], ['foo'], True),
        # Truthy values (= form).
        (['--foo=true'], ['foo'], True),
        (['--foo=1'], ['foo'], True),
        (['--foo=Yes'], ['foo'], True),
        # Falsy values (space-separated).
        (['--foo', 'false'], ['foo'], False),
        (['--foo', '0'], ['foo'], False),
        (['--foo', 'no'], ['foo'], False),
        (['--foo', 'off'], ['foo'], False),
        (['--foo', 'No'], ['foo'], False),
        # Falsy values (= form).
        (['--foo=false'], ['foo'], False),
        (['--foo=0'], ['foo'], False),
        (['--foo=NO'], ['foo'], False),
        # Unrecognized value -> optimistic True (user declared the key).
        (['--foo=bar'], ['foo'], True),
        (['--foo', 'always'], ['foo'], True),
        # First occurrence wins.
        (['--foo', 'false', '--foo'], ['foo'], False),
        (['--foo', '--foo=false'], ['foo'], True),
        # Followed by another key -> bare flag semantics.
        (['--foo', '--other'], ['foo'], True),
        # Multiple alias names.
        (['--bar=false'], ['foo', 'bar'], False),
        (['--foo', '--bar=true'], ['foo', 'bar'], True),
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


@pytest.mark.parametrize(
    'initial_args,new_args,expected',
    [
        # Branch 1: Add tuple (key, value) when key doesn't exist
        ([], [('--host', '127.0.0.1')], ['--host', '127.0.0.1']),
        # Branch 2: Skip tuple when key already exists (exact match)
        (['--host', '0.0.0.0'], [('--host', '127.0.0.1')], ['--host', '0.0.0.0']),
        # Branch 3: Skip tuple when key exists with = format
        (['--host=0.0.0.0'], [('--host', '127.0.0.1')], ['--host=0.0.0.0']),
        # Branch 4: Add single flag when it doesn't exist
        ([], ['--enable-metrics'], ['--enable-metrics']),
        # Branch 5: Skip single flag when it already exists
        (['--enable-metrics'], ['--enable-metrics'], ['--enable-metrics']),
        # Mixed: Add multiple arguments
        (
            ['--port', '8080'],
            [('--host', '127.0.0.1'), '--enable-metrics', ('--workers', '4')],
            [
                '--port',
                '8080',
                '--host',
                '127.0.0.1',
                '--enable-metrics',
                '--workers',
                '4',
            ],
        ),
        # Edge case: No new args to add
        (['--existing', 'value'], [], ['--existing', 'value']),
    ],
)
def test_extend_args_no_exist(initial_args, new_args, expected):
    extend_args_no_exist(initial_args, *new_args)
    assert initial_args == expected


@pytest.mark.parametrize(
    'parameters,expected',
    [
        (
            [
                '--max-model-len',
                '8192',
                '--disable-access-log-for-endpoints',
                '/metrics',
            ],
            ['--max-model-len=8192', '--disable-access-log-for-endpoints=/metrics'],
        ),
        (
            ['--enable-prefix-caching', '--max-seq-len=32768'],
            ['--enable-prefix-caching', '--max-seq-len=32768'],
        ),
        (
            ['--max-prefill-tokens', '-1', '--no-metrics'],
            ['--max-prefill-tokens=-1', '--no-metrics'],
        ),
        (None, []),
    ],
)
def test_format_backend_parameters(parameters, expected):
    assert format_backend_parameters(parameters) == expected


@pytest.mark.parametrize(
    'expr,split_expected,find_key,find_expected',
    [
        ('--tp 2 \\', ['--tp', '2'], 'tp', '2'),
        ('--tp 2 \\\n', ['--tp', '2'], 'tp', '2'),
        ('--tp 2 \\\r\n', ['--tp', '2'], 'tp', '2'),
        (
            '--model-path "C:\\Users\\foo"',
            ['--model-path', 'C:\\Users\\foo'],
            'model-path',
            'C:\\Users\\foo',
        ),
        ('--name foo\\\\', ['--name', 'foo\\'], 'name', 'foo\\'),
        (
            '--extra={"path":"a\\\\b"}',
            ['--extra={"path":"a\\\\b"}'],
            'extra',
            '{"path":"a\\\\b"}',
        ),
        ('\\', [], None, None),
    ],
)
def test_backslash_handling(expr, split_expected, find_key, find_expected):
    assert safe_split(expr) == split_expected
    if find_key is not None:
        assert find_parameter([expr], [find_key]) == find_expected


def test_safe_split_placeholder_collision():
    # A user-supplied literal that looks like the legacy JSON placeholder must
    # not be rewritten when a real JSON segment elsewhere triggers masking.
    tokens = safe_split('--foo=__JSON_0__ --config={"key":"value"}')
    assert tokens == ['--foo=__JSON_0__', '--config={"key":"value"}']


@pytest.mark.parametrize(
    'token,expected',
    [
        ('--foo', True),
        ('-f', True),
        ('--enable-expert-parallel', True),
        ('foo', False),
        ('', False),
        ('-', False),
        # POSIX end-of-options marker is not a key.
        ('--', False),
        # Negative numbers are values, not keys.
        ('-1', False),
        ('-0.5', False),
        ('-.5', False),
        ('-1e10', False),
    ],
)
def test_is_parameter_key(token, expected):
    assert is_parameter_key(token) is expected


@pytest.mark.parametrize(
    'parameters,expected',
    [
        # Empty / falsy inputs.
        (None, []),
        ([], []),
        ([''], []),
        (['   '], []),
        # Form A: one full --key value per element.
        (
            ['--max-model-len 8192', '--gpu-memory-utilization 0.9'],
            ['--max-model-len', '8192', '--gpu-memory-utilization', '0.9'],
        ),
        # Form B: --key=value per element.
        (['--max-model-len=8192'], ['--max-model-len=8192']),
        # Form C: flat argv, one token per element (identity).
        (
            ['--host', '0.0.0.0', '--port', '8080'],
            ['--host', '0.0.0.0', '--port', '8080'],
        ),
        # Form C with multi-value parameter (vLLM --lora-modules).
        (
            [
                '--lora-modules',
                '{"name": "x1", "path": "/p1"}',
                '{"name": "x2", "path": "/p2"}',
            ],
            [
                '--lora-modules',
                '{"name": "x1", "path": "/p1"}',
                '{"name": "x2", "path": "/p2"}',
            ],
        ),
        # Form D: a whole command line pasted into one element (issue #5200).
        (
            ['--a 1 --b=2 --flag'],
            ['--a', '1', '--b=2', '--flag'],
        ),
        # JSON in equal form (JSON-masking path).
        (
            ['--compilation-config={"cudagraph_mode": "FULL_DECODE_ONLY"}'],
            ['--compilation-config={"cudagraph_mode": "FULL_DECODE_ONLY"}'],
        ),
        # Quoted JSON in space form.
        (
            ['--speculative-config \'{"num_speculative_tokens": 2}\''],
            ['--speculative-config', '{"num_speculative_tokens": 2}'],
        ),
        # Negative numeric value is glued to its key (not mistaken for a flag).
        (
            ['--temperature -0.5'],
            ['--temperature', '-0.5'],
        ),
        # Shell line-continuation backslash inside one element.
        (
            ['--tp 2 \\\n--max-model-len 8192'],
            ['--tp', '2', '--max-model-len', '8192'],
        ),
        # Mixed shapes: A + B + D blended.
        (
            ['--tp 2', '--max-model-len=8192 --gpu-memory-utilization 0.9'],
            [
                '--tp',
                '2',
                '--max-model-len=8192',
                '--gpu-memory-utilization',
                '0.9',
            ],
        ),
    ],
)
def test_flatten_to_argv(parameters, expected):
    assert flatten_to_argv(parameters) == expected


def test_flatten_to_argv_issue_5200_full_paste():
    """Round-trip the long paste from the issue body."""
    pasted = (
        '--data-parallel-size 2 --data-parallel-size-local 1 '
        '--data-parallel-backend=ray --tensor-parallel-size 8 '
        '--enable-expert-parallel --tokenizer-mode deepseek_v32 '
        '--tool-call-parser deepseek_v32 --enable-auto-tool-choice '
        '--reasoning-parser deepseek_v3 --quantization ascend --seed 1024 '
        '--max-num-seqs 16 --max-model-len 65536 --max-num-batched-tokens 4096 '
        '--no-enable-prefix-caching --trust-remote-code '
        '--gpu-memory-utilization 0.92 '
        '--compilation-config={"cudagraph_mode": "FULL_DECODE_ONLY"} '
        '--speculative-config \'{"num_speculative_tokens": 2, "method": "deepseek_mtp"}\''
    )
    argv = flatten_to_argv([pasted])
    # 19 keys plus 17 values (2 flags have no value) → 36 tokens total.
    keys = [t for t in argv if is_parameter_key(t)]
    assert len(keys) == 19
    # JSON payloads should each be a single token.
    assert '{"cudagraph_mode": "FULL_DECODE_ONLY"}' in (
        ''.join(argv)
    )  # equal-form retained as one token
    assert '{"num_speculative_tokens": 2, "method": "deepseek_mtp"}' in argv


@pytest.mark.parametrize(
    'parameters,param_names,expected',
    [
        # Cross-element flat form (the case that previously bypassed validation).
        (['--gpu-memory-utilization', '0.8', '--port', '8080'], ['port'], '8080'),
        # Multi-key in one element (issue #5200).
        (['--gpu-memory-utilization 0.8 --port 8080'], ['port'], '8080'),
        (['--gpu-memory-utilization=0.8 --port=8080'], ['port'], '8080'),
        # Multi-value: find returns the first value.
        (
            ['--lora-modules', '{"name": "x1"}', '{"name": "x2"}'],
            ['lora-modules'],
            '{"name": "x1"}',
        ),
    ],
)
def test_find_parameter_argv_stream(parameters, param_names, expected):
    assert find_parameter(parameters, param_names) == expected


def test_find_bool_parameter_argv_stream():
    # Buried in a multi-param paste.
    assert find_bool_parameter(
        ['--tp 8 --enable-expert-parallel --max-model-len 1024'],
        ['enable-expert-parallel'],
    )
    # Cross-element flat form.
    assert find_bool_parameter(
        ['--tp', '8', '--enable-expert-parallel'],
        ['enable-expert-parallel'],
    )
    # Absent.
    assert not find_bool_parameter(
        ['--tp 8 --max-model-len 1024'],
        ['enable-expert-parallel'],
    )
