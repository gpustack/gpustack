import pytest

from gpustack.scheduler.calculator import GGUFParserCommandMutableParameters


def _parse(backend_parameters):
    params = GGUFParserCommandMutableParameters()
    params.from_args(backend_parameters)
    return params


@pytest.mark.parametrize(
    'backend_parameters,expected_ctx_size,expected_flash_attention,expected_parallel_size',
    [
        # The llama.cpp spelling: every parameter carries its value in the same
        # list entry.
        (['--flash-attn on', '--parallel 1', '-c 24576'], 24576, True, 1),
        (['--flash-attn off', '--parallel 1', '-c 24576'], 24576, False, 1),
        # The same parameters as separate list entries.
        (['--flash-attn', 'on', '--parallel', '1', '-c', '24576'], 24576, True, 1),
        # The value attached to the flag.
        (['--flash-attn=on', '--parallel=1', '--ctx-size=24576'], 24576, True, 1),
        # The long form of the context size.
        (['--ctx-size 24576'], 24576, None, 4),
        (['--ctx-size=24576'], 24576, None, 4),
        # The short form of the context size.
        (['-c 24576'], 24576, None, 4),
        (['-c', '24576'], 24576, None, 4),
        # The parameters that are absent keep their default.
        ([], 8192, None, 4),
        # The alternative spellings of the Flash Attention flag.
        (['--flash-attention on'], 8192, True, 4),
        (['--diffusion-fa on'], 8192, True, 4),
        (['-fa on'], 8192, True, 4),
        # The flag that turns Flash Attention off.
        (['--no-flash-attn'], 8192, False, 4),
        # "auto" leaves the decision to the backend, so the estimate takes the
        # state that uses more memory.
        (['--flash-attn auto', '-c 24576'], 24576, False, 4),
        # A value that is not one of the accepted words leaves the flag unset
        # and keeps the other parameters readable.
        (['--flash-attn maybe', '-c 24576'], 24576, None, 4),
    ],
)
def test_from_args_reads_the_backend_parameters(
    backend_parameters,
    expected_ctx_size,
    expected_flash_attention,
    expected_parallel_size,
):
    params = _parse(backend_parameters)
    assert params.ctx_size == expected_ctx_size
    assert params.flash_attention == expected_flash_attention
    assert params.parallel_size == expected_parallel_size


def test_from_args_keeps_the_other_parameters_readable():
    """A parameter that the parser does not know does not hide the known ones."""
    params = _parse(['--unknown-parameter 1', '-c 24576', '--parallel 1'])
    assert params.ctx_size == 24576
    assert params.parallel_size == 1


def test_from_args_does_not_mutate_the_backend_parameters():
    backend_parameters = ['--flash-attn on', '--parallel 1', '-c 24576']
    _parse(backend_parameters)
    assert backend_parameters == ['--flash-attn on', '--parallel 1', '-c 24576']


def test_from_args_survives_a_malformed_quotation():
    """A parameter that cannot be tokenized does not fail the whole parse."""
    params = _parse(["--header 'Bearer sk-xxx"])
    assert params.ctx_size == 8192


def test_extend_command_emits_the_parsed_parameters():
    params = _parse(['--flash-attn on', '--parallel 1', '-c 24576'])
    command = []
    params.extend_command(command)
    assert '--flash-attention=true' in command
    assert '--parallel-size=1' in command
    assert '--ctx-size=24576' in command


def test_extend_command_keeps_flash_attention_off_for_auto():
    """The backend decides, so the estimate asks for the state that uses more memory."""
    params = _parse(['--flash-attn', 'auto'])
    command = []
    params.extend_command(command)
    assert '--flash-attention=false' in command
    assert '--flash-attention=true' not in command
