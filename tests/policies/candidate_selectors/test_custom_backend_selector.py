"""What a custom backend declares about the card it will take.

The selector books a member at its weights unless the engine says it will
take a fraction of the whole card up front. Only a declared fraction counts —
guessing one for an engine GPUStack does not model would make every backend
that does not pre-allocate ask for most of a card — so the reading has to see
everywhere a declaration can be written.
"""


def test_a_fraction_written_into_the_run_command_is_read():
    """A custom backend's invocation is `run_command` first and
    `backend_parameters` after it, and writing engine flags straight into the
    command is the ordinary way to use one. Reading only the parameters missed
    a declaration that was made plainly — and an unread fraction is a member
    booked at its weights while it holds most of the card."""
    from types import SimpleNamespace

    from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (  # noqa: E501
        CustomBackendResourceFitSelector,
    )

    selector = CustomBackendResourceFitSelector.__new__(
        CustomBackendResourceFitSelector
    )
    selector._model = SimpleNamespace(
        run_command="python -m vllm.entrypoints.openai.api_server "
        "--gpu-memory-utilization 0.95",
        backend_parameters=None,
    )

    assert selector._find_whole_card_fraction() == 0.95


def test_the_parameters_win_over_the_command():
    """They are appended after it, so argparse takes theirs."""
    from types import SimpleNamespace

    from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (  # noqa: E501
        CustomBackendResourceFitSelector,
    )

    selector = CustomBackendResourceFitSelector.__new__(
        CustomBackendResourceFitSelector
    )
    selector._model = SimpleNamespace(
        run_command="vllm serve --gpu-memory-utilization 0.95",
        backend_parameters=["--gpu-memory-utilization", "0.5"],
    )

    assert selector._find_whole_card_fraction() == 0.5


def test_an_unquotable_command_still_reads_the_parameters():
    """A malformed command is the worker's to refuse at start-up; here it
    costs its own half of the reading and no more."""
    from types import SimpleNamespace

    from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (  # noqa: E501
        CustomBackendResourceFitSelector,
    )

    selector = CustomBackendResourceFitSelector.__new__(
        CustomBackendResourceFitSelector
    )
    selector._model = SimpleNamespace(
        run_command='vllm serve --name "unbalanced',
        backend_parameters=["--gpu-memory-utilization", "0.8"],
    )

    assert selector._find_whole_card_fraction() == 0.8
