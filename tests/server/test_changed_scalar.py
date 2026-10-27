import pytest

from gpustack.server.controllers import _changed_scalar


@pytest.mark.parametrize(
    "value,expected",
    [
        ((1,), 1),  # find_history: one-sided sequence, value present
        ((), None),  # find_history: empty side of a None<->value transition
        ([1], 1),  # find_history may hand back lists, not just tuples
        ([], None),
        (1, 1),  # detect_changes: flat scalar
        (None, None),
    ],
)
def test_changed_scalar_normalizes_both_event_shapes(value, expected):
    assert _changed_scalar(value) == expected
