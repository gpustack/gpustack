from unittest.mock import patch

import pytest

from gpustack.utils.name import bounded_follower_name, bounded_instance_name


@pytest.mark.parametrize("role", [None, "prefill", "decode", "store", "r" * 100])
@pytest.mark.parametrize("length", [1, 43, 44, 49, 50, 57, 58, 63, 1000])
def test_instance_name_length_preserves_random_suffix(role, length):
    with patch("gpustack.utils.name.random.choices", return_value=list("a1b2c")):
        name = bounded_instance_name("m" * length, role)
    assert len(name) <= 59
    assert name.endswith("-a1b2c")
    assert name.startswith("m")
    if role and len(role) < 50:
        assert name.endswith(f"-{role}-a1b2c")


@pytest.mark.parametrize("role", [None, "", "prefill", "decode"])
def test_short_names_keep_their_format(role):
    with patch("gpustack.utils.name.random.choices", return_value=list("abc12")):
        name = bounded_instance_name("model", role)
    expected_role = f"-{role}" if role else ""
    assert name == f"model{expected_role}-abc12"


@pytest.mark.parametrize("index", [0, 9, 99, 100, 1000])
def test_follower_name_is_bounded_and_stable(index):
    instance = "m" * 53 + "-abc12"
    name = bounded_follower_name(instance, index)
    assert len(name) <= 63
    assert name.endswith(f"-f{index}")
    assert name == bounded_follower_name(instance, index)
    if index < 100:
        assert name == f"{instance}-f{index}"


def test_overflow_keeps_instance_and_follower_identities_distinct():
    first = "m" * 53 + "-abc12"
    second = "m" * 53 + "-def34"
    assert bounded_follower_name(first, 100) != bounded_follower_name(second, 100)
    assert bounded_follower_name(first, 100) != bounded_follower_name(first, 101)
