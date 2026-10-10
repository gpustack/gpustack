from gpustack.utils.api_keys import mask_credential


def test_mask_credential_keeps_only_a_safe_suffix():
    assert mask_credential("abcdefghijkl") == "********ijkl"


def test_mask_credential_fully_masks_short_values():
    assert mask_credential("abcd") == "****"
    assert mask_credential("xy") == "**"


def test_mask_credential_marks_empty_values():
    assert mask_credential("") == "<empty>"
    assert mask_credential(None) == "<empty>"


def test_mask_credential_caps_log_growth_for_long_values():
    masked = mask_credential("x" * 10000)

    assert masked.endswith("xxxx")
    assert len(masked) == 68
