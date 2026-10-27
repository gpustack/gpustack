from gpustack.utils.version import major_version, pick_runtime_version


def test_pick_newest_at_or_below_host():
    versions = ["13", "12"]
    assert pick_runtime_version(versions, "13.0") == "13"
    assert pick_runtime_version(versions, "12.8") == "12"

    minors = ["12.8", "12.6", "12.4"]
    assert pick_runtime_version(minors, "12.7") == "12.6"
    assert pick_runtime_version(minors, "12.6") == "12.6"


def test_pick_all_newer_prefers_host_major_then_oldest():
    # same-major minor compatibility: 12.6 build serves a 12.4 host
    assert pick_runtime_version(["13", "12.6"], "12.4") == "12.6"
    # no same-major candidate: the oldest is the least-wrong guess
    assert pick_runtime_version(["13", "12.6"], "11.8") == "12.6"
    assert pick_runtime_version(["13"], "12.8") == "13"


def test_pick_without_host_takes_newest():
    assert pick_runtime_version(["12", "13"], None) == "13"


def test_pick_handles_prefixes_and_empty():
    assert pick_runtime_version(["v12.8", "v12.4"], "v12.6") == "v12.4"
    assert pick_runtime_version([], "12.8") is None
    assert major_version("v12.8") == "12"
    assert major_version(None) is None


def test_pick_tolerates_non_pep440_versions():
    """Runner/runtime version strings are not guaranteed PEP 440 (CANN
    ships e.g. "8.1.RC1.alpha003"); the picker must compare them without
    raising — this path serves every inference-backend image resolution,
    not just cache services."""
    candidates = ["8.1.RC1.alpha003", "8.0.RC2"]
    assert pick_runtime_version(candidates, "8.1.RC1.alpha003") == "8.1.RC1.alpha003"
    assert pick_runtime_version(candidates, "8.0.RC3") == "8.0.RC2"
    assert pick_runtime_version(candidates, None) == "8.1.RC1.alpha003"
