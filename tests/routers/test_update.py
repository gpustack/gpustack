from gpustack.routes.update import is_newer_version


def test_is_newer_version():
    test_cases = [
        ("1.0.0", "0.2.3", True),
        ("0.2.1", "0.2.0", True),
        ("0.2.1", "0.1.0", True),
        ("0.2.0", "0.1.0", True),
        ("0.1.0", "0.2.0", False),
        ("0.1.0", "0.1.0", False),
        ("0.1.0", "0.1.0rc1", True),
        ("0.0.0", "0.0.0", False),
    ]

    for given, current, expected in test_cases:
        assert is_newer_version(given, current) == expected
