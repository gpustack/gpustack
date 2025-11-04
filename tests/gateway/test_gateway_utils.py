from gpustack.gateway.utils import RoutePrefix


def test_flattened_prefixes():
    assert RoutePrefix(["/chat/completions", "/completions"]).flattened_prefixes() == [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1-openai/chat/completions",
        "/v1-openai/completions",
    ]


def test_regex_prefixes():
    assert RoutePrefix(["/chat/completions", "/completions"]).regex_prefixes() == [
        r"/(v1)(-openai)?(/chat/completions)",
        r"/(v1)(-openai)?(/completions)",
    ]
    assert RoutePrefix(
        ["/chat/completions", "/completions"], support_legacy=False
    ).regex_prefixes() == [
        r"/(v1)()(/chat/completions)",
        r"/(v1)()(/completions)",
    ]
