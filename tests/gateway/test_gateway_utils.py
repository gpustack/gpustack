from gpustack.gateway.utils import RoutePrefix


def test_flattened_prefixes():
    assert RoutePrefix(
        ["/chat/completions", "/completions", "/responses"]
    ).flattened_prefixes() == [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1-openai/chat/completions",
        "/v1-openai/completions",
        "/v1-openai/responses",
    ]


def test_regex_prefixes():
    assert RoutePrefix(
        ["/chat/completions", "/completions", "/responses"]
    ).regex_prefixes() == [
        r"/(v1)(-openai)?(/chat/completions)",
        r"/(v1)(-openai)?(/completions)",
        r"/(v1)(-openai)?(/responses)",
    ]
    assert RoutePrefix(
        ["/chat/completions", "/completions"], support_legacy=False
    ).regex_prefixes() == [
        r"/(v1)()(/chat/completions)",
        r"/(v1)()(/completions)",
    ]
