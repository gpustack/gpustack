from unittest.mock import MagicMock

from gpustack.gateway import transformer_plugin
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    get_4xx_5xx_fallback_value,
    get_ingress_fallback_envoyfilter,
)
from gpustack.gateway.utils import (
    gpustack_fallback_path_header,
    gpustack_original_path_header,
)


FALLBACK_PATH_VALUE_EXPR = f"%REQ({gpustack_original_path_header.upper()})%"


def _redirect_policy(typed_per_filter_config):
    return typed_per_filter_config["typed_per_filter_config"][
        "envoy.filters.http.custom_response"
    ]["value"]["custom_response_matcher"]["matcher_list"]["matchers"][0]["on_match"][
        "action"
    ][
        "typed_config"
    ][
        "value"
    ]


def test_get_4xx_5xx_fallback_value_default():
    value = get_4xx_5xx_fallback_value("ai-route-route-41.internal")
    policy = _redirect_policy(value)

    # Baseline redirect policy: keep the original request for re-entry so the
    # wasm transformer can see the same :path / body on the second pass.
    assert policy["use_original_request_uri"] is True
    assert policy["use_original_request_body"] is True
    assert policy["keep_original_response_code"] is False
    assert policy["only_redirect_upstream_code"] is False
    assert policy["max_internal_redirects"] == 10

    # Without extras, only the standard fallback-from marker is injected, and
    # it is mirrored on both request and response sides.
    expected_header = {
        "append": False,
        "header": {
            "key": "x-higress-fallback-from",
            "value": "ai-route-route-41.internal",
        },
    }
    assert policy["request_headers_to_add"] == [expected_header]
    assert policy["response_headers_to_add"] == [expected_header]

    # Trigger condition: fall back on any 4xx / 5xx response.
    matcher = value["typed_per_filter_config"]["envoy.filters.http.custom_response"][
        "value"
    ]["custom_response_matcher"]["matcher_list"]["matchers"][0]
    classes = sorted(
        p["single_predicate"]["value_match"]["exact"]
        for p in matcher["predicate"]["or_matcher"]["predicate"]
    )
    assert classes == ["4xx", "5xx"]


def test_get_4xx_5xx_fallback_value_with_extra_req_headers():
    policy = _redirect_policy(
        get_4xx_5xx_fallback_value(
            "ingress-x",
            fallback_header="x-custom-from",
            extra_req_headers={gpustack_fallback_path_header: FALLBACK_PATH_VALUE_EXPR},
        )
    )

    # Extras must be appended to request_headers_to_add so the redirected
    # request carries them back into the filter chain.
    assert policy["request_headers_to_add"] == [
        {
            "append": False,
            "header": {"key": "x-custom-from", "value": "ingress-x"},
        },
        {
            "append": False,
            "header": {
                "key": gpustack_fallback_path_header,
                "value": FALLBACK_PATH_VALUE_EXPR,
            },
        },
    ]

    # Guard against leaking the internal x-gpustack-fallback-path header back
    # to the downstream client via the response.
    assert policy["response_headers_to_add"] == [
        {
            "append": False,
            "header": {"key": "x-custom-from", "value": "ingress-x"},
        },
    ]


def test_get_ingress_fallback_envoyfilter_shape():
    ef = get_ingress_fallback_envoyfilter(
        ingress_name="ai-route-route-41.internal",
        namespace="higress-system",
        labels={"gpustack.ai/managed": "true"},
        extra_req_headers={gpustack_fallback_path_header: FALLBACK_PATH_VALUE_EXPR},
    )

    assert ef.metadata["name"] == "ai-route-route-41.internal"
    assert ef.metadata["namespace"] == "higress-system"
    assert ef.metadata["labels"] == {"gpustack.ai/managed": "true"}

    patch = ef.spec.configPatches[0]
    # The patch must target the exact route by name so fallback only fires
    # for the intended ingress.
    assert (
        patch.match.routeConfiguration.vhost.route.name == "ai-route-route-41.internal"
    )

    # extra_req_headers must flow through to the nested redirect policy.
    policy = _redirect_policy(patch.patch.value)
    req_keys = [h["header"]["key"] for h in policy["request_headers_to_add"]]
    assert gpustack_fallback_path_header in req_keys


def test_transformer_plugin_fallback_path_req_rules():
    # fix-4890: ai-proxy rewrites :path before upstream, so `use_original_request_uri`
    # is not enough. We snapshot :path into x-gpustack-original-path first, then on
    # redirect the EnvoyFilter injects x-gpustack-fallback-path=%REQ(...)% which this
    # plugin renames back to :path — so the ordering below is load-bearing.
    cfg = MagicMock()
    cfg.gateway_plugin_server_url = "http://127.0.0.1"
    _, spec = transformer_plugin(cfg)
    rules = spec.defaultConfig["reqRules"]

    rename_idx = next(
        i
        for i, r in enumerate(rules)
        if r["operate"] == "rename"
        and {"oldKey": gpustack_fallback_path_header, "newKey": ":path"} in r["headers"]
    )
    map_idx = next(
        i
        for i, r in enumerate(rules)
        if r["operate"] == "map"
        and r["headers"]
        == [{"fromKey": ":path", "toKey": gpustack_original_path_header}]
    )
    remove_idx = next(
        i
        for i, r in enumerate(rules)
        if r["operate"] == "remove"
        and any(h.get("key") == gpustack_fallback_path_header for h in r["headers"])
    )

    # rename must run before map so the redirect's fallback-path value lands in
    # :path before we snapshot; the final remove purges the helper header so it
    # is not visible to ai-proxy / upstream.
    assert rename_idx < map_idx < remove_idx
