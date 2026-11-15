import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

import gpustack.routes.proxy as proxy_module
from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import register_handlers
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.routes.proxy import (
    canonicalize_hf_url,
    match_proxy_rule,
    may_attach_hf_token,
    process_headers,
    proxy_rule_header_func,
    router,
)
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.users import User


def make_user(is_admin: bool = False) -> User:
    return User(username="alice", is_admin=is_admin)


def make_tenant_context(
    is_platform_admin: bool = False, org_role=None
) -> TenantContext:
    return TenantContext(
        user=make_user(is_admin=is_platform_admin),
        is_platform_admin=is_platform_admin,
        current_principal_id=None,
        org_role=org_role,
    )


@pytest.fixture
def app(monkeypatch):
    app = FastAPI()
    register_handlers(app)
    app.include_router(router, prefix="/proxy")
    # The role of the caller is scenario-specific; every test that cares
    # overrides these again.
    app.dependency_overrides[get_current_user] = lambda: make_user(is_admin=True)
    app.dependency_overrides[get_tenant_context] = lambda: make_tenant_context()

    captured = {}

    async def fake_proxy_to(request, url, header_func=None):
        captured["url"] = url
        headers = process_headers(request.headers)
        if header_func is not None:
            headers = header_func(headers)
        captured["headers"] = headers
        return JSONResponse({})

    monkeypatch.setattr(proxy_module, "proxy_to", fake_proxy_to)
    app.state.captured = captured
    return app


@pytest.fixture
def client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def stub_global_config(monkeypatch, token):
    config = type("Config", (), {"huggingface_token": token})()
    monkeypatch.setattr(proxy_module, "get_global_config", lambda: config)


@pytest.mark.asyncio
async def test_proxy_allows_read_only_huggingface_paths(client):
    for url in [
        "https://huggingface.co/api/models?search=llama&limit=10",
        "https://huggingface.co/api/models/meta-llama/Llama-3-8b",
        "https://huggingface.co/api/models/meta-llama/Llama-3-8b/tree/main",
        "https://huggingface.co/meta-llama/Llama-3-8b/resolve/main/config.json",
    ]:
        response = await client.get("/proxy", params={"url": url})
        assert response.status_code == 200, url


@pytest.mark.asyncio
async def test_proxy_denies_path_traversal_variants(client):
    for url in [
        "https://huggingface.co/api/models/%2e%2e/whoami-v2",
        "https://huggingface.co/api/models/../whoami-v2",
        "https://huggingface.co/api/models/%2E%2E%2Fwhoami-v2",
        "https://huggingface.co/api/models/..",
    ]:
        response = await client.get("/proxy", params={"url": url})
        assert response.status_code == 403, url


@pytest.mark.asyncio
async def test_proxy_denies_huggingface_write_and_identity_paths(client):
    for method, url in [
        ("POST", "https://huggingface.co/api/repos/create"),
        ("DELETE", "https://huggingface.co/api/repos/delete"),
        ("PUT", "https://huggingface.co/api/models/x"),
        ("GET", "https://huggingface.co/api/whoami-v2"),
        ("POST", "https://huggingface.co/api/settings"),
    ]:
        response = await client.request(method, "/proxy", params={"url": url})
        assert response.status_code == 403, (method, url)


@pytest.mark.asyncio
async def test_proxy_denies_lookalike_hosts(client):
    for url in [
        "https://huggingface.co.evil.io/api/models",
        "https://evilhuggingface.co/api/models",
        "https://modelscope.cn/api/v1/dolphin/models",
    ]:
        response = await client.put("/proxy", params={"url": url})
        assert response.status_code == 403, url


@pytest.mark.asyncio
async def test_proxy_denies_malformed_urls(client, monkeypatch):
    # urlparse raises ValueError only when the malformed component is read
    # (bracketed host without closing bracket, non-numeric port); the rule
    # matcher turns that into "no rule" -> 403, never a 500.
    monkeypatch.setattr(proxy_module, "HF_ENDPOINT", "https://hf-mirror.com")
    for url in [
        "https://[invalid/api/models",
        "https://hf-mirror.com:invalid/api/models",
    ]:
        response = await client.get("/proxy", params={"url": url})
        assert response.status_code == 403, url


@pytest.mark.asyncio
async def test_proxy_allows_modelscope_search_put(client):
    response = await client.put(
        "/proxy",
        params={"url": "https://www.modelscope.cn/api/v1/dolphin/models"},
        json={"Name": "llama", "PageSize": 10},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_proxy_allows_modelscope_read_paths(client):
    for url in [
        "https://modelscope.cn/api/v1/models/LLM-research/meta-Llama-3-8B",
        "https://modelscope.cn/api/v1/models/LLM-research/meta-Llama-3-8B/repo/files?Revision=master",
        "https://modelscope.cn/LLM-research/meta-Llama-3-8B/resolve/master/config.json",
    ]:
        response = await client.get("/proxy", params={"url": url})
        assert response.status_code == 200, url


@pytest.mark.asyncio
async def test_proxy_token_only_on_huggingface_rules(client, app, monkeypatch):
    stub_global_config(monkeypatch, "hf_SERVER_TOKEN")

    response = await client.get(
        "/proxy", params={"url": "https://huggingface.co/api/models"}
    )
    assert response.status_code == 200
    assert app.state.captured["headers"]["Authorization"] == "Bearer hf_SERVER_TOKEN"

    response = await client.put(
        "/proxy", params={"url": "https://www.modelscope.cn/api/v1/dolphin/models"}
    )
    assert response.status_code == 200
    assert "Authorization" not in app.state.captured["headers"]


@pytest.mark.asyncio
async def test_proxy_token_reserved_for_admin_and_org_owner(app, client, monkeypatch):
    stub_global_config(monkeypatch, "hf_SERVER_TOKEN")
    url = "https://huggingface.co/org/private-repo/resolve/main/config.json"

    def as_caller(user, ctx):
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_tenant_context] = lambda: ctx

    # A plain member must not spend the server token on private repos:
    # the request still goes out, but anonymously.
    as_caller(make_user(is_admin=False), make_tenant_context(org_role=OrgRole.MEMBER))
    response = await client.get("/proxy", params={"url": url})
    assert response.status_code == 200
    assert "Authorization" not in app.state.captured["headers"]

    # An Org owner and a platform admin may.
    as_caller(make_user(is_admin=False), make_tenant_context(org_role=OrgRole.OWNER))
    response = await client.get("/proxy", params={"url": url})
    assert response.status_code == 200
    assert app.state.captured["headers"]["Authorization"] == "Bearer hf_SERVER_TOKEN"

    as_caller(make_user(is_admin=True), make_tenant_context())
    response = await client.get("/proxy", params={"url": url})
    assert response.status_code == 200
    assert app.state.captured["headers"]["Authorization"] == "Bearer hf_SERVER_TOKEN"


@pytest.mark.parametrize(
    "user,ctx,expected",
    [
        (make_user(is_admin=True), make_tenant_context(), True),
        (make_user(is_admin=False), make_tenant_context(is_platform_admin=True), True),
        (make_user(is_admin=False), make_tenant_context(org_role=OrgRole.OWNER), True),
        (
            make_user(is_admin=False),
            make_tenant_context(org_role=OrgRole.MEMBER),
            False,
        ),
        (make_user(is_admin=False), make_tenant_context(), False),
        (make_user(is_admin=False), None, False),
    ],
)
def test_may_attach_hf_token(user, ctx, expected):
    assert may_attach_hf_token(user, ctx) is expected


@pytest.mark.asyncio
async def test_proxy_strips_caller_credentials(client, app, monkeypatch):
    stub_global_config(monkeypatch, None)

    await client.get(
        "/proxy",
        params={"url": "https://modelscope.cn/api/v1/models/some/model"},
        headers={"Authorization": "Bearer user-token", "X-API-Key": "user-key"},
    )
    headers = app.state.captured["headers"]
    assert "Authorization" not in headers
    assert "X-API-Key" not in headers and "x-api-key" not in headers


@pytest.mark.parametrize(
    "endpoint,url,expected",
    [
        # No mirror configured: URL unchanged.
        (
            None,
            "https://huggingface.co/api/models?search=x",
            "https://huggingface.co/api/models?search=x",
        ),
        # Plain mirror host.
        (
            "https://hf-mirror.com",
            "https://hf-mirror.com/api/models",
            "https://huggingface.co/api/models",
        ),
        # Internal mirror with scheme and port.
        (
            "http://10.0.0.1:8080",
            "http://10.0.0.1:8080/org/repo/resolve/main/config.json",
            "https://huggingface.co/org/repo/resolve/main/config.json",
        ),
        # Path-prefixed mirror: prefix stripped on path-segment boundary.
        (
            "https://example.com/hf",
            "https://example.com/hf/api/models",
            "https://huggingface.co/api/models",
        ),
        # Same host but outside the prefix: not canonicalized.
        (
            "https://example.com/hf",
            "https://example.com/other/api/models",
            "https://example.com/other/api/models",
        ),
        # Lookalike host is never canonicalized.
        (
            "https://hf-mirror.com",
            "https://hf-mirror.com.evil.io/api/models",
            "https://hf-mirror.com.evil.io/api/models",
        ),
    ],
)
def test_canonicalize_hf_url(monkeypatch, endpoint, url, expected):
    monkeypatch.setattr(proxy_module, "HF_ENDPOINT", endpoint)
    assert canonicalize_hf_url(url) == expected


@pytest.mark.parametrize(
    "endpoint,method,url,allowed",
    [
        ("https://hf-mirror.com", "GET", "https://hf-mirror.com/api/models", True),
        (
            "https://hf-mirror.com",
            "GET",
            "https://hf-mirror.com/api/repos/create",
            False,
        ),
        ("https://hf-mirror.com", "POST", "https://hf-mirror.com/api/models", False),
        (
            "https://hf-mirror.com",
            "GET",
            "https://hf-mirror.com/org/repo/resolve/main/config.json",
            True,
        ),
        (
            "https://hf-mirror.com",
            "PUT",
            "https://hf-mirror.com/api/v1/dolphin/models",
            False,
        ),
        # A caller naming the mirror host directly gets the same rules as the
        # rewritten huggingface.co URL — mirrors are never more permissive.
        (
            "https://example.com/hf",
            "POST",
            "https://example.com/hf/api/repos/create",
            False,
        ),
    ],
)
def test_mirror_urls_follow_huggingface_rules(
    monkeypatch, endpoint, method, url, allowed
):
    monkeypatch.setattr(proxy_module, "HF_ENDPOINT", endpoint)
    rule = match_proxy_rule(method, url)
    assert (rule is not None) == allowed


@pytest.mark.parametrize(
    "endpoint,url,allowed",
    [
        # The port is part of the mirror origin: the same host on another
        # port is a different service and must not receive the server token.
        ("http://10.0.0.1:8080", "http://10.0.0.1:9000/api/models", False),
        ("http://10.0.0.1:8080", "http://10.0.0.1:8080/api/models", True),
        # Default ports are equivalent to their omitted form.
        ("https://hf-mirror.com", "https://hf-mirror.com:443/api/models", True),
        ("https://hf-mirror.com", "https://hf-mirror.com:8443/api/models", False),
    ],
)
def test_mirror_port_is_part_of_origin(monkeypatch, endpoint, url, allowed):
    monkeypatch.setattr(proxy_module, "HF_ENDPOINT", endpoint)
    rule = match_proxy_rule("GET", url)
    assert (rule is not None) == allowed


def test_proxy_rule_header_func_variants(monkeypatch):
    stub_global_config(monkeypatch, "hf_T")

    headers = proxy_rule_header_func(True, {"Accept": "application/json"})
    assert headers["Authorization"] == "Bearer hf_T"

    headers = proxy_rule_header_func(False, {"Authorization": "Bearer x"})
    assert "Authorization" not in headers
