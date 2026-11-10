import os
from urllib.parse import unquote, urlparse, urlunparse
import aiohttp
from fastapi.responses import JSONResponse
import logging
import re
from typing import Annotated, Callable, Optional
from functools import partial

from fastapi import APIRouter, Depends, Request, Response

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import ForbiddenException
from gpustack.api.tenant import TenantContext, get_tenant_context
from gpustack.config.config import get_global_config
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.users import User
from gpustack.utils.network import use_proxy_env_for_url

router = APIRouter()

logger = logging.getLogger(__name__)


HEADER_FORWARDED_PREFIX = "x-forwarded-"
HEADER_SKIPPED = [
    "date",
    "set-cookie",
    "host",
    "port",
    "proto",
    "referer",
    "server",
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "cookie",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto",
    "x-forwarded-server",
]

# Credentials of the calling user must never reach the upstream host through
# the /proxy route: the server attaches the Hugging Face token itself when the
# matched rule allows it, and ModelScope requests need no credentials at all.
CREDENTIAL_HEADERS = ["authorization", "x-api-key"]

HF_ENDPOINT = os.getenv("HF_ENDPOINT")


def is_huggingface_url(url: str) -> bool:
    """Whether url points at the huggingface.co host (exact host match)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme == "https" and parsed.hostname == "huggingface.co"


def parse_hf_endpoint() -> Optional[dict]:
    """
    Parse HF_ENDPOINT into scheme / host / port / path prefix.

    Mirrors are administrator-configured (hf-mirror.com, an internal
    http://host:port, or a path-prefixed https://example.com/hf), so they are
    parsed strictly with urlparse instead of string prefix matching. The port
    is part of the origin: requests to the same host on a different port must
    not be treated as the mirror, or the server token could be sent to
    whatever unrelated service listens there.
    """
    if not HF_ENDPOINT:
        return None
    endpoint = HF_ENDPOINT if HF_ENDPOINT.endswith("/") else HF_ENDPOINT + "/"
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.hostname:
        return None
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "path_prefix": parsed.path.rstrip("/"),
    }


def _effective_port(scheme: Optional[str], port: Optional[int]) -> Optional[int]:
    if port is not None:
        return port
    return 443 if scheme == "https" else 80


def canonicalize_hf_url(url: str) -> str:
    """
    Map an HF_ENDPOINT mirror URL back to https://huggingface.co.

    Mirrors must satisfy exactly the same read-only rule set as the official
    host, so rule matching runs against the canonical URL regardless of which
    host the request actually targets.
    """
    endpoint = parse_hf_endpoint()
    if endpoint is None:
        return url
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if parsed.scheme != endpoint["scheme"] or parsed.hostname != endpoint["host"]:
        return url
    if _effective_port(parsed.scheme, parsed.port) != _effective_port(
        endpoint["scheme"], endpoint["port"]
    ):
        return url
    path = parsed.path
    if endpoint["path_prefix"]:
        if path == endpoint["path_prefix"]:
            path = "/"
        elif path.startswith(endpoint["path_prefix"] + "/"):
            path = path[len(endpoint["path_prefix"]) :]
        else:
            return url
    return urlunparse(("https", "huggingface.co", path, "", parsed.query, ""))


class ProxyRule:
    """
    An allow-list entry for the /proxy route: which methods may be forwarded
    to which scheme://host/path prefix, and whether the server-side Hugging
    Face token may be attached.
    """

    def __init__(self, methods: set, pattern: "re.Pattern", attach_token: bool):
        self.methods = methods
        self.pattern = pattern
        self.attach_token = attach_token

    def matches(self, method: str, canonical_url: str) -> bool:
        if method not in self.methods:
            return False
        parsed = urlparse(canonical_url)
        # aiohttp (yarl) removes dot segments — including percent-encoded ones
        # like %2e%2e — before sending, so a path such as
        # /api/models/%2e%2e/whoami-v2 would reach the upstream as
        # /api/whoami-v2 while matching the /api/models rule here. Match
        # against the decoded path and refuse dot segments entirely: no
        # allow-listed API shape needs path traversal.
        path = unquote(parsed.path)
        if "\\" in path or any(segment == ".." for segment in path.split("/")):
            return False
        target = urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip(
            "/"
        )
        return self.pattern.match(target) is not None


# Hugging Face: the UI only performs read-only model search, file listing and
# file downloads, so only those API shapes are proxied and only for them the
# server token is attached. Write endpoints (repos create/delete, settings,
# whoami) stay unreachable through /proxy.
#
# ModelScope: the search API (api/v1/dolphin/models) only accepts PUT with a
# JSON body, so that single path is allow-listed for PUT. No server-side
# token exists for ModelScope.
PROXY_RULES = [
    ProxyRule(
        {"GET", "HEAD"},
        re.compile(r"^https://huggingface\.co/api/models(/|$)"),
        attach_token=True,
    ),
    ProxyRule(
        {"GET", "HEAD"},
        re.compile(r"^https://huggingface\.co/[^/]+/[^/]+/(tree|resolve|blob)(/|$)"),
        attach_token=True,
    ),
    ProxyRule(
        {"PUT"},
        re.compile(r"^https://www\.modelscope\.cn/api/v1/dolphin/models$"),
        attach_token=False,
    ),
    ProxyRule(
        {"GET"},
        re.compile(r"^https://(www\.)?modelscope\.cn/api/v1/models(/|$)"),
        attach_token=False,
    ),
    ProxyRule(
        {"GET"},
        re.compile(r"^https://modelscope\.cn/[^/]+/[^/]+/resolve/"),
        attach_token=False,
    ),
]


def match_proxy_rule(method: str, url: str) -> Optional[ProxyRule]:
    """
    Find the allow-list entry permitting this method and URL.

    Matching runs on the canonical URL so HF_ENDPOINT mirrors follow the same
    rules as huggingface.co. Malformed URLs (unparseable IPv6 brackets,
    non-numeric ports, ...) fail closed: urlparse raises ValueError only when
    the malformed part is read, so every access stays inside this guard and
    maps to "no rule" instead of a server error.
    """
    try:
        canonical_url = canonicalize_hf_url(url)
        for rule in PROXY_RULES:
            if rule.matches(method, canonical_url):
                return rule
    except ValueError:
        return None
    return None


def replace_hf_endpoint(url: str) -> str:
    """
    Replace the huggingface.co domain with HF_ENDPOINT when set (mirror).
    """
    if HF_ENDPOINT and is_huggingface_url(url):
        return url.replace("https://huggingface.co", HF_ENDPOINT, 1)
    return url


def apply_hf_token_to_headers(url: str, headers: dict) -> dict:
    """
    Add Authorization Bearer when a token is configured (same rules as the proxy route).
    """
    global_config = get_global_config()
    is_hf = is_huggingface_url(url) or bool(HF_ENDPOINT and url.startswith(HF_ENDPOINT))
    if global_config.huggingface_token and is_hf:
        headers["Authorization"] = f"Bearer {global_config.huggingface_token}"
    return headers


def hf_hub_api_headers(url: str) -> dict:
    """Headers for JSON calls to the Hub HTTP API (after replace_hf_endpoint)."""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }
    return apply_hf_token_to_headers(url, headers)


timeout = aiohttp.ClientTimeout(
    connect=15.0,
    sock_read=60.0,
    sock_connect=10.0,
)


def may_attach_hf_token(user: Optional[User], ctx: Optional[TenantContext]) -> bool:
    """
    Whether the caller may spend the server-side Hugging Face token.

    The token reads every private / gated repository the administrator
    configured it for, so attaching it is reserved to platform admins and
    Org owners. Everyone else still reaches the allow-listed read-only
    paths, but anonymously — public model browsing keeps working while
    private repositories stay invisible.
    """
    if user is not None and user.is_admin:
        return True
    if ctx is not None and ctx.is_platform_admin:
        return True
    return ctx is not None and ctx.org_role == OrgRole.OWNER


@router.api_route("", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(
    request: Request,
    url: str,
    user: Annotated[User, Depends(get_current_user)],
    tenant_context: Annotated[TenantContext, Depends(get_tenant_context)],
):

    rule = match_proxy_rule(request.method, url)
    if rule is None:
        raise ForbiddenException(
            message="This site, path or HTTP method is not allowed"
        )

    effective_url = replace_hf_endpoint(url)
    attach_token = rule.attach_token and may_attach_hf_token(user, tenant_context)
    return await proxy_to(
        request,
        effective_url,
        header_func=partial(proxy_rule_header_func, attach_token),
    )


def proxy_rule_header_func(attach_token: bool, headers: dict) -> dict:
    """
    Header transform for the /proxy route: drop the caller's own credentials,
    then attach the server-side Hugging Face token only when both the matched
    rule and the caller's role permit spending it on this request.
    """
    for key in list(headers.keys()):
        if key.lower() in CREDENTIAL_HEADERS:
            del headers[key]

    if attach_token:
        global_config = get_global_config()
        if global_config.huggingface_token:
            headers["Authorization"] = f"Bearer {global_config.huggingface_token}"
    return headers


async def proxy_to(
    request: Request, url: str, header_func: Optional[Callable[[dict], dict]] = None
):
    forwarded_headers = process_headers(request.headers)
    if header_func is not None:
        forwarded_headers = header_func(forwarded_headers)
    try:
        data = (
            await request.body()
            if request.method in ["POST", "PUT", "DELETE"]
            else None
        )

        use_proxy_env = use_proxy_env_for_url(url)
        async with aiohttp.ClientSession(
            timeout=timeout, trust_env=use_proxy_env
        ) as session:
            async with session.request(
                method=request.method,
                url=url,
                headers=forwarded_headers,
                data=data,
            ) as resp:
                content = await resp.read()
                headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if k.lower() not in HEADER_SKIPPED
                }
                return Response(
                    status_code=resp.status,
                    content=content,
                    headers=headers,
                    media_type=headers.get("Content-Type"),
                )
    except Exception as e:
        logger.error(f"Error proxying request to {url}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to proxy the request."},
            media_type="application/json",
        )


def process_headers(headers: dict) -> dict:
    processed_headers = {}
    for key, value in headers.items():
        if key.lower() in HEADER_SKIPPED:
            continue
        elif key.lower().startswith(HEADER_FORWARDED_PREFIX):
            new_key = key[len(HEADER_FORWARDED_PREFIX) :]
            processed_headers[new_key] = value
        # set accept-encoding to identity to avoid decompression
        # httpx automatically decodes the content and we want to keep it raw
        # See https://www.python-httpx.org/quickstart/#binary-response-content
        elif key.lower() == "accept-encoding":
            processed_headers[key] = "identity"
        else:
            processed_headers[key] = value

    return processed_headers
