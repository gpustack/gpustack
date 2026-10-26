import logging

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


class BasePathMiddleware:
    """Restore the mount prefix onto ``scope["path"]`` when a proxy stripped it.

    ASGI says ``root_path`` is a *prefix of* ``path``: an app mounted at
    ``/gpustack`` and asked for ``/gpustack/docs`` sees ``root_path="/gpustack"``
    and ``path="/gpustack/docs"``. Starlette's routing is written to that
    contract, and one place depends on it strictly. ``Mount.matches`` hands its
    child ``root_path + matched_path``, so ``/static`` under a ``/gpustack``
    root_path becomes ``root_path="/gpustack/static"`` — which the ``StaticFiles``
    inside then tries to strip off a path that never had it, leaving the mount
    looking for a file named after its own URL prefix. Plain routes survive
    because ``get_route_path`` gives up gracefully when ``path`` does not start
    with ``root_path``; mounts do not, and the three that serve the UI (``/css``,
    ``/js``, ``/static``) would 404 for every asset.

    That is not hypothetical, because the common nginx recipe strips: a
    ``proxy_pass`` with a trailing slash forwards ``/docs`` for a browser request
    to ``/gpustack/docs``, which is precisely the off-contract shape. So rather
    than support one proxy style and break the UI under the other, this
    normalises the shape once, outermost, and lets everything downstream see the
    spec's version of events.

    It also makes ``request.url`` agree with what the browser typed, which is
    what the redirects and cookie scoping downstream are reasoning about.

    A path that already carries the prefix is left alone, so a proxy that
    preserves it is unaffected — as is the root deployment, where ``prefix`` is
    empty and this middleware is not installed at all.
    """

    def __init__(self, app: ASGIApp, prefix: str):
        self.app = app
        self.prefix = prefix.rstrip("/")

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if not self.prefix or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # Exact match included: ``/gpustack`` is the prefix itself, already in
        # the shape root_path wants. Guarding on the separator rather than a bare
        # ``startswith`` keeps a sibling route like ``/gpustack-internal`` from
        # being mistaken for a path under the mount.
        if path == self.prefix or path.startswith(f"{self.prefix}/"):
            await self.app(scope, receive, send)
            return

        scope = dict(scope)
        scope["path"] = f"{self.prefix}{path}"
        raw_path = scope.get("raw_path")
        if raw_path is not None:
            # The undecoded twin of ``path``. Nothing in the installed Starlette
            # routes on it, but leaving the two disagreeing would plant a
            # difference that only shows up in whatever reads the other one.
            scope["raw_path"] = self.prefix.encode("latin-1") + raw_path
        await self.app(scope, receive, send)
