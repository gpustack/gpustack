from starlette.types import ASGIApp, Receive, Scope, Send


class ForwardedHostPortMiddleware:
    """
    Middleware to support X-Forwarded-Host.
    It rewrites the 'server' and 'headers' in the ASGI scope accordingly.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            headers = dict((k.lower(), v) for k, v in scope.get("headers", []))
            # X-Forwarded-Host
            xfh = headers.get(b"x-forwarded-host")
            if xfh:
                # Only use the first value if multiple hosts are present
                host = xfh.split(b",")[0].strip()
                # Update the host header
                new_headers = [
                    (k, v) if k != b"host" else (b"host", host)
                    for k, v in scope["headers"]
                ]
                # If no host header, add it
                if not any(k == b"host" for k, _ in new_headers):
                    new_headers.append((b"host", host))
                scope["headers"] = new_headers
                # Optionally update scope["server"]
                server = list(scope.get("server", (None, None)))
                try:
                    host_str = host.decode()
                    if ":" in host_str:
                        host_name, host_port = host_str.rsplit(":", 1)
                        server[0] = host_name
                        server[1] = int(host_port)
                    else:
                        server[0] = host_str
                except Exception:
                    pass
                scope["server"] = tuple(server)
        await self.app(scope, receive, send)
