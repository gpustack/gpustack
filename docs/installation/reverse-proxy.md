# Behind a Reverse Proxy

GPUStack can be served from a subpath of a shared hostname, for example
`https://example.com/gpustack/`, so it can sit alongside your other applications
on one origin and one certificate.

The UI needs no build-time configuration for this: it derives the mount prefix
from the browser's location at runtime. The server, however, cannot. A proxy that
strips the prefix leaves nothing in the request that names it — no header carries
the path — so the prefix has to be configured, with `--server-external-url`.

## Configuring the server

Set `--server-external-url` to the URL a browser uses, including the subpath:

```bash
gpustack start --server-external-url https://example.com/gpustack
```

This is the same flag used to derive `--trusted-hosts` and the SSO callback URLs,
so a deployment that already sets it only needs the path appended. The value is
the **external** URL — what your users type — not the `proxy_pass` target.

Compose it from your proxy's own configuration:

| Part | Comes from |
|---|---|
| scheme | whether the proxy terminates TLS |
| host | the proxy's `server_name`, plus its port if it is not 443/80 |
| path | the `location` prefix GPUStack is mounted under |

A trailing slash is accepted and normalized away. Omit the path entirely to serve
from the root, which is the default.

What the setting turns on:

- **`/docs` and `/openapi.json`.** The Swagger page is HTML that names absolute
  paths. Without the prefix it asks the origin root for `/openapi.json` and for
  the swagger-ui bundle — under a subpath mount, that is whichever application
  sits at `/`, not GPUStack.
- **Cookie scope.** Session cookies are issued with `Path=<prefix>` instead of
  `Path=/`, so they are not offered to the rest of a shared origin.
- **Tolerance of a proxy that cannot rewrite paths.** See
  [Proxies that preserve the prefix](#proxies-that-preserve-the-prefix).

## nginx

```nginx
server {
    listen 443 ssl;
    server_name example.com;

    location /gpustack/ {
        proxy_pass http://gpustack-server:80/;

        proxy_http_version 1.1;
        proxy_set_header Host $http_host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering off;
        proxy_read_timeout 3600s;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location = /gpustack {
        return 301 /gpustack/;
    }
}
```

One `location` is enough. If you find yourself adding more for `/v1`, `/v2`,
`/js` or `/css`, something above is misconfigured.

Most of those settings are load-bearing, and the server cannot compensate for any
of them:

| Setting | What breaks without it |
|---|---|
| `proxy_http_version 1.1` | Defaults to 1.0, which has no chunked encoding: streaming inference responses arrive only after the whole answer is generated. |
| `proxy_buffering off` | The playground appears frozen until a response completes, because nginx holds the incremental body. |
| `proxy_read_timeout` | Defaults to 60s, so long generations are cut off mid-stream. |
| `Upgrade` / `Connection` | Workers dial the server over a WebSocket and cannot connect. Not needed if workers reach the server directly — see [Worker traffic](#worker-traffic). |

The trailing slash on `proxy_pass` is what strips the prefix before forwarding.
It is the conventional form and the one this guide assumes, but it is not
required: with `--server-external-url` set, the server also routes requests that
still carry the prefix. See
[Proxies that preserve the prefix](#proxies-that-preserve-the-prefix).

`proxy_set_header Host $http_host` passes the browser's own `Host` through, port
included. **Prefer it over `$host`**, which is the same value with the port
stripped — identical on 443 and 80, wrong anywhere else, and wrong in a way that
surfaces far from its cause: the server builds absolute URLs from this header, so
what breaks is the address handed to a registering cluster, or the
`post_logout_redirect_uri` sent to an identity provider.

`Host` is taken as sent. `--trusted-hosts` is a separate control: it gates
`X-Forwarded-Host`, which the server honors in place of `Host` only from a host on
that allowlist. It defaults to the host in `--server-external-url`, and matching
ignores the port, so a proxy that sends `X-Forwarded-Host` needs no extra
configuration once the external URL is set.

## Ports: which one to proxy to

GPUStack listens on two ports, and which one your proxy should target depends on
whether the gateway is enabled:

| `--gateway-mode` | Proxy to |
|---|---|
| `auto` (the default; resolves to `embedded` outside Kubernetes), `embedded` | `--port` (default 80). The gateway is the public entry point, and the API server is bound to loopback. |
| `disabled` | `--api-port` (default 30080). |

## Proxies that preserve the prefix

Some proxies cannot rewrite the path. AWS ALB listener rules, for example, route
on a path but forward it unchanged, so the server receives
`/gpustack/v1/models` rather than `/v1/models`.

With `--server-external-url` set, GPUStack routes both forms, so this works
without further configuration.

One limitation applies with the gateway enabled: the gateway matches the
generic-proxy inference routes (`/model/proxy/...`) on an absolute path prefix,
which a preserved mount prefix does not match. Model routes served through
`/v1/...` are unaffected, as those are matched on the path suffix. If you need
generic-proxy routes behind a proxy that cannot strip the prefix, run with
`--gateway-mode disabled`.

## Worker traffic

Workers do not need to go through the proxy, and normally should not: they
register against `--server-url`, which can point straight at the server. Routing
them through a subpath proxy adds a hop with no benefit and requires the
WebSocket headers above.

Per-cluster registration URLs take precedence over `--server-external-url`, so a
cluster can be given a direct address while browsers keep using the proxied one.

## Cookie scope on a shared origin

Under a subpath mount, session cookies are issued with `Path=<prefix>`, so the
browser stops attaching them to requests for the rest of the hostname. Sharing an
origin with an unrelated application otherwise means sending it your session
cookie on every request it serves.

What path scoping does not do is make the origin a boundary. It controls which
requests carry the cookie, not who can cause such a request: script running on
the same origin can call GPUStack's API, and the browser will attach the cookie
because those requests are in scope. The cookies are `HttpOnly`, so that script
cannot read them, and `SameSite=Lax` — which treats every subdomain of a
registrable domain as the same site — will not keep it out either. Share an
origin only with applications you trust as much as GPUStack itself.

## Verifying a deployment

```bash
# Every asset the UI references must resolve. Relative paths here are correct:
# they resolve against the mount prefix.
curl -s https://example.com/gpustack/ | grep -oE '(src|href)="[^"]+"'

# The Swagger page must name only prefixed URLs.
curl -s https://example.com/gpustack/docs | grep -oE '"/[^"]*"'

# Logging in must set the cookie at the prefix, not at /.
curl -si -X POST https://example.com/gpustack/auth/login \
    -d 'username=admin' -d 'password=<password>' | grep -i set-cookie
```

If the UI comes up blank, its assets are being requested at the origin root
rather than under the prefix — the UI build predates subpath support. If the
Swagger page is blank, `--server-external-url` is unset or has no path.
