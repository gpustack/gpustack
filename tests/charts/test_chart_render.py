"""Render assertions for charts/gpustack-chart.

`helm lint` only checks that a chart renders; it says nothing about *what* it
renders. These tests pin the parts that are easy to break silently: which
components each mode deploys, where the workers are told to find their server,
which Secret carries the token, and which misconfigurations are refused outright
rather than installed into a broken cluster.

Skipped when `helm` is absent, or when the chart's dependencies cannot be
fetched, so neither a machine without helm nor an outage at somebody else's
chart repository turns this suite red.
"""

import json
import pathlib
import shutil
import subprocess
import sys
from copy import deepcopy
from functools import lru_cache

import pytest
import yaml

from gpustack import __operator_version__
from gpustack.gateway.client.networking_istio_io_v1alpha3_api import (
    get_ingress_tracing_envoyfilter,
)

HELM = shutil.which("helm")
SAFE_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

pytestmark = pytest.mark.skipif(HELM is None, reason="helm is not installed")

CHART = "charts/gpustack-chart"
CHART_DIR = pathlib.Path(CHART)

# The gpustack-operator chart accepts `global.hub` as an alias for its own
# `global.imageRegistry` from this version on. Below it, one registry value
# cannot cover the whole release; at or above it, it must. Deriving the
# expectation from the pin rather than from a hand-managed marker means bumping
# the dependency is the only edit — nothing is left to remember afterwards.
OPERATOR_HUB_ALIAS_SINCE = (0, 8, 7)

# The same release carries the other half of the `global.*` contract: from this
# version the operator chart falls back to `global.nodeSelector` for every
# workload it can confine, rather than for its own Deployment alone. Both changes
# merged before any tag was cut, so one threshold covers both — and if that ever
# stops being true, the interlock below says so on the bump rather than in a
# cluster.
OPERATOR_NODE_SELECTOR_SINCE = (0, 8, 7)


def pinned_operator_version() -> tuple[int, ...]:
    chart = yaml.safe_load((CHART_DIR / "Chart.yaml").read_text())
    for dependency in chart.get("dependencies") or []:
        if dependency.get("name") == "gpustack-operator":
            core = dependency["version"].split("-")[0].split("+")[0]
            return tuple(int(part) for part in core.split("."))
    # Absent dependency: treated as "before the alias". Its presence is asserted
    # in test_operator_version_pin.py, so this does not swallow a missing pin.
    return ()


@pytest.fixture(scope="module", autouse=True)
def chart_dependencies(tmp_path_factory):
    """Prepare dependencies without modifying the checkout's vendored archive.

    `helm template` resolves dependencies while loading the chart — before any
    value is read — so a missing `charts/` fails every render here regardless of
    what the test asks for, including when its sub-charts are disabled.

    `dependency update` rather than `build`: the lock file is not tracked either,
    and `build` refuses a repository that was never `helm repo add`-ed, which is
    the normal state of a runner.
    """
    render_path = tmp_path_factory.mktemp("helm-chart") / CHART_DIR.name
    shutil.copytree(CHART_DIR, render_path)
    chart = yaml.safe_load((render_path / "Chart.yaml").read_text())
    wanted = [d["name"] for d in chart.get("dependencies") or []]
    vendored = {path.name for path in (render_path / "charts").glob("*")}
    missing = [
        name for name in wanted if not any(entry.startswith(name) for entry in vendored)
    ]
    if missing:
        try:
            result = subprocess.run(
                [HELM, "dependency", "update", str(render_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            pytest.skip(f"chart dependency download timed out ({', '.join(missing)})")
        if result.returncode != 0:
            # Last line, not the whole log: `dependency update` narrates each repo.
            detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
            pytest.skip(
                f"chart dependencies unavailable ({', '.join(missing)}): "
                f"{detail[-1] if detail else 'helm dependency update failed'}"
            )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys.modules[__name__], "CHART", str(render_path))
        yield


# Every render has to name the gpustack image: `image.tag` is deliberately
# required so a checkout cannot silently pair these templates with the last
# released image. Any value works here — nothing is pulled.
BASE = ["--set", "image.tag=test"]
WORKER_ONLY = [
    "--set",
    "server.enabled=false",
    "--set",
    "higress-core.enabled=false",
    "--set",
    "worker.enabled=true",
]
# Worker-only refuses to render without both: there is no in-release server to
# address and no server to mint a token.
SERVER_AND_TOKEN = [
    "--set",
    "worker.serverURL=http://gpustack:30080",
    "--set",
    "registrationToken=from-server",
]


def render(*args: str) -> list[dict]:
    # Identical values render once, but assertions receive independent objects.
    return deepcopy(_render(*args))


@pytest.fixture(scope="module", autouse=True)
def render_cache():
    yield
    _render.cache_clear()


@lru_cache(maxsize=None)
def _render(*args: str) -> list[dict]:
    result = subprocess.run(
        [HELM, "template", "gpustack", CHART, *BASE, *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"helm template failed:\n{result.stderr}")
    return [doc for doc in yaml.load_all(result.stdout, Loader=SAFE_YAML_LOADER) if doc]


def render_error(*args: str) -> str:
    """Stderr of a render expected to be refused."""
    result = subprocess.run(
        [HELM, "template", "gpustack", CHART, *BASE, *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0, "render was expected to fail but succeeded"
    return result.stderr


def names(docs: list[dict], kind: str) -> set[str]:
    return {d["metadata"]["name"] for d in docs if d["kind"] == kind}


def container_env(docs: list[dict], kind: str, name: str) -> dict[str, str]:
    for doc in docs:
        if doc["kind"] == kind and doc["metadata"]["name"] == name:
            container = doc["spec"]["template"]["spec"]["containers"][0]
            return {
                e["name"]: e.get("value")
                for e in container.get("env", [])
                if "value" in e
            }
    pytest.fail(f"{kind}/{name} not rendered")


@pytest.mark.parametrize("trace_mode", ["builtin", "external", "disabled"])
@pytest.mark.parametrize("logs", [False, True])
def test_observability_endpoints_reach_server_and_workers(logs, trace_mode):
    log_endpoint = "https://logs.test:4318/v1/logs"
    external_endpoint = "collector.monitoring.svc:4317"
    trace_endpoint = {
        "builtin": "gpustack-tempo.inference.svc.tenant.test:4317",
        "external": external_endpoint,
        "disabled": None,
    }[trace_mode]
    docs = render(
        "--namespace",
        "inference",
        "--set",
        "worker.enabled=true,worker.gpuVendors={nvidia,amd},clusterDomain=tenant.test",
        "--set",
        f"tracing.enabled={str(trace_mode != 'disabled').lower()}",
        "--set-string",
        f"logging.endpoint={log_endpoint if logs else ''}",
        "--set-string",
        f"tracing.endpoint={external_endpoint if trace_mode != 'builtin' else ''}",
        "--set-string",
        f"tracing.queryURL={'https://tempo.example' if trace_mode != 'builtin' else ''}",
    )
    for kind, name in (
        ("StatefulSet", "gpustack-server"),
        ("DaemonSet", "gpustack-worker"),
        ("DaemonSet", "gpustack-worker-nvidia"),
        ("DaemonSet", "gpustack-worker-amd"),
    ):
        env = container_env(docs, kind, name)
        assert env.get("GPUSTACK_LOG_OTLP_ENDPOINT") == (log_endpoint if logs else None)
        assert env.get("GPUSTACK_TRACE_OTLP_ENDPOINT") == trace_endpoint
        assert env.get("GPUSTACK_DISABLE_TRACING") == (
            "true" if trace_mode == "disabled" else None
        )
    server = container_env(docs, "StatefulSet", "gpustack-server")
    assert (
        server.get("GPUSTACK_TRACE_QUERY_URL")
        == {
            "builtin": "http://127.0.0.1:3200",
            "external": "https://tempo.example",
            "disabled": None,
        }[trace_mode]
    )
    assert server.get("GPUSTACK_BUILTIN_TEMPO_POD") == (
        "gpustack-server-0" if trace_mode == "builtin" else None
    )
    assert server.get("GPUSTACK_BUILTIN_TEMPO_MAX_RETAINED_BYTES") == (
        "30000000000" if trace_mode == "builtin" else None
    )
    assert "GPUSTACK_BUILTIN_TEMPO_POD" not in container_env(
        docs, "DaemonSet", "gpustack-worker"
    )
    assert ("gpustack-tempo" in names(docs, "Service")) is (trace_mode == "builtin")
    assert "gpustack-tempo-query-tls" not in names(docs, "Secret")
    if trace_mode == "builtin":
        tempo = next(
            resource
            for resource in docs
            if resource["metadata"]["name"] == "gpustack-tempo"
        )
        assert [port["port"] for port in tempo["spec"]["ports"]] == [4317]
    assert "gpustack-higress-log-collector" not in names(docs, "DaemonSet")
    assert ("gpustack-otel-tracing" in names(docs, "EnvoyFilter")) is (
        logs or trace_mode != "disabled"
    )


@pytest.mark.parametrize("host_path", [None, "/srv/gpustack"])
def test_bundled_tempo_runs_with_the_server_data_volume(host_path, tmp_path):
    values = tmp_path / "tempo-values.yaml"
    values.write_text("tracing:\n  tempo:\n    maxRetainedBytes: 12000000000\n")
    docs = render(
        "--namespace",
        "inference",
        "--values",
        str(values),
        "--set",
        "server.replicas=2,tracing.tempo.retention=12h,"
        "tracing.advertiseEndpoint=trace.example:4317",
        "--set-string",
        "server.externalDatabaseURL=postgresql://user:pass@database.example/db",
        "--set-string",
        f"server.dataVolume.hostPath={host_path or ''}",
    )
    tempo = [doc for doc in docs if doc["metadata"]["name"] == "gpustack-tempo"]
    (service,) = tempo
    assert service["kind"] == "Service"
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["selector"] == {
        "statefulset.kubernetes.io/pod-name": "gpustack-server-0"
    }
    assert [
        (port["name"], port["port"], port["targetPort"])
        for port in service["spec"]["ports"]
    ] == [("otlp-grpc", 4317, 4317), ("https", 3200, 3200)]
    secret = next(
        doc
        for doc in docs
        if doc["kind"] == "Secret"
        and doc["metadata"]["name"] == "gpustack-tempo-query-tls"
    )
    assert secret["metadata"]["namespace"] == "inference"
    assert set(secret["data"]) == {
        "ca.crt",
        "server.crt",
        "server.key",
        "client.crt",
        "client.key",
    }
    server_workload = next(
        doc
        for doc in docs
        if doc["metadata"]["name"] == "gpustack-server" and doc["kind"] == "StatefulSet"
    )
    assert server_workload["spec"]["replicas"] == 2
    pod = server_workload["spec"]["template"]["spec"]
    (server_container,) = pod["containers"]
    assert {"mountPath": "/var/lib/gpustack", "name": "gpustack-data-dir"} in (
        server_container["volumeMounts"]
    )
    tls_dir = "/etc/gpustack/tempo-query-tls"
    assert {"mountPath": tls_dir, "name": "tempo-query-tls", "readOnly": True} in (
        server_container["volumeMounts"]
    )
    if host_path:
        assert {
            "name": "gpustack-data-dir",
            "hostPath": {"path": host_path, "type": "DirectoryOrCreate"},
        } in pod["volumes"]
        assert "volumeClaimTemplates" not in server_workload["spec"]
    else:
        (claim,) = server_workload["spec"]["volumeClaimTemplates"]
        assert claim["metadata"]["name"] == "gpustack-data-dir"
    server = container_env(docs, "StatefulSet", "gpustack-server")
    assert server["GPUSTACK_BUILTIN_TEMPO_POD"] == "gpustack-server-0"
    assert server["GPUSTACK_BUILTIN_TEMPO_RETENTION"] == "12h"
    assert server["GPUSTACK_BUILTIN_TEMPO_MAX_RETAINED_BYTES"] == "12000000000"
    assert server["GPUSTACK_TRACE_ADVERTISE_ENDPOINT"] == "trace.example:4317"
    assert server["GPUSTACK_TRACE_OTLP_ENDPOINT"] == (
        "gpustack-tempo.inference.svc.cluster.local:4317"
    )
    assert server["GPUSTACK_TRACE_QUERY_URL"] == (
        "https://gpustack-tempo.inference.svc.cluster.local:3200"
    )
    assert server["GPUSTACK_TRACE_QUERY_TLS_DIR"] == tls_dir
    assert server["GPUSTACK_BUILTIN_TEMPO_TLS_DIR"] == tls_dir


def test_bundled_tempo_can_mount_a_stable_query_tls_secret():
    def render_tls(name):
        docs = render(
            "--namespace",
            "inference",
            "--set",
            "server.replicas=2",
            "--set-string",
            "server.externalDatabaseURL=postgresql://database.example/db",
            "--set-string",
            f"tracing.tempo.queryTLSSecretName={name}",
        )
        assert not any(
            doc["metadata"]["name"].startswith("gpustack-tempo")
            for doc in docs
            if doc["kind"] == "Secret"
        )
        template = next(
            doc["spec"]["template"]
            for doc in docs
            if doc["kind"] == "StatefulSet"
            and doc["metadata"]["name"] == "gpustack-server"
        )
        volume = next(
            v for v in template["spec"]["volumes"] if v["name"] == "tempo-query-tls"
        )
        assert volume["secret"]["secretName"] == name
        env = container_env(docs, "StatefulSet", "gpustack-server")
        assert (
            env["GPUSTACK_TRACE_QUERY_URL"]
            == "https://gpustack-tempo.inference.svc.cluster.local:3200"
        )
        return template

    first = render_tls("tempo-tls-v1")
    assert render_tls("tempo-tls-v1") == first
    assert render_tls("tempo-tls-v2") != first


def test_external_query_url_does_not_use_the_bundled_tempo_client_credentials():
    docs = render(
        "--set",
        "server.replicas=2,tracing.queryURL=https://tempo.example",
        "--set-string",
        "server.externalDatabaseURL=postgresql://user:pass@database.example/db",
    )
    server = container_env(docs, "StatefulSet", "gpustack-server")
    assert server["GPUSTACK_TRACE_QUERY_URL"] == "https://tempo.example"
    assert "GPUSTACK_TRACE_QUERY_TLS_DIR" not in server
    assert "GPUSTACK_BUILTIN_TEMPO_TLS_DIR" in server


@pytest.mark.parametrize("auth_secret", ["", "logs-auth"])
def test_logging_references_existing_secrets(auth_secret):
    docs = render(
        "--set",
        f"logging.authorizationSecret.name={auth_secret},logging.caSecret.name=logs-ca",
        "--set-string",
        "logging.endpoint=https://collector:4318/v1/logs",
    )
    assert "logs-auth" not in names(docs, "Secret")
    server = container_env(docs, "StatefulSet", "gpustack-server")
    assert "GPUSTACK_LOG_OTLP_AUTHORIZATION" not in server
    assert "GPUSTACK_LOG_OTLP_CA_FILE" not in server
    alloy = next(
        doc
        for doc in docs
        if doc["kind"] == "Deployment" and doc["metadata"]["name"] == "gpustack-alloy"
    )
    pod = alloy["spec"]["template"]["spec"]
    env = {entry["name"]: entry for entry in pod["containers"][0]["env"]}
    if auth_secret:
        assert env["GPUSTACK_LOG_OTLP_AUTHORIZATION"]["valueFrom"]["secretKeyRef"] == {
            "name": auth_secret,
            "key": "authorization",
        }
    else:
        assert "GPUSTACK_LOG_OTLP_AUTHORIZATION" not in env
    ca = next(
        volume["secret"]
        for volume in pod["volumes"]
        if volume["name"] == "gpustack-otel-ca"
    )
    assert ca == {
        "secretName": "logs-ca",
        "items": [{"key": "ca.crt", "path": "ca.crt"}],
    }


@pytest.mark.parametrize(
    "patterns",
    [[], [r"\b1[3-9][0-9]{9}\b", r'(?i)"email"\s*:', r"\Q$(POD_NAMESPACE)\E"]],
)
def test_logging_redaction_patterns_reach_alloy(patterns):
    docs = render(
        "--set-string",
        "logging.endpoint=http://collector:4318/v1/logs",
        "--set-json",
        f"logging.redactPatterns={json.dumps(patterns)}",
    )
    alloy = container_env(docs, "Deployment", "gpustack-alloy")
    value = alloy["GPUSTACK_LOG_REDACT_PATTERNS"]
    assert "$(POD_NAMESPACE)" not in value
    assert json.loads(value) == patterns


@pytest.mark.parametrize("invalid", ["", 1, "[", r"(x)\1"])
def test_logging_rejects_invalid_redaction_pattern(invalid):
    error = render_error(
        "--set-string",
        "logging.endpoint=http://collector:4318/v1/logs",
        "--set-json",
        f"logging.redactPatterns={json.dumps(['valid', invalid])}",
    )
    assert "logging.redactPatterns[1] must be a non-empty RE2 expression" in error


@pytest.mark.parametrize("pattern", ["foo|", "(?:)", "a*", r"\B"])
def test_logging_rejects_redaction_patterns_that_match_the_empty_string(pattern):
    patterns = [r"\btoken\b", pattern]
    error = render_error(
        "--set-string",
        "logging.endpoint=http://collector:4318/v1/logs",
        "--set-json",
        f"logging.redactPatterns={json.dumps(patterns)}",
    )
    assert "logging.redactPatterns[1] must not match the empty string" in error


def test_legacy_higress_log_collection_values_do_not_mount_host_logs():
    docs = render(
        "--is-upgrade",
        "--set",
        "logging.higress.mode=nodeCollector",
        "--set-string",
        "logging.endpoint=https://collector:4318/v1/logs",
    )
    obsolete = "gpustack-higress-log-collector"
    assert obsolete not in names(docs, "DaemonSet")
    assert obsolete not in names(docs, "ConfigMap")
    for doc in docs:
        if doc["kind"] in {"Deployment", "DaemonSet", "StatefulSet"}:
            volumes = doc["spec"]["template"]["spec"].get("volumes", [])
            assert all(
                not volume.get("hostPath", {})
                .get("path", "")
                .startswith(("/var/log/pods", "/var/log/containers"))
                for volume in volumes
            )


@pytest.mark.parametrize("kind", ["Deployment", "DaemonSet"])
@pytest.mark.parametrize(
    "logs,traces", [(False, False), (False, True), (True, False), (True, True)]
)
def test_bundled_higress_uses_stdout_and_shared_alloy(kind, logs, traces):
    docs = render(
        "--namespace",
        "inference",
        "--set",
        f"higress-core.gateway.kind={kind},higress-core.gateway.name=edge",
        "--set",
        f"tracing.enabled={str(traces).lower()}",
        "--set-string",
        f"logging.endpoint={'https://otel:4318/v1/logs' if logs else ''}",
    )
    (gateway,) = (
        doc for doc in docs if doc["kind"] == kind and doc["metadata"]["name"] == "edge"
    )
    template = gateway["spec"]["template"]
    assert [entry["name"] for entry in template["spec"]["containers"]] == [
        "higress-gateway"
    ]
    labels = template["metadata"]["labels"]
    assert labels.get("gpustack.ai/log-source") == "higress"
    assert "gpustack.ai/log-release" not in labels

    (higress_config,) = (
        doc for doc in docs if doc["metadata"]["name"] == "higress-config"
    )
    mesh = yaml.safe_load(higress_config["data"]["mesh"])
    assert mesh["accessLogFile"] == "/dev/stdout"
    assert mesh["accessLogEncoding"] == "JSON"
    fields = json.loads(mesh["accessLogFormat"])
    assert fields["trace_id"] == "%REQ(X-B3-TRACEID)%"
    assert fields["traceparent"] == "%REQ(TRACEPARENT):55%"
    assert fields["path"] == "%REQ(X-ENVOY-ORIGINAL-PATH?:PATH)%"
    assert fields["request_id"] == "%REQ(X-REQUEST-ID)%"
    assert "GPUSTACK_HELM_HIGRESS_LOG_FORMAT" not in container_env(
        docs, "StatefulSet", "gpustack-server"
    )
    assert "gpustack-higress-file-log" not in names(docs, "EnvoyFilter")
    assert ("gpustack-alloy" in names(docs, "Deployment")) is logs
    if not logs:
        return
    alloy = next(
        doc
        for doc in docs
        if doc["kind"] == "Deployment" and doc["metadata"]["name"] == "gpustack-alloy"
    )
    assert alloy["spec"]["replicas"] == 1
    assert alloy["spec"]["strategy"]["type"] == "Recreate"
    alloy_pod = alloy["spec"]["template"]["spec"]
    assert alloy_pod["serviceAccountName"] == "gpustack-alloy"
    assert (
        container_env(docs, "Deployment", "gpustack-alloy")[
            "GPUSTACK_LOG_HIGRESS_SELECTOR"
        ]
        == "edge;inference-edge"
    )
    assert all("hostPath" not in volume for volume in alloy_pod["volumes"])
    role = next(
        doc
        for doc in docs
        if doc["kind"] == "Role" and doc["metadata"]["name"] == "gpustack-alloy"
    )
    assert role["metadata"]["namespace"] == "inference"
    assert {resource for rule in role["rules"] for resource in rule["resources"]} == {
        "pods",
        "pods/log",
    }
    assert "gpustack-alloy" not in names(docs, "ClusterRole")


def test_bundled_higress_log_format_survives_worker_values_upgrade():
    def get_mesh(docs):
        return next(
            doc["data"]["mesh"]
            for doc in docs
            if doc["metadata"]["name"] == "higress-config"
        )

    options = ("--set-string", "logging.endpoint=http://otel:4318/v1/logs")
    before = get_mesh(render(*options))
    after = get_mesh(render(*options, "--set-string", "worker.nodeSelector.zone=other"))
    assert before == after
    assert yaml.safe_load(after)["accessLogEncoding"] == "JSON"


def test_existing_gateway_agent_can_disable_only_gateway_collection():
    docs = render(
        "--set",
        "worker.enabled=true,logging.higress.enabled=false,"
        "logging.endpoint=http://otel:4318/v1/logs",
    )
    assert "GPUSTACK_LOG_HIGRESS_SELECTOR" not in container_env(
        docs, "Deployment", "gpustack-alloy"
    )
    assert "gpustack-alloy" in names(docs, "Deployment")
    for kind, name in (
        ("StatefulSet", "gpustack-server"),
        ("DaemonSet", "gpustack-worker"),
    ):
        assert container_env(docs, kind, name)["GPUSTACK_LOG_POD_RELEASE"] == "gpustack"


def test_bundled_higress_tracing_preserves_custom_stdout_format():
    docs = render(
        "--set",
        "tracing.endpoint=otel:4317,logging.endpoint=http://otel:4318/v1/logs,"
        "logging.higress.enabled=false,higress-core.meshConfig.accessLogEncoding=TEXT",
        "--set-string",
        "higress-core.meshConfig.accessLogFormat=%REQ(:METHOD)%",
    )
    higress_config = next(
        doc for doc in docs if doc["metadata"]["name"] == "higress-config"
    )
    mesh = yaml.safe_load(higress_config["data"]["mesh"])
    assert mesh["accessLogFormat"] == "%REQ(:METHOD)%"
    assert mesh["accessLogEncoding"] == "TEXT"
    assert "GPUSTACK_HELM_HIGRESS_LOG_FORMAT" not in container_env(
        docs, "StatefulSet", "gpustack-server"
    )


@pytest.mark.parametrize(
    "encoding,access_log_format",
    [
        ("JSON", '{"path":"%REQ(:PATH)%","request_id":"%REQ(X-REQUEST-ID)%"}'),
        ("TEXT", '{"path":"%REQ(:PATH)%","request_id":"%REQ(X-REQUEST-ID)%"}'),
        (
            "TEXT",
            '{"path":"%REQ(:PATH)%","request_id":"%REQ(X-REQUEST-ID)%",'
            '"response_code":%RESPONSE_CODE%}',
        ),
    ],
)
def test_bundled_higress_log_collection_preserves_custom_format(
    encoding, access_log_format
):
    docs = render(
        "--set",
        "logging.endpoint=http://otel:4318/v1/logs,"
        f"higress-core.meshConfig.accessLogEncoding={encoding}",
        "--set-literal",
        f"higress-core.meshConfig.accessLogFormat={access_log_format}",
    )
    configmap = next(doc for doc in docs if doc["metadata"]["name"] == "higress-config")
    mesh = yaml.safe_load(configmap["data"]["mesh"])
    assert mesh["accessLogEncoding"] == encoding
    assert mesh["accessLogFormat"] == access_log_format
    assert "GPUSTACK_HELM_HIGRESS_LOG_FORMAT" not in container_env(
        docs, "StatefulSet", "gpustack-server"
    )


@pytest.mark.parametrize(
    "setting,containers",
    [
        ("higress-core.global.enableIstioAPI=false", {"higress-gateway"}),
        ("higress-core.global.o11y.enabled=true", {"higress-gateway", "promtail"}),
    ],
)
def test_higress_logs_keep_existing_gateway_pod_layout(setting, containers):
    docs = render(
        "--set",
        f"{setting},logging.endpoint=http://otel:4318/v1/logs",
    )
    gateway = next(
        doc
        for doc in docs
        if doc["kind"] == "Deployment" and doc["metadata"]["name"] == "higress-gateway"
    )
    pod = gateway["spec"]["template"]["spec"]
    assert {container["name"] for container in pod["containers"]} == containers
    assert "automountServiceAccountToken" not in pod
    assert "gpustack-higress-file-log" not in names(docs, "EnvoyFilter")
    assert (
        container_env(docs, "StatefulSet", "gpustack-server")[
            "GPUSTACK_LOG_OTLP_ENDPOINT"
        ]
        == "http://otel:4318/v1/logs"
    )


@pytest.mark.parametrize(
    "endpoint",
    ["collector.monitoring.svc:4317", "10.32.1.2:4317", "[fd00::7]:4317", None, ""],
)
def test_bundled_higress_tracing_matches_embedded_settings(endpoint):
    setting = (
        f"tracing.endpoint={endpoint}"
        if endpoint is not None
        else "logging.endpoint=http://collector:4318/v1/logs"
    )
    docs = render(
        "--namespace",
        "inference",
        "--set",
        f"higress-core.gateway.name=edge,tracing.enabled={str(endpoint is not None).lower()}",
        "--set-string",
        setting,
    )
    effective_endpoint = (
        "gpustack-tempo.inference.svc.cluster.local:4317"
        if endpoint == ""
        else endpoint
    )
    expected = get_ingress_tracing_envoyfilter(
        "gpustack-otel-tracing", "inference", effective_endpoint
    )
    tracing = next(
        doc
        for doc in docs
        if doc["kind"] == "EnvoyFilter"
        and doc["metadata"]["name"] == "gpustack-otel-tracing"
    )
    assert (
        tracing["spec"]["configPatches"]
        == expected.model_dump(by_alias=True, mode="json", exclude_none=True)["spec"][
            "configPatches"
        ]
    )
    gateway = next(
        doc
        for doc in docs
        if doc["kind"] == "Deployment" and doc["metadata"]["name"] == "edge"
    )
    assert (
        tracing["spec"]["workloadSelector"]["labels"]
        == gateway["spec"]["selector"]["matchLabels"]
    )


def test_existing_gateway_and_worker_only_telemetry():
    endpoint = "collector.monitoring.svc:4317"
    settings = (
        "--set-string",
        "logging.endpoint=http://collector:4318/v1/logs",
        "--set-string",
        f"tracing.endpoint={endpoint}",
    )
    docs = render(
        "--set", "higress-core.enabled=false,gateway.namespace=gateway", *settings
    )
    server = container_env(docs, "StatefulSet", "gpustack-server")
    assert server["GPUSTACK_TRACE_OTLP_ENDPOINT"] == endpoint
    assert server["GPUSTACK_GATEWAY_NAMESPACE"] == "gateway"
    assert "gpustack-otel-tracing" not in names(docs, "EnvoyFilter")
    assert "GPUSTACK_LOG_HIGRESS_SELECTOR" not in container_env(
        docs, "Deployment", "gpustack-alloy"
    )
    assert "gpustack-higress-file-log" not in names(docs, "EnvoyFilter")
    assert "gpustack-higress-collector" not in names(docs, "ConfigMap")
    assert "gpustack-tempo" not in names(docs, "Service")
    docs = render(*WORKER_ONLY, *SERVER_AND_TOKEN, *settings)
    assert (
        container_env(docs, "DaemonSet", "gpustack-worker")[
            "GPUSTACK_TRACE_OTLP_ENDPOINT"
        ]
        == endpoint
    )
    assert "gpustack-otel-tracing" not in names(docs, "EnvoyFilter")
    assert "gpustack-higress-log-collector" not in names(docs, "DaemonSet")
    assert "gpustack-higress-file-log" not in names(docs, "EnvoyFilter")
    assert "gpustack-higress-collector" not in names(docs, "ConfigMap")
    assert "gpustack-tempo" not in names(docs, "Service")


def test_worker_only_without_a_trace_endpoint_has_no_tempo():
    docs = render(*WORKER_ONLY, *SERVER_AND_TOKEN)
    worker = container_env(docs, "DaemonSet", "gpustack-worker")
    assert "GPUSTACK_TRACE_OTLP_ENDPOINT" not in worker
    assert "gpustack-tempo" not in names(docs, "Service")


def test_builtin_tracing_does_not_require_the_higress_istio_api():
    docs = render("--set", "higress-core.global.enableIstioAPI=false")
    assert "gpustack-otel-tracing" not in names(docs, "EnvoyFilter")
    assert "gpustack-tempo" in names(docs, "Service")
    server = container_env(docs, "StatefulSet", "gpustack-server")
    assert server["GPUSTACK_TRACE_OTLP_ENDPOINT"] == (
        "gpustack-tempo.default.svc.cluster.local:4317"
    )


@pytest.mark.parametrize(
    "settings,message",
    [
        (
            "logging.endpoint=http://logs/v1/logs,logging.authorizationSecret.name=logs-auth",
            "require an HTTPS",
        ),
        ("logging.endpoint=https://user:password@logs/v1/logs", "without credentials"),
        ("tracing.endpoint=https://collector:4317", "OTLP/gRPC host:port"),
        ("tracing.endpoint=collector:65536", "OTLP/gRPC host:port"),
        ("tracing.advertiseEndpoint=collector:0", "tracing.advertiseEndpoint"),
        (
            "tracing.endpoint=collector:4317,higress-core.global.enableIstioAPI=false",
            "global.enableIstioAPI",
        ),
    ],
)
def test_invalid_telemetry_settings_are_rejected(settings, message):
    assert message in render_error("--set", settings)


class TestServerOnly:
    """The default: a control plane and no workers."""

    def test_deploys_server_without_worker_or_operator(self):
        docs = render()
        assert "gpustack-server" in names(docs, "StatefulSet")
        assert names(docs, "DaemonSet") == set()
        # The operator sub-chart is gated on worker.enabled; losing that
        # condition would silently add it, Kueue, NFD and the CSI drivers to
        # every server-only release.
        assert not [d for d in docs if "operator" in d["metadata"]["name"]]
        assert not [d for d in docs if "kueue" in d["metadata"]["name"].lower()]

    def test_no_registration_token_secret(self):
        # Nothing registers, so no token is minted.
        assert "registration-token" not in names(render(), "Secret")


class TestHigressPlugins:
    """The plugin-server Deployment rides the server image by default.

    The server publishes WasmPlugin URLs from its own bundled
    gpustack-higress-plugins manifest, so a plugin-server on a stale dedicated
    image 404s them and wedges the gateway listener. Sharing the server
    image.tag makes the two move together under `helm upgrade --set
    image.tag=...` without pulling a new chart.
    """

    def _deployment(self, docs: list[dict]) -> dict:
        for doc in docs:
            if doc["kind"] == "Deployment" and doc["metadata"]["name"].endswith(
                "higress-plugins"
            ):
                return doc
        pytest.fail("higress-plugins Deployment not rendered")

    def test_defaults_to_the_server_image(self):
        pod_spec = self._deployment(render())["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        assert container["image"] == "docker.io/gpustack/gpustack:test"
        # The server image's ENTRYPOINT funnels everything into
        # `gpustack start`; the plugin server must bypass it explicitly, with
        # the command the dedicated image's Dockerfile CMD already carries.
        assert container["command"] == [
            "gpustack-plugins",
            "start",
            "--port",
            "8080",
            "--host",
            "0.0.0.0",
        ]

    def test_pinned_dedicated_image_still_renders(self):
        docs = render(
            "--set",
            "higressPlugins.image.repository=gpustack/higress-plugins",
            "--set",
            "higressPlugins.image.tag=0.5.0.post2",
        )
        container = self._deployment(docs)["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "docker.io/gpustack/higress-plugins:0.5.0.post2"


class TestServerAndWorker:
    def test_deploys_both_and_the_operator(self):
        docs = render("--set", "worker.enabled=true")
        assert "gpustack-server" in names(docs, "StatefulSet")
        assert "gpustack-worker" in names(docs, "DaemonSet")
        assert "gpustack-operator-worker" in names(docs, "Deployment")
        assert "kueue-controller-manager" in names(docs, "Deployment")

    def test_operator_and_kueue_images_match_the_exported_release(self):
        docs = render("--set", "worker.enabled=true")
        images = {
            container["image"]
            for doc in docs
            if doc["kind"] == "Deployment"
            for container in doc["spec"]["template"]["spec"]["containers"]
        }
        assert f"docker.io/gpustack/gpustack-operator:{__operator_version__}" in images
        assert "docker.io/gpustack/mirrored-kueue:v0.18.11" in images

    def test_workers_address_the_in_release_server(self):
        docs = render("--set", "worker.enabled=true")
        env = container_env(docs, "DaemonSet", "gpustack-worker")
        assert env["GPUSTACK_SERVER_URL"] == "http://gpustack-server.default.svc:30080"

    def test_one_daemonset_per_vendor_plus_cpu(self):
        docs = render(
            "--set", "worker.enabled=true", "--set", "worker.gpuVendors={nvidia,amd}"
        )
        assert names(docs, "DaemonSet") >= {
            "gpustack-worker",
            "gpustack-worker-nvidia",
            "gpustack-worker-amd",
        }


@pytest.mark.parametrize(
    "certmanager, override, expected",
    [
        ("false", {}, {"insecureSkipVerify": True}),
        (
            "true",
            {},
            {
                "serverName": "kueue-controller-manager-metrics-service.default.svc",
                "insecureSkipVerify": False,
                "ca": {
                    "secret": {
                        "name": "kueue-metrics-server-cert",
                        "key": "ca.crt",
                    }
                },
            },
        ),
        (
            "true",
            {"serverName": "metrics.example.com", "insecureSkipVerify": False},
            {"serverName": "metrics.example.com", "insecureSkipVerify": False},
        ),
        (
            "false",
            {"serverName": "metrics.example.com"},
            {"serverName": "metrics.example.com", "insecureSkipVerify": True},
        ),
    ],
)
def test_kueue_metrics_tls(certmanager, override, expected):
    args = [
        "--set",
        "worker.enabled=true",
        "--set",
        "gpustack-operator.kueue.enablePrometheus=true",
        "--set",
        f"gpustack-operator.global.certmanager.enabled={certmanager}",
    ]
    for key, value in override.items():
        args.extend(
            [
                "--set",
                "gpustack-operator.kueue.metrics.serviceMonitor.tlsConfig."
                f"{key}={str(value).lower() if isinstance(value, bool) else value}",
            ]
        )
    monitors = [doc for doc in render(*args) if doc["kind"] == "ServiceMonitor"]
    assert len(monitors) == 1
    assert monitors[0]["spec"]["endpoints"][0]["tlsConfig"] == expected


class TestCPUWorkerDisabled:
    """`worker.cpuEnabled=false` — no workers on the nodes no runtime claims.

    The case it exists for is a control plane sharing the cluster with its GPU
    nodes: the CPU DaemonSet would otherwise cover exactly the nodes that must
    not gain a worker.
    """

    ARGS = (
        "--set",
        "worker.enabled=true",
        "--set",
        "worker.cpuEnabled=false",
        "--set",
        "worker.gpuVendors={nvidia}",
    )

    def test_renders_the_vendor_daemonsets_and_not_the_cpu_one(self):
        docs = render(*self.ARGS)
        daemonsets = names(docs, "DaemonSet")
        assert "gpustack-worker-nvidia" in daemonsets
        # Suffixed even as the only worker DaemonSet. Promoting it onto the
        # unsuffixed name would make it adopt the CPU DaemonSet's pods on an
        # upgrade and reschedule the whole runtime for a nodeSelector change.
        assert "gpustack-worker" not in daemonsets

    def test_the_worker_service_still_selects_the_daemonset_it_has(self):
        # The Service picks its selector from the same mode flag the DaemonSet
        # labels do. Deriving that flag from the DaemonSet count instead would
        # leave this single-vendor case labelled `app: gpustack-worker` and
        # selected by `component: worker`, i.e. a Service with no endpoints.
        docs = render(*self.ARGS)
        service = next(
            d
            for d in docs
            if d["kind"] == "Service" and d["metadata"]["name"] == "worker"
        )
        daemonset = next(
            d
            for d in docs
            if d["kind"] == "DaemonSet"
            and d["metadata"]["name"] == "gpustack-worker-nvidia"
        )
        labels = daemonset["spec"]["template"]["metadata"]["labels"]
        assert service["spec"]["selector"].items() <= labels.items()

    def test_refuses_a_release_with_no_worker_left_to_deploy(self):
        error = render_error(
            "--set",
            "worker.enabled=true",
            "--set",
            "worker.cpuEnabled=false",
            "--set",
            "worker.gpuVendors=null",
        )
        assert "no worker DaemonSet at all" in error

    def test_refuses_a_vendor_the_daemonsets_would_not_render(self):
        # The DaemonSet template iterates the *canonical* vendors and renders
        # nothing for a name outside them, so a guard counting the raw list
        # would pass a typo straight into the state it exists to reject.
        error = render_error(
            "--set",
            "worker.enabled=true",
            "--set",
            "worker.cpuEnabled=false",
            "--set",
            "worker.gpuVendors={bogus}",
        )
        assert "no supported GPU vendor" in error
        # The names it would have accepted, so the typo is fixable from the
        # message alone.
        assert "nvidia" in error

    def test_an_unset_value_keeps_the_cpu_daemonset(self):
        # Helm drops a key set to null, and plain truthiness would read that
        # `nil` as false — deleting the CPU workers from a cluster that only
        # meant to leave the key out.
        docs = render("--set", "worker.enabled=true", "--set", "worker.cpuEnabled=null")
        assert "gpustack-worker" in names(docs, "DaemonSet")

    @pytest.mark.parametrize("written", ["false", "False", "FALSE"])
    def test_a_string_false_turns_it_off_like_a_bool(self, written):
        # `--set-string` (and a values file quoting the value) would otherwise
        # be a non-empty string, i.e. read as "on" — the opposite of what it
        # says. YAML reads an unquoted `False` as the bool, so only the quoted
        # spellings reach the template with their case intact.
        docs = render(
            "--set",
            "worker.enabled=true",
            "--set-string",
            f"worker.cpuEnabled={written}",
        )
        assert "gpustack-worker" not in names(docs, "DaemonSet")

    def test_a_server_only_release_is_unaffected(self):
        # The guard is about workers; a control plane that renders none of them
        # must not be refused for switching this off.
        render("--set", "worker.cpuEnabled=false")


class TestWorkerOnly:
    def test_deploys_no_server_side_components(self):
        docs = render(*WORKER_ONLY, *SERVER_AND_TOKEN)
        assert names(docs, "StatefulSet") == set()
        assert "server-config" not in names(docs, "ConfigMap")
        assert not [d for d in docs if "higress" in d["metadata"]["name"]]
        # The workers and the operator are the point of this mode.
        assert "gpustack-worker" in names(docs, "DaemonSet")
        assert "gpustack-operator-worker" in names(docs, "Deployment")

    def test_workers_address_the_external_server(self):
        docs = render(*WORKER_ONLY, *SERVER_AND_TOKEN)
        env = container_env(docs, "DaemonSet", "gpustack-worker")
        assert env["GPUSTACK_SERVER_URL"] == "http://gpustack:30080"

    def test_supplied_token_wins_over_the_namespace(self):
        # Worker-only takes its token from the server that owns the cluster, so
        # the supplied value is authoritative — unlike the all-in-one mode, where
        # an existing Secret is preserved across upgrades.
        docs = render(
            *WORKER_ONLY,
            "--set",
            "worker.serverURL=http://gpustack:30080",
            "--set",
            "registrationToken=from-server",
        )
        secret = next(
            d
            for d in docs
            if d["kind"] == "Secret" and d["metadata"]["name"] == "registration-token"
        )
        assert secret["data"]["GPUSTACK_TOKEN"] == "ZnJvbS1zZXJ2ZXI="  # from-server

    def test_refuses_a_render_without_a_server_address(self):
        assert "worker.serverURL is required" in render_error(*WORKER_ONLY)

    def test_refuses_a_render_without_a_token(self):
        # No server to mint one, and generating a random token would produce
        # workers that can never register.
        error = render_error(
            *WORKER_ONLY, "--set", "worker.serverURL=http://gpustack:30080"
        )
        assert "registrationToken is required" in error


class TestRegistrationTokenSecretName:
    """Referencing a Secret this release does not own."""

    ARGS = (
        "--set",
        "worker.enabled=true",
        "--set",
        "registrationTokenSecretName=shared-token",
    )

    def test_creates_nothing_and_references_the_named_secret(self):
        docs = render(*self.ARGS)
        assert "shared-token" not in names(docs, "Secret")
        assert "registration-token" not in names(docs, "Secret")
        for kind, name in (
            ("DaemonSet", "gpustack-worker"),
            ("StatefulSet", "gpustack-server"),
        ):
            doc = next(
                d for d in docs if d["kind"] == kind and d["metadata"]["name"] == name
            )
            refs = [
                source["secretRef"]["name"]
                for source in doc["spec"]["template"]["spec"]["containers"][0].get(
                    "envFrom", []
                )
                if "secretRef" in source
            ]
            assert "shared-token" in refs, f"{kind}/{name} envFrom: {refs}"

    def test_no_token_required_when_the_secret_is_external(self):
        # The `required` on registrationToken must not fire for a Secret this
        # release is not going to write.
        render(
            *WORKER_ONLY,
            "--set",
            "worker.serverURL=http://gpustack:30080",
            "--set",
            "registrationTokenSecretName=shared-token",
        )


class TestImagePullSecret:
    """Two releases in one namespace can only share the canonical Secret."""

    def test_created_and_referenced_by_default(self):
        docs = render()
        assert "gpustack-image-pull-secret" in names(docs, "Secret")

    def test_create_false_references_without_creating(self):
        docs = render("--set", "imagePullSecret.create=false")
        assert "gpustack-image-pull-secret" not in names(docs, "Secret")
        # The reference is independent of creation — it comes from
        # global.imagePullSecrets, which sub-charts read too.
        pod = next(d for d in docs if d["kind"] == "StatefulSet")["spec"]["template"][
            "spec"
        ]
        assert pod["imagePullSecrets"] == [{"name": "gpustack-image-pull-secret"}]

    def test_refuses_credentials_that_would_be_discarded(self):
        error = render_error(
            "--set",
            "imagePullSecret.create=false",
            "--set",
            "imagePullSecret.credentials.username=u",
            "--set",
            "imagePullSecret.credentials.password=p",
        )
        assert "would be discarded" in error


class TestWorkerEnvironment:
    """The worker's environment is a contract with the runtime, not a detail.

    Registration used to render its own DaemonSet, and its template carried
    vendor-specific variables the chart did not. Deleting that renderer moved the
    contract here without moving those variables with it, so a MIG-partitioned
    NVIDIA cluster registered fine and then could not see its own MIG instances.
    These pin the ones whose absence is silent.
    """

    def test_the_nvidia_worker_declares_it_manages_mig(self):
        # The NVIDIA container runtime hides the driver's MIG capability subtree
        # from a container that does not declare it manages MIG, and the
        # management library authorizes partition-scoped calls by opening a file
        # under it. Without these, a MIG instance carved by the operator's
        # device manager is invisible to the worker and every partition-scoped
        # call fails with NO_PERMISSION — on a MIG cluster only, which is why no
        # install test would catch it.
        docs = render(
            "--set", "worker.enabled=true", "--set", "worker.gpuVendors={nvidia}"
        )
        vendor = next(
            doc
            for doc in docs
            if doc["kind"] == "DaemonSet"
            and doc["metadata"]["name"] == "gpustack-worker-nvidia"
        )
        env = {
            entry["name"]: entry.get("value")
            for entry in vendor["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        assert env.get("NVIDIA_MIG_CONFIG_DEVICES") == "all"
        assert env.get("NVIDIA_MIG_MONITOR_DEVICES") == "all"

    def test_the_cpu_worker_declares_nothing_vendor_specific(self):
        # The same variables on the CPU DaemonSet would ask the NVIDIA runtime
        # hook for a capability on nodes that have no driver to grant it.
        docs = render("--set", "worker.enabled=true")
        cpu = next(
            doc
            for doc in docs
            if doc["kind"] == "DaemonSet"
            and doc["metadata"]["name"] == "gpustack-worker"
        )
        env = {
            entry["name"]
            for entry in cpu["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        assert not {name for name in env if "MIG" in name}


class TestGuards:
    def test_refuses_an_empty_release(self):
        error = render_error(
            "--set", "server.enabled=false", "--set", "higress-core.enabled=false"
        )
        assert "Nothing to deploy" in error

    @pytest.mark.xfail(
        pinned_operator_version() < OPERATOR_HUB_ALIAS_SINCE,
        strict=True,
        reason=(
            "the pinned gpustack-operator predates the global.hub alias, so its "
            "tree renders without the mirror prefix. strict=True both ways: below "
            "the alias version this must fail, at or above it must pass."
        ),
    )
    def test_hub_covers_every_image(self):
        # One registry key has to reach everything: higress-core reads
        # global.hub natively, the operator tree reads global.imageRegistry and
        # accepts hub as its alias. Anything left behind means a mirrored install
        # that pulls fine on a connected cluster and fails halfway on an
        # air-gapped one. Rendering the real vendored sub-charts is what makes
        # this catch a regression in the operator's patched image helper.
        docs = render(
            "--set", "global.hub=mirror.example.com", "--set", "worker.enabled=true"
        )
        images = set()
        for doc in docs:
            if doc["kind"] not in ("Deployment", "DaemonSet", "StatefulSet", "Job"):
                continue
            pod = doc["spec"]["template"]["spec"]
            # `or []`, not a .get default: a key present with a null value would
            # otherwise put None into the concatenation.
            containers = (pod.get("containers") or []) + (
                pod.get("initContainers") or []
            )
            for container in containers:
                images.add(container["image"])
        strays = sorted(i for i in images if not i.startswith("mirror.example.com/"))
        assert not strays, f"images not pointing at the mirror: {strays}"

    def test_requires_an_image_tag(self):
        result = subprocess.run(
            [HELM, "template", "gpustack", CHART],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0
        assert "image.tag is required" in result.stderr

    @pytest.mark.xfail(
        pinned_operator_version() < OPERATOR_NODE_SELECTOR_SINCE,
        strict=True,
        reason=(
            "the pinned gpustack-operator falls back to global.nodeSelector for "
            "its own Deployment only, so the components it deploys render "
            "unconfined. strict=True both ways: below that version this must "
            "fail, at or above it must pass."
        ),
    )
    def test_the_node_selector_confines_what_the_operator_deploys(self):
        # The server hands a cluster's node selector over as `global.nodeSelector`
        # and nothing else, so what it reaches is decided entirely inside the
        # operator's tree — by a fallback in each sub-chart that no value here can
        # substitute for. Rendering the vendored sub-charts is the only thing that
        # says whether a bump actually delivered it.
        docs = render(
            "--set",
            "worker.enabled=true",
            "--set",
            "global.nodeSelector.gpustack\\.ai/pool=infra",
        )
        selectors = {
            doc["metadata"]["name"]: (
                doc["spec"]["template"]["spec"].get("nodeSelector") or {}
            )
            for doc in docs
            if doc["kind"] in ("Deployment", "DaemonSet", "StatefulSet")
        }

        for name in (
            "gpustack-operator-worker",
            "kueue-controller-manager",
            "node-feature-discovery-master",
            "node-feature-discovery-gc",
            "csi-nfs-controller",
            "csi-s3-controller",
        ):
            assert (
                selectors.get(name, {}).get("gpustack.ai/pool") == "infra"
            ), f"{name} is confined by global.nodeSelector"

        # And the ones that have to cover the nodes they serve are not: confining
        # NFD's worker would leave every node outside the pool unlabelled, which
        # is what the workers themselves select on.
        for name in ("node-feature-discovery-worker", "csi-nfs-node", "csi-s3-node"):
            assert "gpustack.ai/pool" not in selectors.get(
                name, {}
            ), f"{name} covers the nodes it serves and must not be confined"


class TestGatewayTLSVersions:
    """The chart's Ingress is the anchor the whole gateway's TLS floor comes from.

    GPUStack copies the bounds off it onto every Ingress it generates for an LLM
    route, so this one annotation is the only place they are set. The server is
    deliberately *not* told through its environment: two owners for one anchor is
    how the chart and the server end up disagreeing about it.
    """

    def ingress_annotations(self, docs: list[dict]) -> dict[str, str]:
        for doc in docs:
            if doc["kind"] == "Ingress" and doc["metadata"]["name"] == "gpustack":
                return doc["metadata"].get("annotations") or {}
        pytest.fail("Ingress/gpustack not rendered")

    def test_unset_by_default(self):
        # Existing releases keep Higress' defaults: stamping a floor on upgrade
        # would drop clients they still serve.
        assert self.ingress_annotations(render()) == {}

    def test_both_bounds_land_on_the_anchor(self):
        docs = render(
            "--set",
            "server.ingress.tls.minProtocolVersion=TLSv1.2",
            "--set",
            "server.ingress.tls.maxProtocolVersion=TLSv1.3",
        )
        assert self.ingress_annotations(docs) == {
            "higress.io/tls-min-protocol-version": "TLSv1.2",
            "higress.io/tls-max-protocol-version": "TLSv1.3",
        }

    def test_a_floor_alone_leaves_the_ceiling_to_higress(self):
        docs = render("--set", "server.ingress.tls.minProtocolVersion=TLSv1.2")
        assert self.ingress_annotations(docs) == {
            "higress.io/tls-min-protocol-version": "TLSv1.2"
        }

    def test_the_server_is_not_told_through_its_environment(self):
        # In-cluster the server reads the anchor, not its own environment. An env
        # var here would be a second owner for the same setting.
        docs = render("--set", "server.ingress.tls.minProtocolVersion=TLSv1.2")
        env = container_env(docs, "StatefulSet", "gpustack-server")
        assert not [name for name in env if "TLS_MIN" in name or "TLS_MAX" in name]

    def test_a_release_predating_these_keys_renders_unchanged(self):
        # A values file that sets only cert/key, as every existing release has.
        # The new keys must be absent rather than empty-valued: `higress.io/
        # tls-min-protocol-version: ""` is a value Higress rejects, and it
        # rejects it by keeping TLS 1.0.
        docs = render(
            "--set",
            "server.ingress.hostname=gpustack.example.com",
            "--set",
            "server.ingress.tls.cert=dGVzdA==",
            "--set",
            "server.ingress.tls.key=dGVzdA==",
        )
        assert self.ingress_annotations(docs) == {}

    def test_the_envoy_spelling_is_normalized(self):
        # TLSv1_2 is Envoy's spelling and the likeliest thing to reach for.
        # Higress rejects it, so the chart translates rather than passes it on.
        docs = render("--set", "server.ingress.tls.minProtocolVersion=tlsv1_2")
        assert self.ingress_annotations(docs) == {
            "higress.io/tls-min-protocol-version": "TLSv1.2"
        }

    def test_an_unusable_version_is_refused(self):
        # Refusing the install is the point: Higress ignores a value it cannot
        # parse and keeps accepting TLS 1.0, so a render that succeeded here
        # would look like a raised floor and be none.
        error = render_error("--set", "server.ingress.tls.minProtocolVersion=TLSv1.4")
        assert "is not a TLS version Higress accepts" in error

    def test_an_inverted_range_is_refused(self):
        error = render_error(
            "--set",
            "server.ingress.tls.minProtocolVersion=TLSv1.3",
            "--set",
            "server.ingress.tls.maxProtocolVersion=TLSv1.2",
        )
        assert "no TLS version would be accepted" in error

    def test_falsey_scalars_are_validated_not_skipped(self):
        # `with` treats false and 0 as unset. Neither is a TLS version, but
        # skipping them would render no annotation and leave TLS 1.0 in place --
        # the silent failure this validation exists to prevent.
        for value in ("false", "0"):
            error = render_error(
                "--set", f"server.ingress.tls.minProtocolVersion={value}"
            )
            assert "is not a TLS version Higress accepts" in error
            # The operator has to recognize what they typed in the message.
            assert f'"{value}"' in error


class TestWorkerOperatorSettingsAccess:
    """The worker SA carries the server's operator-settings calls through the
    cluster proxy, so it must be able to read and patch ``gpustack.ai``
    Settings."""

    def test_worker_can_read_and_patch_settings(self):
        docs = render("--set", "worker.enabled=true")
        role = next(
            d
            for d in docs
            if d["kind"] == "ClusterRole"
            and d["metadata"]["name"] == "gpustack-worker-workload-api"
        )
        rule = next(
            r
            for r in role["rules"]
            if "settings" in r.get("resources", [])
            and "gpustack.ai" in r.get("apiGroups", [])
        )
        assert {"get", "patch"} <= set(rule["verbs"])
        binding = next(
            d
            for d in docs
            if d["kind"] == "ClusterRoleBinding"
            and d["roleRef"]["name"] == role["metadata"]["name"]
        )
        assert binding["subjects"][0]["name"] == "gpustack-worker"
