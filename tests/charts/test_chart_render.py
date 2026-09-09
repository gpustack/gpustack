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

import pathlib
import shutil
import subprocess

import pytest
import yaml

HELM = shutil.which("helm")

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
def chart_dependencies():
    """Vendor the chart's dependencies, which every render needs.

    `helm template` resolves dependencies while loading the chart — before any
    value is read — so a missing `charts/` fails every render here regardless of
    what the test asks for, including the ones whose sub-charts are conditioned
    off. The directory is generated, not committed, so a fresh checkout and a CI
    runner both start without it.

    `dependency update` rather than `build`: the lock file is not tracked either,
    and `build` refuses a repository that was never `helm repo add`-ed, which is
    the normal state of a runner.
    """
    chart = yaml.safe_load((CHART_DIR / "Chart.yaml").read_text())
    wanted = [d["name"] for d in chart.get("dependencies") or []]
    vendored = {path.name for path in (CHART_DIR / "charts").glob("*")}
    missing = [
        name for name in wanted if not any(entry.startswith(name) for entry in vendored)
    ]
    if not missing:
        return

    result = subprocess.run(
        [HELM, "dependency", "update", CHART],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        # Last line, not the whole log: `dependency update` narrates every repo
        # it refreshes before saying what went wrong.
        detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
        pytest.skip(
            f"chart dependencies unavailable ({', '.join(missing)}): "
            f"{detail[-1] if detail else 'helm dependency update failed'}"
        )


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
    result = subprocess.run(
        [HELM, "template", "gpustack", CHART, *BASE, *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"helm template failed:\n{result.stderr}")
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


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


class TestServerAndWorker:
    def test_deploys_both_and_the_operator(self):
        docs = render("--set", "worker.enabled=true")
        assert "gpustack-server" in names(docs, "StatefulSet")
        assert "gpustack-worker" in names(docs, "DaemonSet")
        assert "gpustack-operator-worker" in names(docs, "Deployment")
        assert "kueue-controller-manager" in names(docs, "Deployment")

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
