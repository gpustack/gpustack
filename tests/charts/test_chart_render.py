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
