"""The generated values have to be values the chart actually reads.

Asserting the dict's shape alone would pass for a key the chart ignores, which
is the failure this is most exposed to: a worker-only install that renders
cleanly and then deploys the wrong thing. So the important cases render the real
chart with the generated values and assert on the objects that come out.

Rendering is skipped when helm or the packaged chart is absent; the pure mapping
tests always run.
"""

import json
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

import pytest
import yaml

from gpustack.k8s.chart import chart_available
from gpustack.k8s.manifest_template import TemplateConfig
from gpustack.k8s.values import (
    CANONICAL_GPU_VENDORS,
    applied_revision,
    build_chart_values,
    split_image_reference,
)
from gpustack.schemas.clusters import (
    ImageCredential,
    K8sOptions,
    K8sVolumeMount,
    VolumeSource,
)
from gpustack_runtime.detector import ManufacturerEnum

HELM = shutil.which("helm")
# Anchored to the repository rather than the working directory, so `pytest` from
# a subdirectory renders the same chart as `pytest` from the root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHART = str(REPO_ROOT / "charts" / "gpustack-chart")


def config(**kwargs) -> TemplateConfig:
    defaults: Dict[str, Any] = {
        "token": "tok",
        "server_url": "http://gpustack.example.com:30080",
        "image": "docker.io/gpustack/gpustack:dev",
        "env": {},
        "args": [],
    }
    defaults.update(kwargs)
    return TemplateConfig(**defaults)


def render(values: Dict[str, Any], namespace: str = "gpustack-system") -> List[dict]:
    if HELM is None or not chart_available():
        pytest.skip("helm or the packaged chart is unavailable")
    # `tempfile` rather than the `mktemp` binary: its `-t` means a prefix on BSD
    # and a template on GNU, where a name without `XXXXXX` is refused outright —
    # so shelling out for it passes on a developer's macOS and fails on CI.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as with_values:
        yaml.safe_dump(values, with_values)
        with_values.flush()
        result = subprocess.run(
            [
                HELM,
                "template",
                "gpustack",
                CHART,
                "--namespace",
                namespace,
                "-f",
                with_values.name,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        pytest.fail(f"helm template failed:\n{result.stderr}")
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def worker_daemonset(docs: List[dict]) -> dict:
    """The gpustack worker DaemonSet, by name.

    The operator's tree contributes DaemonSets of its own — device managers, NFD,
    the CSI node plugins — so picking "the first DaemonSet" silently asserts
    against whichever one helm happened to emit first.
    """
    for doc in docs:
        if doc["kind"] == "DaemonSet" and doc["metadata"]["name"] == "gpustack-worker":
            return doc
    pytest.fail("the gpustack-worker DaemonSet was not rendered")


class TestSplitImageReference:
    @pytest.mark.parametrize(
        "reference,expected",
        [
            # A leading segment is a registry only when it carries a "." or ":",
            # which is the only thing telling these two apart.
            (
                "docker.io/gpustack/gpustack:dev",
                ("docker.io", "gpustack/gpustack", "dev"),
            ),
            ("gpustack/gpustack:dev", (None, "gpustack/gpustack", "dev")),
            (
                "localhost:5000/gpustack/gpustack:v1",
                ("localhost:5000", "gpustack/gpustack", "v1"),
            ),
            ("localhost/gpustack:v1", ("localhost", "gpustack", "v1")),
            ("gpustack/gpustack", (None, "gpustack/gpustack", "")),
        ],
    )
    def test_splits(self, reference, expected):
        assert split_image_reference(reference) == expected

    def test_refuses_a_digest(self):
        # The chart composes `repository:tag`; dropping the digest would deploy
        # a different image than the one named.
        with pytest.raises(ValueError, match="digest"):
            split_image_reference("gpustack/gpustack@sha256:" + "0" * 64)

    def test_an_untagged_worker_image_is_refused_here(self):
        # `split_image_reference` returns an empty tag rather than raising, since
        # the operator's image may legitimately go untagged. The worker's may
        # not: the chart requires `image.tag`, and left empty the failure lands
        # in the in-cluster Job's logs minutes later instead of in this response.
        with pytest.raises(ValueError, match="no tag"):
            build_chart_values(config(image="gpustack/gpustack"))


class TestGpuVendors:
    """The chart normalizes this list itself, so these pin the values, not the
    render: they are hashed into the revision the in-cluster Job compares
    against, and a list that reorders itself would spend a Helm revision
    installing an identical release."""

    def test_the_no_gpu_sentinel_is_not_a_vendor(self):
        values = build_chart_values(
            config(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.UNKNOWN])
        )
        assert values["worker"]["gpuVendors"] == ["nvidia"]

    def test_selection_order_does_not_change_the_values(self):
        one = build_chart_values(
            config(runtimes=[ManufacturerEnum.NVIDIA, ManufacturerEnum.AMD])
        )
        other = build_chart_values(
            config(runtimes=[ManufacturerEnum.AMD, ManufacturerEnum.NVIDIA])
        )
        assert one["worker"]["gpuVendors"] == other["worker"]["gpuVendors"]
        assert applied_revision(one) == applied_revision(other)

    def test_a_vendor_named_twice_is_one_vendor(self):
        values = build_chart_values(
            config(runtimes=[ManufacturerEnum.AMD, ManufacturerEnum.AMD])
        )
        assert values["worker"]["gpuVendors"] == ["amd"]

    def test_the_order_is_the_one_the_chart_calls_canonical(self):
        # Sorting agrees with the chart's canonical vendor order only because
        # that order is alphabetical. If it ever stops being, sorting here
        # silently disagrees with the order the DaemonSets render in, so read it
        # out of the chart and check the assumption rather than restate it.
        helper = (pathlib.Path(CHART) / "templates" / "_helper.tpl").read_text()
        match = re.search(
            r'define "gpustack\.canonicalVendorOrder".*?(\[[^]]*\])', helper, re.S
        )
        assert match, "the chart no longer declares a canonical vendor order"
        canonical = json.loads(match.group(1))
        assert canonical == sorted(canonical)

    def test_python_knows_the_same_vendors_the_chart_renders(self):
        # `CANONICAL_GPU_VENDORS` decides whether a manifest would deploy any
        # worker at all, and it is only right while it names what the chart
        # renders a DaemonSet for. A vendor added on one side alone makes the
        # server refuse a release the chart installs, or hand over an empty one.
        helper = (pathlib.Path(CHART) / "templates" / "_helper.tpl").read_text()
        match = re.search(
            r'define "gpustack\.canonicalVendorOrder".*?(\[[^]]*\])', helper, re.S
        )
        assert match, "the chart no longer declares a canonical vendor order"
        assert set(json.loads(match.group(1))) == CANONICAL_GPU_VENDORS


class TestCPUWorker:
    """`cpu_worker_enabled` — whether the manifest covers the nodes no GPU
    runtime claims. Off is for a cluster whose CPU-only nodes carry the control
    plane and must not gain a worker."""

    def test_enabled_by_default(self):
        assert build_chart_values(config())["worker"]["cpuEnabled"] is True

    def test_disabling_it_leaves_the_vendor_daemonsets_suffixed(self):
        # The name is the whole point: were the single remaining DaemonSet
        # promoted to `gpustack-worker`, it would adopt the CPU DaemonSet's pods
        # on the upgrade that disables this and reschedule the runtime.
        values = build_chart_values(
            config(runtimes=[ManufacturerEnum.NVIDIA], cpu_worker_enabled=False)
        )
        assert values["worker"]["cpuEnabled"] is False
        # DaemonSets only: the worker ServiceAccount and RBAC carry the
        # unsuffixed name too, and they are not per-runtime.
        daemonsets = {
            doc["metadata"]["name"]
            for doc in render(values)
            if doc["kind"] == "DaemonSet"
        }
        assert "gpustack-worker-nvidia" in daemonsets
        assert "gpustack-worker" not in daemonsets

    def test_refuses_a_manifest_that_would_deploy_no_worker(self):
        # The chart refuses it too, but there the failure lands in the
        # in-cluster Job's logs instead of in this response.
        with pytest.raises(ValueError, match="no worker at all"):
            build_chart_values(config(cpu_worker_enabled=False))

    def test_the_no_gpu_sentinel_is_not_a_selected_runtime(self):
        # `_gpu_vendors` drops it, so a request carrying only the sentinel
        # selects nothing — the case the guard has to catch on a non-empty list.
        with pytest.raises(ValueError, match="no worker at all"):
            build_chart_values(
                config(runtimes=[ManufacturerEnum.UNKNOWN], cpu_worker_enabled=False)
            )

    def test_an_overlay_can_supply_the_runtime_the_request_left_out(self):
        # `helmValues` reaches `worker.gpuVendors`, so the release this asks for
        # has a worker in it. Checking the request instead of the merged values
        # would refuse a manifest the chart installs happily.
        values = build_chart_values(
            config(
                cpu_worker_enabled=False,
                k8s_options=K8sOptions(
                    helm_values={"worker": {"gpuVendors": ["nvidia"]}}
                ),
            )
        )
        assert values["worker"]["gpuVendors"] == ["nvidia"]
        daemonsets = {
            doc["metadata"]["name"]
            for doc in render(values)
            if doc["kind"] == "DaemonSet"
        }
        assert "gpustack-worker-nvidia" in daemonsets
        assert "gpustack-worker" not in daemonsets

    @pytest.mark.parametrize("written", [False, "false", "False"])
    def test_an_overlay_can_be_what_empties_the_release(self, written):
        # The mirror image: the request kept the CPU worker, the overlay took it
        # away, and nothing selected a runtime. Read off the request this passes
        # and fails in the cluster instead. Any spelling of false, as in the
        # chart.
        with pytest.raises(ValueError, match="no worker at all"):
            build_chart_values(
                config(
                    k8s_options=K8sOptions(
                        helm_values={"worker": {"cpuEnabled": written}}
                    )
                )
            )

    def test_an_overlay_vendor_the_chart_would_drop_is_not_a_worker(self):
        # The chart renders a DaemonSet only for a vendor it knows, so a name
        # outside that set leaves the release as empty as no name at all.
        with pytest.raises(ValueError, match="no worker at all"):
            build_chart_values(
                config(
                    cpu_worker_enabled=False,
                    k8s_options=K8sOptions(
                        helm_values={"worker": {"gpuVendors": ["bogus"]}}
                    ),
                )
            )


class TestWorkerOnlyValues:
    def test_renders_workers_and_the_operator_but_no_server(self):
        docs = render(build_chart_values(config(runtimes=[ManufacturerEnum.NVIDIA])))
        kinds = {doc["kind"] for doc in docs}
        names = {doc["metadata"]["name"] for doc in docs}
        assert "StatefulSet" not in kinds
        assert {"gpustack-worker", "gpustack-worker-nvidia"} <= names
        assert "gpustack-operator-worker" in names

    def test_workers_are_told_where_the_server_is(self):
        docs = render(build_chart_values(config()))
        # By name: the operator's tree brings DaemonSets of its own (device
        # managers, NFD, the CSI node plugins), so "the first DaemonSet" is not
        # the worker.
        daemonset = worker_daemonset(docs)
        env = {
            e["name"]: e.get("value")
            for e in daemonset["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        assert env["GPUSTACK_SERVER_URL"] == "http://gpustack.example.com:30080"

    def test_the_chart_does_not_create_the_token_secret(self):
        # The bootstrap manifest owns it, so a re-render must not be able to
        # rotate or delete the token.
        docs = render(build_chart_values(config()))
        assert not [
            d
            for d in docs
            if d["kind"] == "Secret" and d["metadata"]["name"] == "registration-token"
        ]

    def test_the_chart_does_not_create_pull_secrets_but_references_them(self):
        values = build_chart_values(
            config(
                k8s_options=K8sOptions(
                    image_credentials=[
                        ImageCredential(
                            registry="reg.example.com", username="u", password="p"
                        )
                    ]
                )
            )
        )
        docs = render(values)
        assert not [
            d
            for d in docs
            if d["kind"] == "Secret" and "image-pull-secret" in d["metadata"]["name"]
        ]
        referenced = {
            ref["name"]
            for doc in docs
            if doc["kind"] in ("DaemonSet", "Deployment")
            for ref in doc["spec"]["template"]["spec"].get("imagePullSecrets") or []
        }
        assert referenced == {"gpustack-image-pull-secret-0"}

    def test_no_credentials_means_no_dangling_reference(self):
        # The chart's default references the Secret it would otherwise create;
        # with create=false that would point every pod at a Secret nobody made.
        docs = render(build_chart_values(config()))
        for doc in docs:
            if doc["kind"] not in ("DaemonSet", "Deployment", "Job"):
                continue
            assert not doc["spec"]["template"]["spec"].get("imagePullSecrets")

    def test_registry_reaches_both_subtrees(self):
        values = build_chart_values(
            config(image="mirror.example.com/gpustack/gpustack:dev")
        )
        docs = render(values)
        images = set()
        for doc in docs:
            if doc["kind"] not in ("Deployment", "DaemonSet", "StatefulSet", "Job"):
                continue
            pod = doc["spec"]["template"]["spec"]
            for container in (pod.get("containers") or []) + (
                pod.get("initContainers") or []
            ):
                images.add(container["image"])
        strays = sorted(i for i in images if not i.startswith("mirror.example.com/"))
        assert not strays, f"images not pointing at the mirror: {strays}"

    def test_data_dir_and_extra_mounts_reach_the_daemonset(self):
        options = K8sOptions(
            volume_mounts=[
                K8sVolumeMount(
                    name="gpustack-data-dir",
                    mountPath="/var/lib/gpustack",
                    volumeSource=VolumeSource.model_validate(
                        {
                            "hostPath": {
                                "path": "/data/gpustack",
                                "type": "DirectoryOrCreate",
                            }
                        }
                    ),
                ),
                K8sVolumeMount(
                    name="extra-lib",
                    mountPath="/opt/lib",
                    readOnly=True,
                    volumeSource=VolumeSource.model_validate(
                        {"hostPath": {"path": "/opt/lib", "type": "Directory"}}
                    ),
                ),
            ]
        )
        docs = render(build_chart_values(config(k8s_options=options)))
        pod = worker_daemonset(docs)["spec"]["template"]["spec"]
        volumes = {v["name"]: v for v in pod["volumes"]}
        assert volumes["gpustack-data-dir"]["hostPath"]["path"] == "/data/gpustack"
        assert volumes["extra-lib"]["hostPath"]["path"] == "/opt/lib"
        mounts = {m["name"]: m for m in pod["containers"][0]["volumeMounts"]}
        assert mounts["extra-lib"]["mountPath"] == "/opt/lib"
        assert mounts["extra-lib"]["readOnly"] is True


class TestHelmValues:
    """Passing the chart's own values through, which is what #6011 asked for."""

    def values(self, overlay) -> K8sOptions:
        return K8sOptions.model_validate({"helmValues": overlay})

    def test_a_component_the_cluster_already_runs_is_skipped(self):
        # The issue's case: a cluster with its own Kueue must not get a second.
        docs = render(
            build_chart_values(
                config(
                    k8s_options=self.values(
                        {"gpustack-operator": {"kueue": {"enabled": False}}}
                    )
                )
            )
        )
        names = [doc["metadata"]["name"] for doc in docs]
        assert not [n for n in names if "kueue" in n]
        # Only what was asked for: the rest of the release is untouched.
        assert "gpustack-operator-worker" in names
        assert [n for n in names if "node-feature-discovery" in n]

    def test_arbitrary_chart_values_reach_the_release(self):
        # The point of a passthrough: this needs no field of its own, and no
        # release of ours to become available.
        overlay = {"worker": {"tolerations": [{"key": "gpu", "operator": "Exists"}]}}
        values = build_chart_values(config(k8s_options=self.values(overlay)))
        assert values["worker"]["tolerations"] == overlay["worker"]["tolerations"]

    def test_merging_is_per_key_not_wholesale(self):
        # Setting one key under `worker` must not drop the derived siblings that
        # tell the workers where to register.
        values = build_chart_values(
            config(k8s_options=self.values({"worker": {"port": 20150}}))
        )
        assert values["worker"]["port"] == 20150
        assert values["worker"]["serverURL"] == "http://gpustack.example.com:30080"
        assert values["worker"]["enabled"] is True

    @pytest.mark.parametrize(
        "overlay",
        [
            {"worker": {"serverURL": "http://elsewhere"}},
            {"server": {"enabled": True}},
            {"image": {"tag": "someone-elses"}},
            {"registrationTokenSecretName": "mine"},
            # `global.hub` prefixes every image the chart composes, so leaving
            # it writable redirects the registry while `image.repository` and
            # `image.tag` still read as issued.
            {"global": {"hub": "someone-elses-registry.io"}},
            # The operator chart resolves `worker.image` over its chart-level
            # image, so protecting only the latter is a fence with a gate.
            {"gpustack-operator": {"worker": {"image": {"tag": "someone-elses"}}}},
            # Adoption is by name. Renaming the sub-chart's objects leaves the
            # pre-chart operator running, unowned, beside a new one — and leaves
            # this manifest's cleanup deleting names that no longer exist.
            {"gpustack-operator": {"nameOverride": "elsewhere"}},
            {"gpustack-operator": {"fullnameOverride": "elsewhere"}},
            {"gpustack-operator": {"namespaceOverride": "elsewhere"}},
            # Not the protected path itself, but something standing where it
            # would have to go. Accepting this stores a row that renders no
            # manifest at all: the merge walks into `image` expecting a mapping
            # and the endpoint fails for that cluster until the row is edited.
            {"image": "someone-elses/gpustack:v1"},
            {"worker": []},
        ],
    )
    def test_server_owned_paths_are_refused(self, overlay):
        # They decide what the deployment is, not how it is configured: pointing
        # the workers at another server, adding a second control plane, or
        # breaking the pairing between the worker image and these templates.
        with pytest.raises(ValueError, match="cannot be set here"):
            self.values(overlay)

    def test_server_owned_paths_survive_a_row_written_around_the_api(self):
        # The validator above guards the API; this guards the merge, for a
        # cluster row edited directly. A release that quietly does not match the
        # cluster it was issued for is worth two defences.
        options = K8sOptions.model_construct(
            helm_values={"worker": {"serverURL": "http://elsewhere"}}
        )
        values = build_chart_values(config(k8s_options=options))
        assert values["worker"]["serverURL"] == "http://gpustack.example.com:30080"


class TestAdoptionByName:
    def test_the_operator_objects_keep_the_names_adoption_relies_on(self):
        """The pre-chart manifest deployed the operator itself, and this release
        takes those objects over rather than duplicating them — which works only
        because the chart names its sub-chart `operator`, putting its objects on
        `gpustack-operator-*`, the names that manifest used. The install script's
        cleanup also deletes by exact name.

        So the names are an invariant, not cosmetics, and the overrides that
        would move them are refused in `SERVER_OWNED_VALUE_PATHS`. This asserts
        the invariant itself, so a chart bump that renames them fails here rather
        than leaving two operators running in the field.
        """
        docs = render(build_chart_values(config()))
        names = {doc["metadata"]["name"] for doc in docs}
        assert "gpustack-operator-worker" in names
        assert {
            doc["metadata"]["namespace"]
            for doc in docs
            if doc["kind"] == "Deployment"
            and doc["metadata"]["name"] == "gpustack-operator-worker"
        } == {"gpustack-system"}


class TestRenderedKinds:
    def test_every_kind_the_bootstrap_renders_is_built_in(self):
        """The install script reads its own render with `kubectl get -f` and
        treats any error as fatal, so that a denied or unreachable read cannot
        pass its ownership guard an empty answer. That is only safe while every
        kind in the render is one the API server always knows: a custom resource
        would fail with "no matches for kind" on every first install, before the
        CRD defining it exists.

        The property is not a property of the chart — rendered with its own
        defaults it emits Istio's `EnvoyFilter`, from higress. It holds because
        `higress-core.enabled` and `server.enabled` are values the server owns
        and always sets false, which is what makes them refused in
        `SERVER_OWNED_VALUE_PATHS` rather than merely defaulted.

        Sub-chart CRDs shipped through a `crds/` directory are a separate case:
        Helm applies those outside the manifest, so they never appear in this
        render and never reach the guard either.
        """
        built_in = {
            "APIService",
            "CSIDriver",
            "ClusterRole",
            "ClusterRoleBinding",
            "ConfigMap",
            "CronJob",
            "CustomResourceDefinition",
            "DaemonSet",
            "Deployment",
            "Endpoints",
            "Ingress",
            "IngressClass",
            "Job",
            "MutatingWebhookConfiguration",
            "Namespace",
            "NetworkPolicy",
            "PersistentVolumeClaim",
            "PodDisruptionBudget",
            "PriorityClass",
            "Role",
            "RoleBinding",
            "RuntimeClass",
            "Secret",
            "Service",
            "ServiceAccount",
            "StatefulSet",
            "StorageClass",
            "ValidatingWebhookConfiguration",
        }
        docs = render(build_chart_values(config(runtimes=[ManufacturerEnum.NVIDIA])))
        rendered = {doc["kind"] for doc in docs}
        assert rendered <= built_in, (
            "these kinds are defined by a CRD, so the script's `kubectl get -f` "
            f"over the render fails before that CRD exists: {sorted(rendered - built_in)}"
        )


class TestProtectedPathsAroundTheApi:
    def test_a_scalar_on_a_protected_path_fails_with_the_path_named(self):
        # `K8sOptions` refuses this, so reaching the merge means the row was
        # written around the API. Carrying on is worse either way: restoring
        # nothing leaves the release without the image it is supposed to carry,
        # and walking into the string raises with no path in the message.
        with pytest.raises(ValueError, match="image.repository"):
            build_chart_values(
                config(
                    k8s_options=K8sOptions.model_construct(
                        helm_values={"image": "someone-elses/gpustack:v1"}
                    )
                )
            )

    def test_the_registry_the_server_derived_survives_an_overlay(self):
        # The same defence one level down: `global.hub` is restored rather than
        # dropped, so an overlay cannot quietly repoint the release's registry.
        values = build_chart_values(
            config(
                image="mirror.example.com/gpustack/gpustack:v1",
                k8s_options=K8sOptions.model_construct(
                    helm_values={"global": {"hub": "someone-elses-registry.io"}}
                ),
            )
        )
        assert values["global"]["hub"] == "mirror.example.com"


class TestNodeSelector:
    """One value carries the cluster's node selector, and the release follows it.

    Helm shares `global` with every sub-chart at every depth and each component
    falls back to it, so this is the only place it has to be written. Which
    workloads honour it is the operator chart's contract and is tested there;
    what belongs here is that the value lands in `global` and that the chart's
    own worker DaemonSets still carry the labels they select on.
    """

    def test_the_selector_lands_in_global_and_nowhere_else(self):
        values = build_chart_values(
            config(k8s_options=K8sOptions(node_selector={"pool": "infra"}))
        )
        assert values["global"]["nodeSelector"] == {"pool": "infra"}
        # Writing the components' own keys as well would put the same value in
        # two trees, still miss everything below them, and mean nothing more: a
        # component-level selector replaces the global one rather than adding to
        # it.
        assert "nodeSelector" not in values["worker"]
        assert "nodeSelector" not in values["gpustack-operator"]["worker"]

    def test_it_confines_the_worker_and_the_operator(self):
        docs = render(
            build_chart_values(
                config(k8s_options=K8sOptions(node_selector={"pool": "infra"}))
            )
        )
        worker = worker_daemonset(docs)["spec"]["template"]["spec"]["nodeSelector"]
        assert worker["pool"] == "infra"

        operator = next(
            doc
            for doc in docs
            if doc["kind"] == "Deployment"
            and doc["metadata"]["name"] == "gpustack-operator-worker"
        )
        assert operator["spec"]["template"]["spec"]["nodeSelector"] == {"pool": "infra"}

    def test_a_vendor_daemonset_keeps_the_label_it_selects_on(self):
        # The vendor DaemonSets merge the base selector with a PCI-presence
        # label. Reaching them through `global` rather than `worker.nodeSelector`
        # must not cost them that label, or they schedule everywhere in the pool.
        docs = render(
            build_chart_values(
                config(
                    runtimes=[ManufacturerEnum.NVIDIA],
                    k8s_options=K8sOptions(node_selector={"pool": "infra"}),
                )
            )
        )
        vendor = next(
            doc
            for doc in docs
            if doc["kind"] == "DaemonSet"
            and doc["metadata"]["name"] == "gpustack-worker-nvidia"
        )
        assert vendor["spec"]["template"]["spec"]["nodeSelector"] == {
            "pool": "infra",
            "feature.node.kubernetes.io/pci-10de.present": "true",
        }


class TestOperatorValues:
    def test_operator_env_is_passed_through(self):
        options = K8sOptions.model_validate(
            {"operator": {"env": {"GPUSTACK_LOG_LEVEL": "debug"}}}
        )
        docs = render(build_chart_values(config(k8s_options=options)))
        deployment = next(
            d
            for d in docs
            if d["kind"] == "Deployment"
            and d["metadata"]["name"] == "gpustack-operator-worker"
        )
        env = {
            e["name"]: e.get("value")
            for e in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        assert env["GPUSTACK_LOG_LEVEL"] == "debug"

    def test_gpu_instance_knobs_are_rendered_as_strings(self):
        # Tri-state: an explicit false has to arrive as the string "false", not
        # be dropped as falsy, or the operator's own default takes over.
        options = K8sOptions.model_validate(
            {"gpuInstanceOptions": {"gpuInstanceTypeDerivedFromNode": False}}
        )
        values = build_chart_values(config(k8s_options=options))
        env = values["gpustack-operator"]["worker"]["env"]
        assert env["GPUSTACK_INSTANCE_TYPE_DERIVED_FROM_NODE"] == "false"

    def test_values_dump_without_yaml_anchors(self):
        # The node selector used to reach two keys as one dict, and PyYAML
        # emitted an anchor plus an alias for it. Helm reads that, but the
        # generated anchor name moves with key order — and the revision the
        # in-cluster Job compares against is a hash of this dump, so it would
        # present as a configuration change. It has one home now; this keeps the
        # invariant rather than the arrangement that broke it.
        import yaml as yaml_module

        options = K8sOptions(node_selector={"disktype": "ssd"})
        dumped = yaml_module.safe_dump(
            build_chart_values(config(k8s_options=options)), sort_keys=True
        )
        assert "&id" not in dumped and "*id" not in dumped, dumped

    def test_values_are_json_serialisable(self):
        # They travel to the cluster as a ConfigMap, so anything that cannot be
        # dumped is a manifest that fails at apply time rather than here.
        json.dumps(build_chart_values(config()))
