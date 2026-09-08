"""The registration manifest, and the mechanism that makes re-applying it work.

`kubectl apply` neither prunes nor re-runs a Completed Job. So the manifest has
to name a Job that does not exist yet — every time — and the question of whether
that Job has anything to do is answered in the cluster, not here. These tests pin
both halves: the name is unique per rendering, and the revision that decides the
answer reaches both the Job and the release.
"""

import pathlib
from typing import Dict

import pytest
import yaml

from gpustack.k8s import bootstrap
from gpustack.k8s.bootstrap import BOOTSTRAP_NAME, RELEASE_NAME, render_bootstrap
from gpustack.k8s.manifest_template import TemplateConfig
from gpustack.schemas.clusters import ImageCredential, K8sOptions
from gpustack_runtime.detector import ManufacturerEnum


def config(**kwargs) -> TemplateConfig:
    defaults = {
        "token": "tok",
        "server_url": "http://gpustack.example.com:30080",
        "image": "gpustack/gpustack:dev",
        "env": {"GPUSTACK_TOKEN": "tok"},
        "args": [],
        "cluster_owner_principal_identifier": "1",
    }
    defaults.update(kwargs)
    return TemplateConfig(**defaults)


def objects(**kwargs) -> Dict[str, dict]:
    docs = [d for d in yaml.safe_load_all(render_bootstrap(config(**kwargs))) if d]
    return {f"{d['kind']}/{d['metadata']['name']}": d for d in docs}


def named(rendered: Dict[str, dict], kind: str) -> str:
    for doc in rendered.values():
        if doc["kind"] == kind:
            return doc["metadata"]["name"]
    pytest.fail(f"no {kind} was rendered")


def job_env(rendered: Dict[str, dict]) -> Dict[str, str]:
    job = rendered[f"Job/{named(rendered, 'Job')}"]
    container = job["spec"]["template"]["spec"]["containers"][0]
    return {e["name"]: e["value"] for e in container["env"]}


def chart_values(rendered: Dict[str, dict]) -> dict:
    return yaml.safe_load(
        rendered[f"ConfigMap/{BOOTSTRAP_NAME}"]["data"]["values.yaml"]
    )


class TestJobNaming:
    def test_every_rendering_names_a_new_job(self):
        # The only way an apply can make something happen. A name derived from
        # the configuration would skip a revert to an earlier one, whose Job is
        # already Completed: A -> B -> A would leave the cluster on B.
        names = {named(objects(), "Job") for _ in range(5)}
        assert len(names) == 5, names

    def test_nothing_else_is_named_per_rendering(self):
        # `kubectl apply` does not prune and only a Job has a TTL, so a name
        # that moved would leave an orphan behind on every fetch.
        one, two = objects(), objects()
        for kind in ("ConfigMap", "ServiceAccount", "ClusterRoleBinding"):
            assert named(one, kind) == named(two, kind) == BOOTSTRAP_NAME

    def test_the_job_reads_the_configmap_rather_than_a_baked_copy(self):
        rendered = objects()
        pod = rendered[f"Job/{named(rendered, 'Job')}"]["spec"]["template"]["spec"]
        assert pod["volumes"][0]["configMap"]["name"] == BOOTSTRAP_NAME
        env = job_env(rendered)
        assert env["VALUES_FILE"].startswith("/bootstrap/")
        assert env["RELEASE"] == RELEASE_NAME


class TestAppliedRevision:
    def test_the_job_and_the_values_agree(self):
        # The Job compares one against the other in the cluster; if they were
        # computed differently it would either reinstall forever or never.
        rendered = objects()
        assert (
            job_env(rendered)["DESIRED_REVISION"]
            == chart_values(rendered)["appliedRevision"]
        )

    def test_the_revision_tracks_the_configuration(self):
        one = chart_values(objects(runtimes=[ManufacturerEnum.NVIDIA]))
        two = chart_values(objects(runtimes=[ManufacturerEnum.ASCEND]))
        assert one["appliedRevision"] != two["appliedRevision"]

    def test_the_revision_is_stable_for_the_same_configuration(self):
        # Unlike the Job name. A revision that moved per rendering would make
        # every apply a real `helm upgrade` and fill `helm history` with no-ops.
        one = chart_values(objects(k8s_options=K8sOptions(node_selector={"a": "b"})))
        two = chart_values(objects(k8s_options=K8sOptions(node_selector={"a": "b"})))
        assert one["appliedRevision"] == two["appliedRevision"]

    def test_a_reverted_configuration_returns_to_its_revision(self):
        # A -> B -> A. The Job names differ, so all three apply; the revision
        # returning to A's value is what tells the third Job it has work to do,
        # because the release still records B's.
        a_first = chart_values(objects(runtimes=[ManufacturerEnum.NVIDIA]))
        b = chart_values(objects(runtimes=[ManufacturerEnum.ASCEND]))
        a_again = chart_values(objects(runtimes=[ManufacturerEnum.NVIDIA]))
        assert a_first["appliedRevision"] == a_again["appliedRevision"]
        assert b["appliedRevision"] != a_first["appliedRevision"]

    def test_a_new_chart_is_a_new_revision(self, monkeypatch):
        # The one input that changes without the cluster changing: a server
        # upgrade shipping a bumped operator pin serves different chart bytes at
        # the same, version-free URL. Derived from the URL, the revision would
        # match and the Job would report the cluster already up to date.
        before = chart_values(objects())["appliedRevision"]
        monkeypatch.setattr(bootstrap, "chart_digest", lambda: "a-different-chart")
        assert chart_values(objects())["appliedRevision"] != before


class TestScalarTypes:
    """Every interpolated scalar is quoted, because YAML types a scalar by its
    shape and kubectl rejects a manifest whose string field parsed as something
    else. These are the values whose shape is not ours to choose."""

    @pytest.mark.parametrize("namespace", ["123", "no", "y", "true", "0755"])
    def test_a_namespace_that_looks_like_something_else_stays_a_string(self, namespace):
        # All legal DNS-1123 label names, and every one of them a bool, an int or
        # an octal to a YAML parser.
        rendered = objects(k8s_options=K8sOptions(namespace=namespace))
        for doc in rendered.values():
            assert doc["metadata"].get("namespace", namespace) == namespace
        assert job_env(rendered)["NAMESPACE"] == namespace

    def test_an_all_digit_revision_stays_a_string(self, monkeypatch):
        # 16 hex digits are all decimal about once in four thousand, so this is
        # rare rather than impossible — and it would take down every apply for
        # that one configuration.
        monkeypatch.setattr(bootstrap, "applied_revision", lambda *a, **k: "1" * 16)
        assert job_env(objects())["DESIRED_REVISION"] == "1" * 16


class TestOwnership:
    def test_the_token_secret_survives_a_helm_prune(self):
        # A cluster whose chart used to create this Secret has it in the previous
        # release manifest and not in the new one, which is what Helm prunes.
        # Losing it leaves running workers alive and every new pod unable to
        # start: it is mounted with `optional: false`.
        secret = objects()["Secret/registration-token"]
        assert secret["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"

    def test_both_namespaces_are_created(self):
        rendered = objects()
        namespaces = {
            doc["metadata"]["name"]
            for doc in rendered.values()
            if doc["kind"] == "Namespace"
        }
        assert namespaces == {"gpustack-system", "gpustack-1"}

    def test_pull_credentials_reach_the_namespace_the_job_pulls_from(self):
        # A Job's own imagePullSecrets resolve in the namespace it runs in, so
        # this is what a private registry needs to get the operator image the
        # Job runs — separately from the credentials the release itself uses.
        docs = [
            doc
            for doc in yaml.safe_load_all(
                render_bootstrap(
                    config(
                        k8s_options=K8sOptions(
                            image_credentials=[
                                ImageCredential(
                                    registry="r.io", username="u", password="p"
                                )
                            ]
                        )
                    )
                )
            )
            if doc
        ]
        job = next(doc for doc in docs if doc["kind"] == "Job")
        referenced = {
            entry["name"]
            for entry in job["spec"]["template"]["spec"]["imagePullSecrets"]
        }
        assert referenced, "the Job pulls with the cluster's credentials"

        for name in referenced:
            namespaces = {
                doc["metadata"]["namespace"]
                for doc in docs
                if doc["kind"] == "Secret" and doc["metadata"]["name"] == name
            }
            assert (
                job["metadata"]["namespace"] in namespaces
            ), f"{name} is missing from the namespace the Job pulls from"

    def test_the_job_is_not_confined_by_the_cluster_node_selector(self):
        # It installs the release, so it runs before anything the release
        # deploys exists. A selector naming a label the release produces — an
        # NFD one being the obvious way to write "only GPU nodes" — would leave
        # this Job Pending on the labeller it is about to install.
        rendered = objects(
            k8s_options=K8sOptions(
                node_selector={"feature.node.kubernetes.io/pci-10de.present": "true"}
            )
        )
        job = rendered[f"Job/{named(rendered, 'Job')}"]
        assert "nodeSelector" not in job["spec"]["template"]["spec"]

    def test_the_grant_is_shared_and_therefore_not_revoked(self):
        # cluster-admin, and one binding per release rather than per rendering —
        # so a Job that deleted it on the way out could strip a concurrent newer
        # Job of the permissions it is installing with, and nothing would
        # recreate it. Revoking would not be worth that even if it were free:
        # the release namespace holds three permanent cluster-admin
        # ServiceAccounts the chart creates (the operator, its device managers,
        # its migration hook) alongside a worker that may create Pods there, so
        # the reachable privilege there is the chart's to narrow, not this Job's.
        rendered = objects()
        binding = rendered[f"ClusterRoleBinding/{BOOTSTRAP_NAME}"]
        assert binding["roleRef"]["name"] == "cluster-admin"
        assert binding["metadata"]["name"] == BOOTSTRAP_NAME

        script = (
            pathlib.Path(__file__).resolve().parents[2]
            / "gpustack"
            / "k8s"
            / "bootstrap.sh"
        ).read_text()
        assert "clusterrolebinding/${BINDING}" not in script
