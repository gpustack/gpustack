from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from kubernetes import client, config

from gpustack.worker.kubernetes_cpu_budget import KubernetesCPUCollector


def _container(name, cpu, *, sidecar=False):
    return client.V1Container(
        name=name,
        resources=client.V1ResourceRequirements(requests={"cpu": cpu}),
        restart_policy="Always" if sidecar else None,
    )


def _pod(containers, *, node="worker-a", phase="Running", **options):
    return client.V1Pod(
        spec=client.V1PodSpec(containers=containers, node_name=node, **options),
        status=client.V1PodStatus(phase=phase),
    )


def _node_api(allocatable):
    api = Mock(spec=client.CoreV1Api)
    api.read_node.return_value = client.V1Node(
        status=client.V1NodeStatus(allocatable={"cpu": allocatable})
    )
    return api


def _page(pods, cursor=None):
    return client.V1PodList(items=pods, metadata=client.V1ListMeta(_continue=cursor))


def _collect(api):
    return KubernetesCPUCollector(lambda: nullcontext(api), "worker-a").collect()


def test_cpu_budget_counts_bound_pod_requests_from_every_page():
    api = _node_api("8")
    api.list_pod_for_all_namespaces.side_effect = [
        _page(
            [
                _pod([_container("requests", "2575m")]),
                _pod([_container("completed", "8")], phase="Succeeded"),
            ],
            cursor="next",
        ),
        _page(
            [
                _pod([_container("other-namespace", "4")], phase="Pending"),
                _pod([_container("other-node", "8")], node="worker-b"),
                _pod([_container("rejected", "2")], phase="Failed"),
            ]
        ),
    ]

    result = _collect(api)

    assert result.available_millicores == 1425
    assert result.expires_at is not None
    calls = api.list_pod_for_all_namespaces.call_args_list
    assert [call.kwargs["_continue"] for call in calls] == [None, "next"]
    assert {call.kwargs["field_selector"] for call in calls} == {
        "spec.nodeName=worker-a"
    }


@pytest.mark.parametrize(("pod_level", "expected"), [(None, 6000), ("2.25", 5650)])
def test_cpu_budget_counts_init_sidecars_pod_requests_and_overhead(pod_level, expected):
    api = _node_api("8")
    pod = _pod(
        [_container("app", "1300m")],
        init_containers=[
            _container("sidecar", "250m", sidecar=True),
            _container("init", "1650m"),
            _container("late-sidecar", "100m", sidecar=True),
        ],
        resources=(
            client.V1ResourceRequirements(requests={"cpu": pod_level})
            if pod_level is not None
            else None
        ),
        overhead={"cpu": "100m"},
    )
    api.list_pod_for_all_namespaces.return_value = _page([pod])

    assert _collect(api).available_millicores == expected


@pytest.mark.parametrize("allocated", [None, {"cpu": "2"}])
def test_cpu_budget_accounts_for_optional_container_allocations(allocated):
    api = _node_api("8")
    pod = _pod([_container("app", "1")])
    pod.status.container_statuses = [
        client.V1ContainerStatus(
            image="test",
            image_id="test",
            name="app",
            ready=True,
            restart_count=0,
            allocated_resources=allocated,
        )
    ]
    api.list_pod_for_all_namespaces.return_value = _page([pod])

    assert _collect(api).available_millicores == (6000 if allocated else 7000)


@pytest.mark.parametrize("forbidden", [False, True])
def test_unavailable_budget_is_distinct_from_zero(forbidden):
    api = _node_api("2")
    api.list_pod_for_all_namespaces.return_value = _page(
        [_pod([_container("app", "2")])]
    )
    if forbidden:
        api.list_pod_for_all_namespaces.side_effect = client.ApiException(status=403)

    result = _collect(api)

    assert result.available_millicores == (None if forbidden else 0)
    assert (result.expires_at is None) is forbidden


@pytest.mark.parametrize(
    ("endpoint", "verify"),
    [("http://kube", True), ("https://kube", False), (None, True), ("https:", True)],
)
def test_opted_in_collector_requires_a_verified_https_endpoint(
    monkeypatch, endpoint, verify
):
    def load(*, client_configuration, **_):
        client_configuration.host = endpoint
        client_configuration.verify_ssl = verify

    monkeypatch.setattr(config, "load_kube_config", load)
    api_client = Mock()
    monkeypatch.setattr(client, "ApiClient", api_client)

    collector = KubernetesCPUCollector.from_kubeconfig("/test/kubeconfig", "worker-a")

    assert collector.collect().available_millicores is None
    api_client.assert_not_called()


def test_opted_in_collector_without_a_node_name_does_not_open_kubeconfig(monkeypatch):
    load = Mock()
    monkeypatch.setattr(config, "load_kube_config", load)

    collector = KubernetesCPUCollector.from_kubeconfig("/test/kubeconfig", None)

    assert collector.collect().available_millicores is None
    load.assert_not_called()


@pytest.mark.parametrize("content", [None, "clusters: [", "- not a mapping"])
def test_unusable_kubeconfig_does_not_prevent_collector_startup(tmp_path, content):
    kubeconfig = tmp_path / "kubeconfig.yml"
    if content is not None:
        kubeconfig.write_text(content, encoding="utf-8")

    collector = KubernetesCPUCollector.from_kubeconfig(str(kubeconfig), "worker-a")

    budget = collector.collect()
    assert budget.available_millicores is None
    assert budget.expires_at is None


def test_cpu_budget_recovers_from_missing_kubeconfig_and_reloads_token(
    monkeypatch, tmp_path
):
    api = _node_api("8")
    api.list_pod_for_all_namespaces.return_value = _page([])
    observed = []

    def open_core(api_client):
        observed.append(
            api_client.configuration.get_api_key_with_prefix("authorization")
        )
        return api

    monkeypatch.setattr(client, "CoreV1Api", open_core)
    token = tmp_path / "token"
    token.write_text("first-test-token", encoding="utf-8")
    kubeconfig = tmp_path / "kubeconfig.yml"
    collector = KubernetesCPUCollector.from_kubeconfig(str(kubeconfig), "worker-a")

    assert collector.collect().available_millicores is None

    kubeconfig.write_text(
        """apiVersion: v1
kind: Config
clusters:
  - name: test
    cluster: {server: 'https://kube.example.invalid'}
users:
  - name: reader
    user: {tokenFile: token}
contexts:
  - name: test
    context: {cluster: test, user: reader}
current-context: test
""",
        encoding="utf-8",
    )
    assert collector.collect().available_millicores == 8000
    token.write_text("second-test-token", encoding="utf-8")
    assert collector.collect().available_millicores == 8000
    assert observed == ["Bearer first-test-token", "Bearer second-test-token"]
