from kubernetes_asyncio import client


def get_k8s_client_config(
    server_api_port: int, cluster_id: int
) -> client.Configuration:
    api_config = client.Configuration(
        host=f"http://localhost:{server_api_port}/v2/clusters/{cluster_id}/proxy",
    )
    api_config.verify_ssl = False
    return api_config


def get_k8s_client(
    server_api_port: int, cluster_id: int
) -> client.api_client.ApiClient:
    api_config = get_k8s_client_config(server_api_port, cluster_id)
    api = client.api_client.ApiClient(configuration=api_config)
    api.user_agent = "gpustack/gpustack"
    return api
