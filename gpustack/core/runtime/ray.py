import os
import subprocess
import sys
from typing import Dict
import ray
from ray import serve
import requests

from .inference_engines import TorchInferenceService
from ...core.config import configs
from ...schemas.models import Model
from ...schemas.nodes import Node, ResourceSummary
from ...utils import normalize_route_path
from ...logging import logger


EMBEDDED_REDIS_ADDRESS = "127.0.0.1:6379"
DEFAULT_GCS_PORT = "6380"


class RayRuntime:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RayRuntime, cls).__new__(cls)

            if not configs.address:
                logger.info(
                    f"Cluster management address: {configs.node_ip_address}:{DEFAULT_GCS_PORT}"
                )

                os.environ["RAY_REDIS_ADDRESS"] = EMBEDDED_REDIS_ADDRESS
                os.environ["RAY_GCS_SERVER_PORT"] = DEFAULT_GCS_PORT
                ray.init(_node_ip_address=configs.node_ip_address)
        return cls._instance

    @classmethod
    def start_worker(cls, **kwargs):
        # TODO check if this can be done by ray.init
        address = kwargs.get("address")
        node_ip_address = kwargs.get("node_ip_address")
        command = [
            "ray",
            "start",
            "--node-ip-address",
            node_ip_address,
            "--address",
            address,
        ]
        os.environ["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "1"
        try:
            process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
            process.communicate()
            if process.returncode == 0:
                logger.info("Started worker node successfully!")
            else:
                logger.error("Error starting worker node.")
        except Exception as e:
            logger.error(f"Error starting worker node: {e}")

    def get_nodes(self) -> list[Node]:
        ray_nodes = ray.nodes()
        alive_addresses = {
            node["NodeManagerAddress"] for node in ray_nodes if node["alive"]
        }
        filtered_nodes = [
            node
            for node in ray_nodes
            if not (
                node["alive"] is False and node["NodeManagerAddress"] in alive_addresses
            )
        ]
        node_dict: Dict[str, Node] = {}

        resource_filters: list[str] = ["memory", "cpu"]

        for ray_node in filtered_nodes:
            node = Node(
                id=ray_node["NodeID"],
                name=ray_node["NodeName"],
                hostname=ray_node["NodeManagerHostname"],
                address=ray_node["NodeManagerAddress"],
                alive=ray_node["Alive"],
                labels=ray_node["Labels"],
            )
            for resource_name, total in ray_node["Resources"].items():
                if resource_name.lower() in resource_filters:
                    node.resources[resource_name.lower()] = ResourceSummary(total=total)
            node_dict[ray_node["NodeManagerHostname"]] = node

        try:
            # get resource usage from ray dashboard APIs
            url = "http://127.0.0.1:8265/nodes?view=summary"
            response = requests.get(url)
            data = response.json()

            for node_info in data["data"]["summary"]:
                hostname = node_info["hostname"]
                if hostname not in node_dict:
                    continue
                node = node_dict[hostname]
                node.resources["memory"].free = node_info["mem"][1]
                node.resources["memory"].percent = node_info["mem"][2]
                node.resources["memory"].used = node_info["mem"][3]

                node.resources["cpu"].percent = node_info["cpu"]
                node.resources["cpu"].used = (
                    node_info["cpus"][0] * node_info["cpu"] / 100
                )
                node.resources["cpu"].free = (
                    node_info["cpus"][0] - node.resources["cpu"].used
                )

                # TODO add GPU usage

                node_dict[hostname] = node
        except Exception as e:
            logger.error(f"Failed to get resource usage from dashboard: {e}")

        nodes = list(node_dict.values())
        return nodes

    def serve_model(self, model: Model, **kwargs):
        serve.run(
            TorchInferenceService.bind(model),
            name=model.name,
            route_prefix=normalize_route_path(model.name),
            **kwargs,
        )

    def shutdown(self):
        ray.shutdown()
