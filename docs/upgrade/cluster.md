# Upgrade a Cluster Deployment

GPUStack organizes worker nodes into [clusters](../user-guide/cluster-management.md). This page covers upgrading the worker deployment of a cluster after the GPUStack server has been upgraded.

!!! note

    1. Upgrade the GPUStack **server** first (see [Upgrade via Docker](docker.md) or [Upgrade via Helm](helm.md)), then upgrade the cluster deployments.

    2. Upgrade a worker to the **same version** as the server. Do not mix a `main`(dev) or release candidate(rc) worker image with a stable server.

The upgrade procedure mirrors how the cluster was originally deployed (see [Cluster Management](../user-guide/cluster-management.md)): re-run the deployment for a Docker cluster, or re-apply the manifests for a Kubernetes cluster, using the new image.

## Docker Cluster

Workers in a Docker cluster run as containers started by the `docker run` command from the cluster's `Add Worker` step. Upgrade each worker by pulling the new image, removing the old container, and recreating it with the **same arguments and volumes** as before:

```bash
docker pull gpustack/gpustack:vx.y.z   # match the server version

docker stop gpustack-worker
docker rm gpustack-worker

docker run -d --name gpustack-worker \
  ... \
  gpustack/gpustack:vx.y.z \
  --server-url http://your_gpustack_server_url \
  --token your_cluster_registration_token \
  --advertise-address your_worker_ip
```

!!! tip

    You can obtain an up-to-date `docker run` command for the target version from the GPUStack UI: on the `Clusters` page, open the cluster's `Add Worker` step and copy the generated `Run command`.

### Enable Tracing on Existing Docker Workers

If an existing Worker was started without a trace endpoint, add `-e GPUSTACK_TRACE_OTLP_ENDPOINT=<worker-reachable-host>:<port>` when recreating its container. Keep its existing Worker name and data volume to preserve its identity. For built-in Tempo with a Docker bridge network, publish the Server's ingest port on an interface reachable from the Workers' trusted network. With default ports, add `-p <private-server-ip>:4317:4317` to the Server container's Docker options. If Workers use a different address or published port, set `GPUSTACK_TRACE_ADVERTISE_ENDPOINT` on the Server so newly generated Worker commands use that address.

## Kubernetes Cluster

Workers in a Kubernetes cluster are deployed as DaemonSets, managed by the GPUStack Operator, from the manifests applied during cluster registration. To upgrade, re-apply the manifests generated for the new version:

1. Go to the `Clusters` page and find the Kubernetes cluster you want to upgrade.
2. Click the ellipsis button in the operations column, then select `Register Cluster`.
3. Follow the steps to the `Run command` step and copy the generated command.
4. Run the command in an environment where `kubectl` is installed and `kubeconfig` is configured for the target cluster. This re-applies the updated manifests, and the Operator rolls the worker DaemonSets to the new version.

Once the worker pods reach `Ready`, the cluster is running the new version.

For a Worker-only Helm release, reapply the generated registration configuration or set `tracing.endpoint` to the receiver address that the Workers and inference containers can reach. Configure log export separately through `logging.endpoint`. See [Configure Helm](../user-guide/observability.md#configure-helm).

## Cloud Provider Cluster

For cloud-provisioned clusters (e.g. DigitalOcean), each worker node is bootstrapped by cloud-init, which writes the worker config to `/var/lib/gpustack/config.yaml` and starts a `gpustack-worker` container via `/opt/gpustack-run-worker.sh`. To upgrade an existing node, SSH into it and recreate the container with the new image, reusing the same config and volumes:

```bash
# SSH into the worker node, then:
NEW_IMAGE=gpustack/gpustack:vx.y.z   # match the server version

docker pull "$NEW_IMAGE"

docker rm -f gpustack-worker

docker run -d --name gpustack-worker \
  -e "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=gpustack-worker" \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  -v /var/lib/gpustack:/var/lib/gpustack \
  -v /var/run/docker.sock:/var/run/docker.sock \
  "$NEW_IMAGE" \
  --config-file=/var/lib/gpustack/config.yaml
```

!!! note

    So that **newly provisioned** nodes (e.g. from scaling up a worker pool) come up on the new version too, also update the cluster's image configuration in the UI. See [Cluster Management](../user-guide/cluster-management.md#operating-worker-pools).
