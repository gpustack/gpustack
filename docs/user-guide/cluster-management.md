# Cluster Management

GPUStack supports cluster-based worker management and provides multiple cluster types. You can provision a cluster through a `Cloud Provider` such as `DigitalOcean`, or create a self-hosted cluster and add workers using `Docker` run commands. Alternatively, you can register all nodes in a self-hosted `Kubernetes` cluster as GPUStack workers.

## Create Cluster

1. Go to the `Clusters` page.
2. Click the `Add Cluster` button.
3. Select a cluster provider. There are `Docker` and `Kubernetes` for the `Self-Host` provider and `DigitalOcean` for the `Cloud Provider`.
4. Depending on the provider, different options need to be set in the `Base Configuration` and `Add Worker` steps.
5. The `Advanced` cluster settings in the Base Configuration allow you to pre-configure the worker options using the `Worker Configuration YAML`.

### Create Docker Cluster

1. In the `Basic Configuration` step, the `Name` field is required and `Description` is optional.
2. Click `Save`.
3. In the `Add Worker` step, some options and validations are needed before adding a worker via the `docker run` command.
4. `Select the GPU vendor`. Tested vendors include `Nvidia`, `AMD`, and `Ascend`. Experimental vendors include `Hygon`, `Moore Threads`, `Iluvatar`, `Cambricon`, and `Metax`. Click `Next` after selecting a vendor.
5. `Check Environment`. A shell command is provided to verify your environment is ready to add a worker. Copy the script and run it in your environment. Click `Next` after the script returns OK.
6. `Specify arguments` for the worker to be added. Provide the following arguments and click `Next`:
   - `Specify the worker IP`, or let the worker `Auto-detect the Worker IP`. Make sure the worker IP is accessible from the server.
   - Specify a `Additional Volume Mount` for the worker container. The mount path can be used to reuse existing model files.

7. `Run command` to create and start the worker container. Copy the bash script and run it in your environment.

The worker also can be added after the cluster is created.

1. Go to `Clusters` page.
2. Find the cluster which you want to add workers.
3. Click the ellipsis button in the operations column, then select `Add Worker`.
4. Select the options to add worker. Following the same steps as above, from `Select the GPU vendor` to `Run command`.

### Register Kubernetes Cluster

1. In the `Basic Configuration` step, the `Name` field is required and `Description` is optional.
2. Click `Save`.
3. `Select the GPU vendor`. Tested vendors include `Nvidia`, `AMD`, and `Ascend`. Experimental vendors include `Hygon`, `Moore Threads`, `Iluvatar`, `Cambricon`, and `Metax`. For `Kubernetes` clusters you can select multiple GPU vendors, or select none for CPU-only clusters — one worker DaemonSet is rendered per selected vendor, and each DaemonSet's `nodeSelector` is derived from the vendor's PCI-presence label at manifest time. Click `Next` after selecting the vendor(s).
4. `Check environment`. A shell command is provided to verify that your environment is ready to add a worker. Copy the script and run it in your environment. Click `Next` after the script returns OK.
5. `Run command` to apply the worker manifests. Copy the bash script and run it in an environment where `kubectl` is installed and `kubeconfig` is configured.

The kubernetes can be registered after the cluster is created.

1. Go to `Clusters` page.
2. Find the cluster which you want to register the Kubernetes cluster.
3. Click the ellipsis button in the operations column, then select `Register Cluster`.
4. Select the options to register the cluster. Follow the same steps as above, from `Select the GPU vendor` to `Run command`.

#### Kubernetes Cluster Options

When creating or editing a `Kubernetes` cluster, the following options are available in the `Basic Configuration` step in addition to `Name` and `Description`:

- `Cluster Type` (required) — choose how the cluster is used:
    - `Model Service` — for LLM inference and API serving, e.g. exposing model APIs and token-based services.
    - `GPU Service` — for on-demand GPU compute, e.g. interactive development, training jobs, or custom environments.

The `Advanced` settings expose the following Kubernetes deployment options:

- `Namespace` — the Kubernetes namespace the cluster's manifests render into. Leave empty to use `gpustack-system`.
- `Volume Mounts` — extra volumes mounted into every worker pod. For each mount, specify a `Volume Name`, `Container Path`, and `Read Only` flag, then choose a `Source Type`:
    - `Host Path` — a path on the node, with a `Path Type` (e.g. `Directory`, `Directory (create if not exists)`, `File`, `Socket`, `Character Device`, `Block Device`).
    - `Persistent Volume Claim (PVC)` — an existing `PVC Name`, optionally read-only.
    - `ConfigMap` — a `ConfigMap Name`, optionally marked optional.
- `Image Credentials` — image pull secrets used to pull GPUStack images from a private registry. For each entry, specify a `Registry`, `Username`, and `Password`.
- `Node Selector` — a pod `nodeSelector` applied to every worker DaemonSet; only nodes whose labels match are eligible to run the worker.
- `Default Container Registry` — the default registry used to resolve GPUStack images for this cluster. Falls back to the server default when unset (placeholder `docker.io`).
- `Operator Image` — override for the GPUStack Operator container image. Leave empty to use the server default.
- `GPU Service Static Access Address` — only shown when `Cluster Type` is `GPU Service`. The static address the operator uses to access GPU instances in this cluster (e.g. a LoadBalancer VIP). Optional.
- `Worker Configuration YAML` — see [Worker Configuration YAML](#worker-configuration-yaml) below.

### Creating DigitalOcean Cluster

1. In the `Basic Configuration` step, the `Name` field is required and `Description` is optional. Create or select a Cloud Credential for communicating with the DigitalOcean API. Select a Region that supports GPU Droplets. You must also configure the `GPUStack Server URL`, which will be accessible from the newly created DigitalOcean Droplets.
2. Click `Next`.
3. Adding one or more `Worker Pools`. For each pool, `Name`, `Instance Type`, `OS Image`, `Replicas`, `Batch Size`, `Labels` and `Volumes` can be specified.
4. Click `Save` after the worker pools are configured.

Additional worker pools can be added after the cluster is created.

1. Go to `Clusters` page.
2. Find the `DigitalOcean` cluster which you want to add worker pool.
3. Click the ellipsis button in the operations column, then select `Add Worker Pool`
4. Adding new worker pool with options from Step 3 above.

### Operating Worker Pools

You can manage worker pools for DigitalOcean clusters on the `Clusters` page:

1. Go to the `Clusters` page.
2. Find the DigitalOcean cluster you want to manage and expand it to view its worker pools.
3. To edit the replica count for a worker, modify it directly in the worker column.
4. To edit a worker pool, click the `Edit` button and update the `Name`, `Replica`, `Batch Size`, and `Labels` as needed.
5. To delete a worker pool, click the ellipsis button in the operations column for the worker pool, then select `Delete`.

## Update Cluster

1. Go to the `Clusters` page.
2. Find the cluster which you want to edit.
3. Click the `Edit` button.
4. Update the `Name`, `Description` and `Worker Configuration YAML` as needed.
5. Click the `Save` button.

## Delete Cluster

1. Go to the `Clusters` page.
2. Find the cluster which you want to delete.
3. Click the ellipsis button in the operations column, then select `Delete`.
4. Confirm the deletion.
5. You cannot delete a cluster if there are any models or workers still present in it.

## Worker Configuration YAML

When creating or updating a cluster, you can predefine the worker configuration using the following example YAML:

```yaml
# ========= log level & tools ===========
debug: false
tools_download_base_url: https://mirror.your_company.com
# ========= directories ===========
pipx_path: "/usr/local/bin/pipx"
cache_dir: "/var/lib/gpustack/cache"
log_dir: "/var/lib/gpustack/log"
bin_dir: "/var/lib/gpustack/bin"
# ========= container & image ===========
image_name_override: "gpustack/gpustack:main"
image_repo: "gpustack/gpustack"
# ========= service & networking ===========
worker_ifname: en0
worker_port: 10150
worker_metrics_port: 10151
disable_worker_metrics: false
service_port_range: "40000-40063"
ray_port_range: "41000-41999"
proxy_mode: worker
# ========= system reserved resources ===========
system_reserved:
  ram: 0
  vram: 0
# ========= huggingface ===========
huggingface_token: xxxxxx
enable_hf_transfer: false
```

The above YAML lists all currently supported options for the `Worker Configuration YAML`. For the meaning of each option, refer to the full GPUStack [config file documentation](../cli-reference/start.md#config-file). The `proxy_mode` option controls how the server reaches the worker — see [Worker Connection Modes](#worker-connection-modes).

The default container registry is no longer set here for either `Docker` or `Kubernetes` clusters; configure it through the cluster-level `Default Container Registry` option in the `Advanced` settings instead. For `Kubernetes` clusters, the namespace is likewise configured through the `Namespace` option (see [Kubernetes Cluster Options](#kubernetes-cluster-options)).

## Worker Connection Modes

To forward inference requests to the model instances on a worker, the server and the API gateway need a network path back to that worker. The `proxy_mode` worker option — set via the [Worker Configuration YAML](#worker-configuration-yaml) or the `--proxy-mode` flag — selects this path:

- `direct` — The server and gateway connect straight to the worker's advertised address and port. Lowest overhead, but the worker must be directly reachable from the server.
- `worker` — Requests pass through the worker's built-in HTTP reverse proxy, which forwards them to the local inference process. The worker must still be reachable from the server, but only its worker port needs to be exposed.
- `tunnel` — The worker keeps a single **outbound** WebSocket connection to the server, and the server reaches the worker only through that tunnel. Use this when the worker cannot accept inbound connections from the server.

The default is `direct` for the embedded worker inside the server, and `worker` for standalone workers.

### Tunnel Mode (Worker Behind a Firewall or NAT)

In `direct` and `worker` modes the server initiates the connection, so the worker must be reachable from the server. That is not always possible — for example when the worker sits behind a firewall or NAT, on a different network, or in a private subnet that only allows outbound traffic.

`tunnel` mode reverses the direction: the connection is established **one way**, from the worker to the server.

1. The worker opens a persistent outbound WebSocket connection to the server — to the same `--server-url` endpoint it uses to register — authenticated with its token and reconnected automatically if it drops.
2. The server runs an HTTP/HTTPS proxy on its proxy port (default `30079`, set via `--proxy-port`).
3. When the gateway needs to reach a model instance on a tunnel-mode worker, it routes the request to the server's proxy port, which relays it to the worker over the existing tunnel.

Since the worker only ever dials out, no inbound ports have to be opened on the worker side. It only needs to reach the server's API port, and the server's proxy port must be reachable by the gateway.

To enable it, set the worker's `proxy_mode` to `tunnel`:

```yaml
proxy_mode: tunnel
```
