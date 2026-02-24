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
3. `Select the GPU vendor`. Tested vendors include `Nvidia`, `AMD`, and `Ascend`. Experimental vendors include `Hygon`, `Moore Threads`, `Iluvatar`, `Cambricon`, and `Metax`. Click `Next` after selecting a vendor.
4. `Check environment`. A shell command is provided to verify that your environment is ready to add a worker. Copy the script and run it in your environment. Click `Next` after the script returns OK.
5. `Run command` to apply the worker manifests. Copy the bash script and run it in an environment where `kubectl` is installed and `kubeconfig` is configured.

The kubernetes can be registerred after the cluster is created.

1. Go to `Clusters` page.
2. Find the cluster which you want to register the Kubernetes cluster.
3. Click the ellipsis button in the operations column, then select `Register Cluster`.
4. Select the options to register cluster. Following the same steps as abovve, from `Select the GPU vendor` to `Run command`.

### Creating DigitalOcean Cluster

1. In the `Basic Configuration` step, the `Name` field is required and `Description` is optional. Create or select a Cloud Credential for communicating with the DigitalOcean API. Select a Region that supports GPU Droplets. You must also configure the `GPUStack Server URL`, which will be accessible from the newly created DigitalOcean Droplets.
2. Click `Next`.
3. Adding one or more `Worker Pools`. For each pool, `Name`, `Instance Type`, `OS Image`, `Replicas`, `Batch Size`, `Labels` and `Volumes` can be specified.
4. Click `Save` after the worker pools are configured.

The worker poll can be added after the cluster is created.

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
system_default_container_registry: "docker.io"
image_name_override: "gpustack/gpustack:main"
image_repo: "gpustack/gpustack"
# ========= service & networking ===========
service_discovery_name: "worker"
namespace: "gpustack-system"
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
enable_hf_xet: false
```

The above YAML lists all currently supported options for the `Worker Configuration YAML`. For the meaning of each option, refer to the full GPUStack [config file documentation](../cli-reference/start.md#config-file).
