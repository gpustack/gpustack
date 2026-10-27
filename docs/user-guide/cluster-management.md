# Cluster Management

GPUStack supports cluster-based worker management and provides multiple cluster types. You can provision a cluster through a `Cloud Provider` such as `DigitalOcean`, or create a self-hosted cluster and add workers using `Docker` run commands. Alternatively, you can register all nodes in a self-hosted `Kubernetes` cluster as GPUStack workers.

## Create Cluster

1. Go to the `Clusters` page.
2. Click the `Add Cluster` button.
3. Select a cluster provider. There are `Docker` and `Kubernetes` for the `Self-Host` provider, and `DigitalOcean` and `SHUIHUA FUTURE` for the `Cloud Provider`.
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
- `Node Selector` — confines this cluster's GPUStack workloads to nodes whose labels match. It is passed as the chart's `global.nodeSelector`, which each component falls back to, so it reaches the worker DaemonSets and the operator, and the components the operator deploys (Kueue, Node Feature Discovery's control plane, the CSI controllers) from the operator release that honours it. Two kinds of workload are left out on purpose: the ones that have to cover the nodes they serve (Node Feature Discovery's labeller, the CSI node plugins), because confining those would leave every other node unlabelled — including with the labels the workers themselves select on — and the chart's install hooks, which run before the release exists and so cannot wait on a label the release itself applies. A worker DaemonSet keeps its GPU vendor label as well as these; a node has to match both.
- `Default Container Registry` — the default registry used to resolve GPUStack images for this cluster. Falls back to the server default when unset (placeholder `docker.io`).
- `Operator Image` — override for the GPUStack Operator container image. Leave empty to use the server default.
- `GPU Service Static Access Address` — only shown when `Cluster Type` is `GPU Service`. The static address the operator uses to access GPU instances in this cluster (e.g. a LoadBalancer VIP). Optional.
- `Worker Configuration YAML` — see [Worker Configuration YAML](#worker-configuration-yaml) below.

#### Namespaces and Pod Security Admission

GPUStack creates the namespaces it owns — `gpustack-system` (or whatever
`Namespace` is set to) and one `gpustack-<org>` per organization using the
cluster — and labels each of them for
[Pod Security Admission](https://kubernetes.io/docs/concepts/security/pod-security-admission/)
at the `privileged` level:

```yaml
pod-security.kubernetes.io/enforce: privileged
pod-security.kubernetes.io/audit: privileged
pod-security.kubernetes.io/warn: privileged
```

**Why it is needed.** Pod Security Admission is built into Kubernetes and
enforced when a Pod is *created*: a Pod that does not fit the level is rejected
outright rather than left Pending. Model, cache service and benchmark Pods
legitimately need what the `baseline` and `restricted` levels forbid:

| Requirement | What needs it |
| --- | --- |
| Host networking | RDMA binds its GID to a NIC address |
| hostPort | the side channel a disaggregated pair hands its peer |
| Host IPC | CUDA-IPC sharing of KV cache buffers |
| Device mounts | the accelerators themselves |

A prefill/decode deployment needs all four at once, so it is the case that
fails first and hardest without the label.

**Why all three keys.** With `warn` and `audit` left on the cluster default,
every Pod creation still returns a warning and writes an audit annotation.
That is noise in a namespace where the exemption is expected, and it hides the
warnings that matter elsewhere.

**Blast radius.** `privileged` means every Pod in those namespaces is exempt
from Pod Security Admission. That is a reason they are namespaces GPUStack
creates and owns rather than ones shared with your own workloads: nothing else
should be scheduled into them. The labels also do nothing about Kyverno,
Gatekeeper or OPA — those are separate admission webhooks, and a policy of
theirs that forbids host networking or device mounts will still reject these
Pods. If your cluster runs one, allow the GPUStack namespaces there as well.

**Checking what a stricter level would refuse.** Against your own running
workloads, without changing anything:

```bash
kubectl label --dry-run=server --overwrite ns <namespace> \
  pod-security.kubernetes.io/enforce=baseline
```

The API server evaluates the namespace's existing Pods against the level and
warns about each one that would be rejected, naming the controls it violates.

#### Chart Values

Registering a `Kubernetes` cluster installs the GPUStack Helm chart into it, and the options above are turned into values for that chart. `helmValues` reaches the same chart directly, for what those options do not cover: the keys are the chart's own, taken verbatim, so anything the chart or its sub-charts offer is configurable without waiting for an option of its own here.

It has no field in the UI yet, so set it through the API. `PUT` replaces the whole cluster, so read the cluster first and send it back with `helmValues` added under `k8sOptions`:

```bash
# Read the cluster.
curl -sS -H "Authorization: Bearer <api-key>" \
  http://<server>/v2/clusters/<id> > cluster.json

# Add the values. This one keeps the chart from installing the S3 CSI driver,
# for a cluster that already provides its own storage.
jq '.k8sOptions.helmValues = {
      "gpustack-operator": {"csi-driver-s3": {"enabled": false}}
    }' cluster.json > updated.json

# Send it back.
curl -sS -X PUT -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" --data @updated.json \
  http://<server>/v2/clusters/<id>
```

The values are merged over the ones derived from the cluster, per key and depth-first; a list replaces rather than extends, as it does in Helm itself. Keys are documented by the charts: the GPUStack chart's `values.yaml`, and for anything under `gpustack-operator`, the [GPUStack Operator chart](https://github.com/gpustack/gpustack-operator).

Some paths are refused rather than merged: the ones the server derives from this cluster's registration, which decide that the release matches the cluster it was issued for — where its workers report, which image they run, which registry that image comes from, which Secret carries the token. The API names the path it refused, and the `helmValues` field description carries the current list; this page does not repeat it, so the two cannot drift.

!!! note

    Saving the cluster changes nothing in Kubernetes on its own. Re-run `Register Cluster` and apply the manifest it gives you: the in-cluster Job compares what the manifest asks for against what the release was installed from, and upgrades only when they differ.

!!! note

    An upgrade that changes what the release deploys — turning off the CPU worker DaemonSet (`worker.cpuEnabled=false` in `helmValues`, for a cluster whose CPU-only nodes carry the control plane), or dropping a GPU vendor the cluster no longer selects — removes the worker DaemonSets it stops rendering on its own: `kubectl apply` would not (it never deletes an object the manifest omits), so the upgrade Job deletes every `<release>-worker[-<vendor>]` DaemonSet that the release no longer renders, whether or not a Helm revision ever owned it.

    A leftover from before this cleanup existed is removed the same way: apply the current manifest once and the Job sweeps it, even when the configuration has not changed since. Only a cluster that cannot run the Job at all needs the manual form:

    ```bash
    kubectl delete daemonset/gpustack-worker -n gpustack-system
    kubectl delete daemonset/gpustack-worker-<vendor> -n gpustack-system
    ```

    Keep the `Register Cluster` parameters (GPU vendors, options) identical to the last applied manifest when you only mean to upgrade the version — anything else is a configuration change, not an upgrade.

!!! warning

    Kueue and Node Feature Discovery are not optional. The operator derives its scheduling chain from them and waits for their CRDs at startup, so switching one off that the cluster does not already provide leaves the operator unable to start. Switching off one this release installed removes it — Kueue's CRDs come from its chart, and every `Workload` and `ClusterQueue` goes with them.

#### Uninstalling a Kubernetes Cluster

Uninstalling is `helm uninstall` of the release(s) the registration installed, plus the objects Helm never owned. One preparatory step decides whether it finishes cleanly: the operator's tree (Kueue, Node Feature Discovery, the CSI drivers, the device managers) deploys controllers whose custom resources carry finalizers, and uninstalling takes those controllers away — so the resources have to be gone first, or the namespaces they live in (`gpustack-system`, `gpustack-default`) never leave `Terminating`. In order:

1. Remove what the cluster runs from the server first — delete (or migrate) the models deployed to it, then delete the cluster record — so the server stops scheduling onto workers that are about to disappear.
2. Delete the custom resources **while their controllers are still running**, and wait for them to clear. Discover what this cluster's releases own rather than guessing:

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "")
        | . == "gpustack" or . == "gpustack-kueue"
          or . == "gpustack-node-feature-discovery"
          or . == "gpustack-csi-driver-nfs" or . == "gpustack-csi-driver-s3"
          or . == "gpustack-operator-device-manager")
      | .metadata.name'
    ```

    For every CRD that lists, delete all of its objects and wait until none remain:

    ```bash
    kubectl delete <crd-name> --all -A          # e.g. clusterqueues.kueue.x-k8s.io
    kubectl get <crd-name> -A                   # empty before you move on
    ```

3. Uninstall the releases. The main one is `gpustack`; the five `gpustack-*` application releases exist on clusters registered before the chart adopted their objects, and `helm uninstall` on a release that does not exist is an error, so check first:

    ```bash
    for release in gpustack gpustack-kueue gpustack-node-feature-discovery \
                   gpustack-csi-driver-nfs gpustack-csi-driver-s3 \
                   gpustack-operator-device-manager; do
      kubectl get secret -n gpustack-system \
        --selector "owner=helm,name=${release}" >/dev/null 2>&1 || continue
      helm uninstall "${release}" -n gpustack-system --timeout 10m
    done
    ```

    Do not pass `--wait` here. With `--wait`, the uninstall reaches the Node Feature Discovery **post-delete prune hook**, creates the hook's `node-feature-discovery-prune` ServiceAccount and ClusterRole, and then blocks in the deletion wait — the wait outlives its `--timeout` (verified with `helm --debug`: it sits on `waiting for resources to be deleted count=1` for minutes past the timeout, and the ServiceAccount it just created is still there). Killing Helm to escape it is what leaves the prune RBAC behind ownerless, so the hook is the reason step 4 removes those objects by hand. Without `--wait` the uninstall returns promptly, the hook resources are cleaned up with the rest, and the objects are gone all the same — step 4 removes the stragglers Helm never owned.

4. Remove what `helm uninstall` does not. Helm never deletes CRDs, and the operator registers three more kinds of cluster-scoped object that no release owns:

    - **The operator's own CRDs** — `devices`, `instances` and `instancetypes` under `worker.gpustack.ai`. The operator creates them at startup rather than the chart, so they carry no `meta.helm.sh/release-name` annotation and step 2's discovery query does not list them. An `instancetype` object survives with the `gpustack.ai/controlled` finalizer; strip it (see below) before or while deleting the CRDs, or the CRD stays in deletion and the group's API never goes away.
    - **The operator's admission webhooks** — `gpustack-worker-mutation` (Mutating) and `gpustack-worker-validation` (Validating). They forward to the operator's Service and fail closed, so until they are deleted every write to a `worker.gpustack.ai` object — including the finalizer strip below — is rejected with `service "gpustack-operator-worker" not found`.
    - **The operator's APIServices** — `v1.gpustack.ai` and `v1.worker.gpustack.ai`, likewise backed by the operator's Service. A namespace stuck deleting reports `stale GroupVersion discovery` for these groups until they are removed, and never finishes.

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    ```

    The Node Feature Discovery prune RBAC can outlive `helm uninstall` as well — `ClusterRole/node-feature-discovery-prune` and its binding are the post-delete prune hook's resources (see the `--wait` note above), left ownerless when Helm is killed mid-hook. A later registration refuses to install rather than adopt them, so remove them with the rest:

    ```bash
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    Then the cluster-scoped bootstrap RBAC, then the namespaces. `gpustack-system` also holds the bootstrap ConfigMap, ServiceAccount and the registration token Secret — deleting the namespace takes them with it:

    ```bash
    kubectl delete clusterrolebinding gpustack-bootstrap --ignore-not-found
    kubectl delete namespace gpustack-default gpustack-system
    ```

5. *(Optional)* Clean the data each node still carries. Worker pods mount `/var/lib/gpustack` from the host, and nothing in the steps above touches it — it holds downloaded model weights and caches. Nothing in a reinstall objects to it: a cluster registered again on the same nodes picks the directory back up and skips re-downloading what is already there, so remove it only when the storage is needed for something else:

    ```bash
    rm -rf /var/lib/gpustack
    ```

Verify with `kubectl get all -n gpustack-system` (should be `NotFound`), `kubectl get clusterrolebinding | grep gpustack`, `kubectl get ds -A | grep -E 'gpustack|csi'`, and — for the objects only step 4 removes — `kubectl get apiservice,mutatingwebhookconfiguration,validatingwebhookconfiguration 2>/dev/null | grep gpustack` and `kubectl get crd | grep -E 'gpustack|worker\.gpustack'` (both should print nothing).

If a namespace is already stuck in `Terminating`, its objects still hold finalizers whose controllers are gone. Find them, then strip the finalizers by hand — safe only on a cluster you are removing. Delete the admission webhooks first (step 4), or the patch itself is intercepted by a webhook whose Service no longer exists:

```bash
kubectl api-resources --verbs=list --namespaced -o name |
  xargs -n1 -I{} sh -c 'kubectl get {} -n gpustack-default -o name 2>/dev/null'
kubectl patch <kind>/<name> -n gpustack-default \
  -p '{"metadata":{"finalizers":null}}' --type=merge
```

The same strip applies to cluster-scoped leftovers — an `instancetype.worker.gpustack.ai` object holding the `gpustack.ai/controlled` finalizer keeps its CRD (and with it the whole `worker.gpustack.ai` group) in deletion: `kubectl patch instancetypes.worker.gpustack.ai <name> -p '{"metadata":{"finalizers":null}}' --type=merge`.

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

### Creating SHUIHUA FUTURE Cluster

1. In the `Basic Configuration` step, the `Name` field is required and `Description` is optional. Create or select a Cloud Credential for communicating with the Shuihua API. Shuihua has no regions, so there is none to select. You must also configure the `GPUStack Server URL`, which will be accessible from the newly created instances.
2. `Default Container Registry` is required for this provider: Shuihua instances cannot reach Docker Hub, so a registry that resolves to one is rejected. The field suggests `quay.io` and `swr.cn-south-1.myhuaweicloud.com`, and accepts any other registry you can reach, such as your own Harbor or mirror.
3. Click `Next`.
4. Adding one or more `Worker Pools`. For each pool, `Name`, `Instance Type`, `OS Image`, `Replicas`, `Batch Size` and `Labels` can be specified. An `Instance Type` is a Shuihua spec template, listed with its GPU model, hourly price and remaining stock; a sold-out template cannot be selected. Volumes are not offered — Shuihua has no block storage API.
5. Click `Save` after the worker pools are configured.

Additional worker pools can be added after the cluster is created, the same way as for a DigitalOcean cluster.

Shuihua instances sit behind NAT with only ports 22 and 80 mapped, so the cluster uses the `tunnel` proxy mode by default and its workers serve inference through the server's WebSocket tunnel. Their listed IP is the instance's private address; use `View SSH Access` on the worker to get the endpoint that is actually reachable. See [Adding a GPU Cluster Using Shuihua](../tutorials/adding-gpucluster-using-shuihua.md) for the full walkthrough.

### Operating Worker Pools

You can manage worker pools for cloud provider clusters on the `Clusters` page:

1. Go to the `Clusters` page.
2. Find the cloud provider cluster you want to manage and expand it to view its worker pools.
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
# ========= TLS ===========
insecure_tls: false
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
