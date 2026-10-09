# Installation via Helm

Since v2.2.0, GPUStack can be deployed on Kubernetes with an in-cluster [Higress](https://higress.io/) API gateway using the official Helm chart.

!!! note

    Deploying GPUStack on Kubernetes with Higress is currently considered **experimental**. Review the [limitations](#limitations) before proceeding.

## Prerequisites

| Component  | Version    |
| ---------- | ---------- |
| Helm       | >= v3.18.4 |
| Kubernetes | >= v1.30.0 |
| GPUStack   | >= v2.2.0  |

In addition:

- A default `StorageClass` must be configured in the cluster for the server's data volume (in k3s the default is `local-path`). Alternatively, set `server.dataVolume.hostPath` to use a host path volume instead of a PVC.
- For GPU workers, ensure the appropriate GPU drivers and container toolkits are installed on the nodes (see [Installation Requirements](./requirements.md)), and that [Node Feature Discovery (NFD)](https://kubernetes-sigs.github.io/node-feature-discovery/) is installed so GPU nodes are labeled.

## Limitations

- The GPUStack server is deployed as a `StatefulSet` and currently does **not** support more than one replica.
- By default, the bundled embedded `PostgreSQL` database is used. It is recommended to specify an external database via `server.externalDatabaseURL` for production.
- Higress plugins are served by a dedicated `gpustack/higress-plugins` Deployment installed alongside GPUStack. The Higress gateway downloads plugins from this service on restart; if the service is unavailable, gateway startup is blocked until the plugins become accessible.
- The bundled `higress-core` sub-chart deploys Higress as the cluster's ingress controller. If another ingress controller is already running, set `higress-core.enabled=false` and point `gateway.ingressClassname` at an existing Higress instance.

## Install k3s (optional)

The following steps use k3s as an example Kubernetes distribution. Other CNCF-conformant distributions (RKE2, kubeadm-based clusters, or managed cloud Kubernetes) work as well, as long as they meet the version requirements above.

Install k3s with Traefik disabled, since Higress is used as the ingress controller. For high-availability k3s clusters, refer to the [k3s documentation](https://docs.k3s.io/datastore/ha).

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.30.11+k3s1 INSTALL_K3S_EXEC="--disable=traefik" sh -
```

Verify the setup:

```bash
kubectl version
```

## Install GPUStack with Helm

The chart is published as an OCI artifact on Docker Hub at `oci://registry-1.docker.io/gpustack/gpustack-chart`. The chart version tracks the GPUStack release (e.g. chart `2.2.0` ships GPUStack `v2.2.0`).

Install the latest stable release. When `--version` is omitted, Helm resolves the newest stable chart version automatically and skips dev/pre-release builds:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace
```

To pin a specific version, append `--version <chart-version>` (the chart version matches the GPUStack release, e.g. `2.2.0`):

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --version <chart-version>
```

By default, the `higress-core` sub-chart is enabled and deployed alongside GPUStack. If you already have Higress installed in your cluster, disable the bundled Higress and point GPUStack at your existing instance:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set higress-core.enabled=false \
  --set gateway.ingressClassname=<your-higress-ingressclass>
```

To customize parameters, use `--set key=value` or `-f your-values.yaml` during installation. See [Chart Parameters](#chart-parameters) below.

### Installing from a Cloned Repository

Alternatively, clone the repository and install from the local chart directory (useful for air-gapped environments or when modifying the chart):

```bash
git clone https://github.com/gpustack/gpustack.git
cd gpustack/charts
helm dependency update ./gpustack-chart
helm install gpustack ./gpustack-chart \
  --namespace gpustack-system --create-namespace
```

### Installing Higress Separately

If you set `higress-core.enabled=false`, install a compatible Higress instance before deploying GPUStack:

```bash
# Add the Higress Helm repository
helm repo add higress.io https://higress.io/helm-charts

# Install higress-core (match the version pinned by the GPUStack chart)
helm install higress higress.io/higress-core \
  --namespace higress-system --create-namespace \
  --version 2.1.9
```

Verify the IngressClass is available, and that its name matches `gateway.ingressClassname`:

```bash
kubectl get ingressclass higress
# NAME      CONTROLLER                      PARAMETERS   AGE
# higress   higress.io/higress-controller   <none>       3m46s
```

For Higress customization, refer to the [Higress documentation](https://higress.cn/en/docs/latest/ops/deploy-by-helm).

## Accessing GPUStack

Wait for the server pod to become ready:

```bash
kubectl get pods -n gpustack-system -w
```

Retrieve the initial admin password:

```bash
kubectl exec -it -n gpustack-system gpustack-server-0 -- \
  cat /var/lib/gpustack/initial_admin_password
```

If you did not set `server.ingress.hostname`, obtain the GPUStack UI address from the ingress:

```bash
kubectl get ingress -n gpustack-system gpustack \
  -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```

Open the address in a browser and log in with username `admin` and the password retrieved above.

## Common Configuration

### Enabling Worker DaemonSets

Worker DaemonSets are disabled by default (`worker.enabled=false`). When enabled, the chart renders a CPU worker DaemonSet (`<release>-worker`), and one DaemonSet per GPU vendor listed in `worker.gpuVendors` (named `<release>-worker-<vendor>`).

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set 'worker.gpuVendors={nvidia}'
```

Supported `worker.gpuVendors` values: `nvidia`, `mthreads`, `amd`, `ascend`, `hygon`, `metax`, `iluvatar`, `cambricon`, `thead`.

!!! note

    Each GPU DaemonSet receives an automatic PCI-presence `nodeSelector` label (e.g. `feature.node.kubernetes.io/pci-10de.present: "true"` for NVIDIA), advertised by Node Feature Discovery. **NFD must be installed** in the cluster, otherwise no nodes carry the required labels and all worker pods stay `Pending`.

Whenever at least one GPU vendor is listed (i.e. `worker.gpuVendors` is non-empty), every worker pod additionally gets a required `podAntiAffinity` (topologyKey=hostname) so two workers cannot share a node — this protects the `hostNetwork: true` ports from collision.

The CPU worker DaemonSet covers the nodes no GPU vendor claims, through the `nodeSelector` `feature.gpustack.ai/acceleratable: "false"`. Set `worker.cpuEnabled=false` to leave those nodes alone — typically when the control plane shares the cluster with the GPU nodes and must not gain workers:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set worker.enabled=true \
  --set worker.cpuEnabled=false \
  --set 'worker.gpuVendors={nvidia}'
```

`worker.gpuVendors` must then name at least one supported vendor; otherwise the release would render no worker DaemonSet at all and the install is refused. The GPU DaemonSets keep their `-<vendor>` suffix either way, so turning the CPU one off never renames one of them to `<release>-worker`.

Alternatively, add GPU clusters and worker nodes through the UI on the **Clusters** and **Workers** pages after installation.

### Using an External Database

By default GPUStack uses the embedded PostgreSQL database. To use an external PostgreSQL or MySQL database, set `server.externalDatabaseURL`:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set server.externalDatabaseURL="postgresql://user:password@host:port/dbname"
```

### Enabling HTTPS with a Custom Certificate

Provide the certificate and key contents (PEM) via `server.ingress.tls`. When both are set, the ingress schema becomes HTTPS:

```yaml
# values.yaml
server:
  ingress:
    hostname: gpustack.example.com
    tls:
      cert: |-
        -----BEGIN CERTIFICATE-----
        MIID...
        -----END CERTIFICATE-----
      key: |-
        -----BEGIN PRIVATE KEY-----
        MIIE...
        -----END PRIVATE KEY-----
```

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  -f values.yaml
```

### Pulling Images From a Private Registry

To pull all images (GPUStack server/worker, higress-plugins, and the bundled higress-core gateway/controller/pilot) from a mirrored private registry, override `global.hub`. This relies on Helm's global-values propagation, so a single setting covers every image:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set global.hub=myregistry.example.com
```

`global.hub` is also passed to the server as `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`, ensuring inference engine images (e.g. vLLM, llama.cpp) are pulled from the same registry.

To supply pull credentials, the chart can create a `docker-registry` Secret named `gpustack-image-pull-secret` and wire it into all pods:

```bash
helm install gpustack oci://registry-1.docker.io/gpustack/gpustack-chart \
  --namespace gpustack-system --create-namespace \
  --set imagePullSecret.credentials.registry=registry.example.com \
  --set imagePullSecret.credentials.username=myuser \
  --set imagePullSecret.credentials.password=mypassword
```

To reference your own pre-existing Secrets instead, replace `global.imagePullSecrets`:

```yaml
global:
  imagePullSecrets:
    - name: my-existing-secret
```

## Chart Parameters

The most commonly used parameters are listed below. For the complete and authoritative list, see the chart's [`values.yaml`](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/values.yaml) and [README](https://github.com/gpustack/gpustack/blob/main/charts/gpustack-chart/README.md).

| Parameter                              | Default                  | Description                                                                |
| -------------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `debug`                                | `false`                  | Enable debug mode.                                                         |
| `registrationToken`                    | `null`                   | Worker registration token; a random one is generated and reused if `null`. |
| `clusterDomain`                        | `cluster.local`          | Kubernetes cluster service domain suffix.                                  |
| `global.hub`                           | `docker.io`              | Container registry host; override for a private registry.                  |
| `global.imagePullSecrets`              | `[gpustack-image-pull-secret]` | Image pull Secrets attached to all pods and propagated to sub-charts. |
| `global.nodeSelector`                  | `{}`                     | Default nodeSelector for every component; replaced by component-level value. |
| `image.repository`                     | `gpustack/gpustack`      | Image repo with namespace; final ref is `{global.hub}/{repository}:{tag}`. |
| `image.tag`                            | `null`                   | Image tag; defaults to the chart's `appVersion`.                           |
| `image.pullPolicy`                     | `IfNotPresent`           | Image pull policy.                                                         |
| `imagePullSecret.credentials.registry` | `docker.io`              | Registry host used when the chart creates the pull Secret.                 |
| `imagePullSecret.credentials.username` | `null`                   | Registry username; creates the Secret when set with password.             |
| `imagePullSecret.credentials.password` | `null`                   | Registry password; creates the Secret when set with username.             |
| `server.ingress.hostname`              | `null`                   | Ingress hostname for the server.                                          |
| `server.ingress.tls.cert`              | `null`                   | Ingress TLS certificate (PEM); enables HTTPS when set with key.           |
| `server.ingress.tls.key`               | `null`                   | Ingress TLS private key (PEM).                                            |
| `server.externalDatabaseURL`           | `null`                   | External database connection string (PostgreSQL or MySQL).                 |
| `server.dataVolume.hostPath`           | `null`                   | Host path for the server data volume; uses hostPath instead of a PVC.      |
| `server.dataVolume.size`               | `100Gi`                  | Server data volume size (PVC).                                            |
| `server.apiPort`                       | `30080`                  | API service port.                                                         |
| `server.metricsPort`                   | `10161`                  | Server metrics port.                                                      |
| `server.environmentConfig`             | `{}`                     | Extra environment variables for the GPUStack server.                      |
| `server.nodeSelector`                  | `{}`                     | Server pod nodeSelector; replaces `global.nodeSelector` when non-empty.    |
| `gateway.ingressClassname`             | `higress`                | Higress IngressClass name; enables in-cluster gateway mode when found.     |
| `higress-core.enabled`                 | `true`                   | Deploy the bundled Higress gateway; disable if already installed.          |
| `worker.enabled`                       | `false`                  | Render worker DaemonSets.                                                 |
| `worker.gpuVendors`                    | `[nvidia]`               | GPU vendors; one DaemonSet per vendor plus the CPU DaemonSet.              |
| `worker.cpuEnabled`                    | `true`                   | Render the CPU worker DaemonSet; `false` requires a GPU vendor.            |
| `worker.nodeSelector`                  | `{}`                     | Base worker nodeSelector; replaces `global.nodeSelector` when non-empty.   |
| `worker.port`                          | `10150`                  | Worker service port.                                                      |
| `worker.metricsPort`                   | `10151`                  | Worker metrics port.                                                      |
| `worker.dataDir`                       | `/var/lib/gpustack`      | Host path mounted at `/var/lib/gpustack` inside each worker pod.           |
| `worker.environmentConfig`             | `{}`                     | Extra environment variables for the GPUStack worker.                      |

## Uninstallation

The release deploys more than the server: the `gpustack-operator` sub-chart brings the Kueue and Node Feature Discovery controllers, and the operator creates custom resources that carry finalizers. `helm uninstall` removes the controllers with everything else, so the custom resources have to be gone **first** — deleted while the controllers are still running — or the objects stay in `Terminating` forever and the `gpustack-system` namespace never leaves it. In order:

1. Drain the workloads while **every** controller is still running. The operator cleans up the finalizers on its `Instance` objects, and Kueue refuses to release a `ClusterQueue` while admitted `Workload`s still occupy it — so these have to go before anything is switched off. `Instance`s are namespaced and live in the tenants' namespaces, so enumerate all of them with `-A`. Delete the models from the server (or their instances directly); each instance's pod terminates with it, and the Workload goes with the Instance. Do not delete the namespace's pods wholesale — that takes the operator, Kueue and NFD controllers down with it, and they are what finish the cleanup:

    ```bash
    kubectl delete instances.worker.gpustack.ai --all -A
    kubectl get instances.worker.gpustack.ai -A; kubectl get workloads.kueue.x-k8s.io -A   # both empty before moving on
    ```

    If Kueue is shared with other applications, do not use `--all` here or in step 3 — delete only GPUStack's objects. Everything the operator creates in Kueue is named with a `gpustack` prefix (`clusterqueue/gpustack--…`, `admissioncheck/gpustack-node-devices`). The cluster-scoped kinds select by name alone; the namespaced ones need the namespace carried along, because `-o name` omits it:

    ```bash
    kubectl get clusterqueues.kueue.x-k8s.io -o name | grep '/gpustack' \
      | xargs -I {} kubectl delete {}                # clusterqueues, resourceflavors, admissionchecks
    kubectl get localqueues.kueue.x-k8s.io -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | awk '$2 ~ /^gpustack/' | while read ns name; do \
        kubectl -n "$ns" delete localqueues.kueue.x-k8s.io "$name"; done
    ```

2. Stop the operator's own controllers, but leave the release installed so the Kueue controllers keep running — they are what finalize the deletions in the next step. As long as `gpustack-operator-worker` and the `gpustack-operator-device-manager-*` DaemonSets run, they recreate the `LocalQueue`s, `ClusterQueue`s and the `gpustack-node-devices` AdmissionCheck as soon as you delete them, so the counts never reach zero:

    ```bash
    kubectl -n gpustack-system scale deploy/gpustack-operator-worker --replicas=0
    kubectl -n gpustack-system get ds -l app.kubernetes.io/name=gpustack-operator-device-manager -o name | xargs -I {} \
      kubectl -n gpustack-system patch {} -p '{"spec":{"template":{"spec":{"nodeSelector":{"gpustack.ai/uninstall-pause":"true"}}}}}'
    ```

3. Delete the Kueue resources in dependency order — `LocalQueue`s, then `ClusterQueue`s, then `AdmissionCheck`s, then `ResourceFlavor`s. Kueue holds the `kueue.x-k8s.io/resource-in-use` finalizer on an object whose users are still present: the `gpustack-node-devices` AdmissionCheck stays in deletion until every ClusterQueue is gone, and each ClusterQueue until its LocalQueues and admitted Workloads are. Wait for each kind to be empty before moving to the next. On a shared cluster, apply the `gpustack`-prefix scoping from step 1 to every command instead of `--all`:

    ```bash
    kubectl delete localqueues.kueue.x-k8s.io --all -A
    kubectl delete clusterqueues.kueue.x-k8s.io --all
    kubectl delete admissionchecks.kueue.x-k8s.io --all
    kubectl delete resourceflavors.kueue.x-k8s.io --all
    ```

4. Uninstall the release. Do not pass `--wait`: the Node Feature Discovery post-delete prune hook blocks the deletion wait for longer than its `--timeout`, and killing Helm to escape it leaves the prune RBAC behind ownerless (step 5 removes those by hand — and run step 5 even without `--wait`):

    ```bash
    helm uninstall gpustack -n gpustack-system
    ```

5. Remove what `helm uninstall` does not. Helm never deletes CRDs, and the operator registers admission webhooks and APIServices that no release owns — both fail closed against the now-gone operator Service, and the webhooks reject the finalizer strip below until they are deleted:

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl get instances.worker.gpustack.ai -A -o jsonpath='{range .items[*]}{.metadata.namespace} {.metadata.name}{"\n"}{end}' \
      | while read ns name; do [ -n "$name" ] && kubectl -n "$ns" patch instances.worker.gpustack.ai "$name" \
        --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null; done
    kubectl get instancetypes.worker.gpustack.ai -o name | xargs -I {} \
      kubectl patch {} --type=json -p='[{"op":"remove","path":"/metadata/finalizers"}]' 2>/dev/null
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    The `worker.gpustack.ai` group is always GPUStack's own — no other component uses it — so its CRDs are safe to delete unconditionally.

6. Delete the CRDs the release leaves behind — **only the ones this release installed**. Deleting a CRD deletes every object of that kind in the whole cluster, so if the cluster already ran Kueue, Node Feature Discovery or Higress before GPUStack (see `higress-core.enabled` above) and anything else still uses them, leave their CRDs in place and skip this step for those groups. Discover what this release owns rather than guessing:

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "") == "gpustack")
      | .metadata.name'
    ```

    On a stock install that lists Kueue's CRDs, and NFD's and Higress's if the chart installed them:

    ```bash
    kubectl get crd -o name | grep 'kueue\.x-k8s\.io' | xargs -I {} kubectl delete {} --ignore-not-found
    kubectl delete crd nodefeatures.nfd.k8s-sigs.io nodefeaturerules.nfd.k8s-sigs.io \
      nodefeaturegroups.nfd.k8s-sigs.io --ignore-not-found
    kubectl delete crd envoyfilters.networking.istio.io http2rpcs.networking.higress.io \
      mcpbridges.networking.higress.io wasmplugins.extensions.higress.io --ignore-not-found
    ```

    The cluster provider's own CRDs are never GPUStack's to delete — `*.k3s.cattle.io` and `helm.cattle.io` on a k3s cluster, for instance, belong to the cluster.

7. *(Optional)* Delete the namespace, which takes the Higress objects, the bootstrap ConfigMap, ServiceAccount and the registration token Secret with it. Know what goes with it: the namespace holds the server's PVC (`gpustack-data-dir-gpustack-server-0`, `<volumeClaimTemplate>-<statefulset>-<ordinal>`), which carries the server's persisted data — the embedded PostgreSQL database with its users, models and clusters. On a `Delete` reclaim policy (the `local-path` default) the underlying volume is deleted as well. Keep the namespace if you want the data for a reinstall; otherwise delete the namespace (and the PVC first, if the reclaim policy does not already remove it):

    ```bash
    kubectl delete pvc gpustack-data-dir-gpustack-server-0 -n gpustack-system
    kubectl delete namespace gpustack-system
    ```
