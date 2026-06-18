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

Worker DaemonSets are disabled by default (`worker.enabled=false`). When enabled, the chart always renders a CPU worker DaemonSet (`<release>-worker`), and one DaemonSet per GPU vendor listed in `worker.gpuVendors` (named `<release>-worker-<vendor>`).

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
| `worker.gpuVendors`                    | `[nvidia]`               | GPU vendors; one DaemonSet per vendor plus a CPU DaemonSet.                |
| `worker.nodeSelector`                  | `{}`                     | Base worker nodeSelector; replaces `global.nodeSelector` when non-empty.   |
| `worker.port`                          | `10150`                  | Worker service port.                                                      |
| `worker.metricsPort`                   | `10151`                  | Worker metrics port.                                                      |
| `worker.dataDir`                       | `/var/lib/gpustack`      | Host path mounted at `/var/lib/gpustack` inside each worker pod.           |
| `worker.environmentConfig`             | `{}`                     | Extra environment variables for the GPUStack worker.                      |

## Uninstallation

Uninstall the release:

```bash
helm uninstall gpustack -n gpustack-system
```

Helm does not remove PVCs created by the `StatefulSet`. The PVC is named `gpustack-data-dir-gpustack-server-0` (`<volumeClaimTemplate>-<statefulset>-<ordinal>`). To delete the persisted data, remove the leftover PVC (and optionally the namespace):

```bash
kubectl delete pvc gpustack-data-dir-gpustack-server-0 -n gpustack-system
kubectl delete namespace gpustack-system
```
