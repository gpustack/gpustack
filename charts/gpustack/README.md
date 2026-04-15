# GPUStack Helm Chart

Since v2.2.0, deploying GPUStack on Kubernetes with an in-cluster gateway is supported. Follow this guide to build a k3s cluster and install GPUStack.

Deploying GPUStack in a Kubernetes cluster with Higress is still considered experimental. Please review the [limitations](#limitations) before proceeding with installation.

## Requirements

| Component  | Version   |
| ---------- | --------- |
| Helm       | v3.18.4   |
| Kubernetes | >=v1.30.0 |
| GPUStack   | v2.2.0    |

## Limitations

Please note the following limitations when deploying GPUStack on Kubernetes:

- GPUStack Server is deployed as a StatefulSet and currently does not support more than one replica.
- By default, the built-in `Postgres` database is used at startup. It is recommended to specify an external database using the `server.externalDatabaseURL` parameter.
- By default, the StatefulSet uses `volumeClaimTemplates` (10Gi PVC), which requires a default `StorageClass` to be configured in your cluster (in k3s, the default is `local-path`). Alternatively, set `server.dataVolume.hostPath` to use a host path volume instead of a PVC.
- In the current version, token usage statistics are not available in Kubernetes deployments. This is because the statistics logic relies on metrics from the Higress gateway, and in this deployment mode, it is currently unable to automatically discover the gateway pod IP address for metrics collection. This limitation will be addressed in future versions.
- Higress plugins are served by a dedicated `gpustack/higress-plugins` Deployment installed alongside GPUStack. When the Higress gateway restarts, it will attempt to download the plugins from this service. If the service is unavailable, the gateway's startup will be blocked until the plugins are accessible.
- The bundled `higress-core` sub-chart deploys Higress as the cluster's ingress controller. If another ingress controller is already running in the cluster, set `higress-core.enabled=false` and configure `gateway.ingressClassname` to use the existing Higress instance instead.

## Install k3s

> **Note:** The following steps use k3s as an example Kubernetes distribution. Other CNCF-conformant Kubernetes distributions (such as RKE2, kubeadm-based clusters, or managed cloud Kubernetes services) should work as well, as long as they meet the version requirements above.

Use the following script to install k3s v1.30.11. Traefik is disabled because Higress will be used as the ingress controller. For high-availability k3s clusters, please refer to the [k3s documentation](https://docs.k3s.io/datastore/ha).

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.30.11+k3s1 INSTALL_K3S_EXEC="--disable=traefik" sh -
```

After installation, you can verify your setup with:

```bash
kubectl version
```

## Install GPUStack with Helm

Clone the GPUStack GitHub repository to obtain the charts:

```bash
git clone https://github.com/gpustack/gpustack.git
```

Navigate to the `charts` directory and install GPUStack:

```bash
cd gpustack/charts
helm install -n gpustack-system gpustack ./gpustack --create-namespace
```

By default, the `higress-core` sub-chart is enabled and deployed alongside GPUStack. If you already have Higress installed in your cluster, disable the bundled Higress and point GPUStack to your existing instance:

```bash
helm install -n gpustack-system gpustack ./gpustack --create-namespace \
  --set higress-core.enabled=false \
  --set gateway.ingressClassname=<your-higress-ingressclass>
```

### Installing Higress Separately

If you set `higress-core.enabled=false`, install a compatible Higress instance before deploying GPUStack:

```bash
# Add the Higress Helm repository
helm repo add higress.io https://higress.io/helm-charts

# Install higress-core v2.1.9
helm install higress higress.io/higress-core -n higress-system --create-namespace --version 2.1.9
```

Verify the IngressClass is available:

```bash
kubectl get ingressclass higress
# NAME      CONTROLLER                      PARAMETERS   AGE
# higress   higress.io/higress-controller   <none>       3m46s
```

If you need to customize Higress parameters, refer to the [Higress documentation](https://higress.cn/en/docs/latest/ops/deploy-by-helm).

### GPUStack Helm Chart Parameters

| Parameter                              | Default                  | Description                                                                       |
| -------------------------------------- | ------------------------ | --------------------------------------------------------------------------------- |
| debug                                  | false                    | Enable debug mode                                                                 |
| registrationToken                      | null                     | Registration token; auto-generated if null, reused across upgrades                |
| defaultDataDir                         | /var/lib/gpustack        | Host data directory path for worker nodes                                         |
| enableWorkers                          | true                     | Enable worker nodes                                                               |
| clusterDomain                          | cluster.local            | Kubernetes cluster service domain suffix                                          |
| systemDefaultContainerRegistry         | null                     | Default image registry prefix                                                     |
| gpustackImage                          | null                     | Full image name (overrides repository/tag)                                        |
| image.repository                       | gpustack/gpustack        | Image repository name                                                             |
| image.tag                              | null                     | Image tag, defaults to chart's appVersion                                         |
| image.pullPolicy                       | IfNotPresent             | Image pull policy                                                                 |
| server.ingress.hostname                | null                     | Ingress hostname                                                                  |
| server.ingress.tls.cert                | null                     | Ingress TLS certificate content                                                   |
| server.ingress.tls.key                 | null                     | Ingress TLS private key content                                                   |
| server.externalDatabaseURL             | null                     | External database connection string                                               |
| server.dataVolume.hostPath             | null                     | Host path for server data volume; if set, uses a hostPath volume instead of a PVC |
| server.dataVolume.size                 | 10Gi                     | Server data volume size (PVC)                                                     |
| server.apiPort                         | 30080                    | API service port                                                                  |
| server.metricsPort                     | 10161                    | Metrics port                                                                      |
| server.environmentConfig               | {}                       | Extra environment variables for GPUStack server                                   |
| gateway.ingressClassname               | higress                  | Higress IngressClass name; GPUStack checks if it exists to enable in-cluster mode |
| higress-core.enabled                   | true                     | Deploy Higress gateway as a sub-chart; disable if Higress is already installed    |
| higress-core.global.ingressClass       | higress                  | Must match `gateway.ingressClassname`                                             |
| higress-core.global.enablePluginServer | false                    | GPUStack manages its own plugin server; keep disabled                             |
| higress-core.global.hub                | docker.io/gpustack       | Image hub for Higress component images                                            |
| higress-core.downstream.idleTimeout    | 1800                     | Downstream idle timeout in seconds                                                |
| higress-core.upstream.idleTimeout      | 3                        | Upstream idle timeout in seconds                                                  |
| higress-core.gateway.hub               | null                     | Image hub override for the gateway component                                      |
| higress-core.gateway.image             | mirrored-higress-gateway | Gateway image name                                                                |
| higress-core.controller.hub            | null                     | Image hub override for the controller component                                   |
| higress-core.controller.image          | mirrored-higress-higress | Controller image name                                                             |
| higress-core.pilot.hub                 | null                     | Image hub override for the pilot component                                        |
| higress-core.pilot.image               | mirrored-higress-pilot   | Pilot image name                                                                  |
| higressPlugins.image.repository        | gpustack/higress-plugins | Higress plugins image repository                                                  |
| higressPlugins.image.tag               | "0.2.0"                  | Higress plugins image tag; required, versioned independently from GPUStack        |
| higressPlugins.image.pullPolicy        | IfNotPresent             | Higress plugins image pull policy                                                 |
| worker.gpuVendor                       | nvidia                   | GPU vendor (null/nvidia/mthreads/amd/ascend/hygon/metax/iluvatar/cambricon/thead) |
| worker.port                            | 10150                    | Worker service port                                                               |
| worker.metricsPort                     | 10151                    | Worker metrics port                                                               |
| worker.environmentConfig               | {}                       | Extra environment variables for GPUStack worker                                   |
| worker.extraVolumeMounts               | []                       | Extra volume mounts appended to the worker container                              |
| worker.extraVolumes                    | []                       | Extra volumes appended to the worker DaemonSet                                    |

To customize parameters, use `--set key=value` or `-f your-values.yaml` during installation.

## Accessing GPUStack

After installing all required charts, retrieve the initial admin password for the GPUStack server with:

```bash
kubectl exec -it -n gpustack-system gpustack-server-0 -- cat /var/lib/gpustack/initial_admin_password
```

If you did not specify `server.ingress.hostname`, you can obtain the GPUStack Server UI address with:

```bash
kubectl get ingress -n gpustack-system gpustack -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
```
