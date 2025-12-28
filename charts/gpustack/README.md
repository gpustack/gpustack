# GPUStack Helm Chart

Since v2.2.0, deploying GPUStack on Kubernetes with an in-cluster gateway is supported. Follow this guide to build a k3s cluster and install GPUStack.

Deploying GPUStack in a Kubernetes cluster with Higress is still considered experimental. Please review the [limitations](#limitations) before proceeding with installation.

## Requirements

| Component    | Version   |
| ------------ | --------- |
| Helm         | v3.18.4   |
| Kubernetes   | >=v1.30.0 |
| Higress-core | v2.1.9    |
| GPUStack     | v2.2.0    |

## Limitations

Please note the following limitations when deploying GPUStack on Kubernetes:

- GPUStack Server is deployed as a StatefulSet and currently does not support more than one replica.
- By default, the built-in `Postgres` database is used at startup. It is recommended to specify an external database using the `server.externalDatabaseURL` parameter.
- By default, the StatefulSet uses `volumeClaimTemplates` (10Gi PVC), which requires a default `StorageClass` to be configured in your cluster (in k3s, the default is `local-path`). Alternatively, set `server.dataVolume.hostPath` to use a host path volume instead of a PVC.
- In the current version, token usage statistics are not available in Kubernetes deployments. This is because the statistics logic relies on metrics from the Higress gateway, and in this deployment mode, it is currently unable to automatically discover the gateway pod IP address for metrics collection. This limitation will be addressed in future versions.
- Higress plugins are served by a dedicated `gpustack/higress-plugins` Deployment installed alongside GPUStack. When the Higress gateway restarts, it will attempt to download the plugins from this service. If the service is unavailable, the gateway's startup will be blocked until the plugins are accessible.

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

## Install Higress with Helm

Ensure your Helm version is `>=3.2`:

```bash
helm version
# Example output:
# version.BuildInfo{Version:"v3.18.4", ...}
```

Install Higress on your Kubernetes cluster with the following steps:

```bash
# Add the Higress Helm repository
helm repo add higress.io https://higress.io/helm-charts

# Install higress-core v2.1.9
helm install higress higress.io/higress-core -n higress-system --create-namespace --version 2.1.9

# Check the ingressClass to ensure Higress is installed
kubectl get ingressclass higress
# NAME      CONTROLLER                      PARAMETERS   AGE
# higress   higress.io/higress-controller   <none>       3m46s
```

If you need to customize Higress parameters, refer to the [Higress documentation](https://higress.cn/en/docs/latest/ops/deploy-by-helm) and set values as needed.

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

### GPUStack Helm Chart Parameters

| Parameter                       | Default                          | Description                                                                       |
| ------------------------------- | -------------------------------- | --------------------------------------------------------------------------------- |
| debug                           | false                            | Enable debug mode                                                                 |
| registrationToken               | random_generated_token_change_me | Registration token, should be changed                                             |
| defaultDataDir                  | /var/lib/gpustack                | Host data directory path for worker nodes                                         |
| enableWorkers                   | true                             | Enable worker nodes                                                               |
| clusterDomain                   | cluster.local                    | Kubernetes cluster service domain suffix                                          |
| systemDefaultContainerRegistry  | null                             | Default image registry prefix                                                     |
| gpustackImage                   | null                             | Full image name (overrides repository/tag)                                        |
| image.repository                | gpustack/gpustack                | Image repository name                                                             |
| image.tag                       | null                             | Image tag, defaults to chart's appVersion                                         |
| image.pullPolicy                | IfNotPresent                     | Image pull policy                                                                 |
| server.ingress.hostname         | null                             | Ingress hostname                                                                  |
| server.ingress.tls.cert         | null                             | Ingress TLS certificate content                                                   |
| server.ingress.tls.key          | null                             | Ingress TLS private key content                                                   |
| server.externalDatabaseURL      | null                             | External database connection string                                               |
| server.dataVolume.hostPath      | null                             | Host path for server data volume; if set, uses a hostPath volume instead of a PVC |
| server.apiPort                  | 30080                            | API service port                                                                  |
| server.metricsPort              | 10161                            | Metrics port                                                                      |
| server.environmentConfig        | null                             | Extra environment variables for GPUStack server                                   |
| gateway.ingressClassname        | higress                          | IngressClass name; if not `higress`, GPUStack uses embedded gateway mode          |
| higressPlugins.image.repository | gpustack/higress-plugins         | Higress plugins image repository                                                  |
| higressPlugins.image.tag        | ""                               | Higress plugins image tag, defaults to chart's appVersion                         |
| higressPlugins.image.pullPolicy | IfNotPresent                     | Higress plugins image pull policy                                                 |
| worker.gpuVendor                | nvidia                           | GPU vendor; controls driver mounts. `nvidia`/`mthreads` set runtimeClassName      |
| worker.port                     | 10150                            | Worker service port                                                               |
| worker.metricsPort              | 10151                            | Worker metrics port                                                               |
| worker.environmentConfig        | null                             | Extra environment variables for GPUStack worker                                   |
| worker.extraVolumeMounts        | []                               | Extra volume mounts appended to the worker container                              |
| worker.extraVolumes             | []                               | Extra volumes appended to the worker DaemonSet                                    |

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
