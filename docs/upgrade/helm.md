# Upgrade via Helm

You can upgrade a Kubernetes-based GPUStack installation with `helm upgrade`.

The following upgrade instructions apply only to GPUStack v2.2.0 and later, which is the first version deployable via Helm.

!!! note

    Deploying and upgrading GPUStack on Kubernetes with Higress is currently considered **experimental**.

!!! warning

    **Backup First:** Before proceeding with an upgrade, it’s strongly recommended to back up your database.

    If you use the bundled embedded PostgreSQL database, its data lives in the server's PVC (named `gpustack-data-dir-<release-name>-server-0`, e.g. `gpustack-data-dir-gpustack-server-server-0` for the default `gpustack-server` release), which is mounted inside the server pod at `/var/lib/gpustack`; the embedded PostgreSQL data directory lives under that mount at `postgresql/data`. Back it up before upgrading.

    If you use an external database (`server.externalDatabaseURL`), follow your database provider's backup procedure instead.

## Trace Storage on Helm Upgrades

Built-in Tempo runs in a separate workload. When upgrading a release where Tempo ran in a Server Pod, the new Tempo workload uses its own volume. Helm leaves the old `<data_dir>/tempo` files in the Server data volume. Back up those files separately if you need to retain historical traces.

By default, the new Tempo Pod uses an `emptyDir` and starts with empty trace storage on every Pod replacement. To keep traces across later Pod replacements, set `tracing.tempo.existingClaim` to an existing PVC before upgrading. See [Configure Helm](../user-guide/observability.md#configure-helm).

## Upgrade the Release

Always pin the target version with `--version <chart-version>` so the upgrade lands on a known, reproducible release (the chart version matches the GPUStack release, e.g. `2.2.0`), and reuse the values you supplied at install time by passing the same `-f values.yaml` (and/or `--set`) flags you used originally, or `--reuse-values` to keep the previously applied values. The release name depends on the documentation you followed at install time — older instructions used `gpustack`, current ones use `gpustack-server` — so the snippet below derives it from the `meta.helm.sh/release-name` annotations on the namespace's StatefulSets (worker-only releases registered through GPUStack clusters own no StatefulSet and are excluded by the pattern) and only runs the upgrade when exactly one matching release is found:

```bash
RELEASE_NAME=$(kubectl get statefulset -n gpustack-system \
  -o jsonpath='{range .items[*]}{.metadata.annotations.meta\.helm\.sh/release-name}{"\n"}{end}' \
  | grep -E '^gpustack(-server)?$' | sort -u)
[ "$(printf '%s\n' "$RELEASE_NAME" | grep -c .)" -eq 1 ] \
  || { echo "expected one GPUStack server release in gpustack-system, got: '$RELEASE_NAME'" >&2; false; } \
  && helm upgrade "$RELEASE_NAME" oci://registry-1.docker.io/gpustack/gpustack-chart \
    --namespace gpustack-system \
    --version <chart-version> \
    -f values.yaml
```

!!! warning

    If you omit `--version`, Helm resolves the newest stable chart version automatically. Avoid this for upgrades: you cannot control which version you land on. Always specify `--version` to target a specific release.

## Server and Worker Ordering

The GPUStack server is deployed as a `StatefulSet` and chart-managed workers as `DaemonSet`s, so both are rolled to the new image as part of the release upgrade. Because the server and workers belong to the **same release**, the "server first, then workers" ordering cannot be enforced by a single `helm upgrade`. If strict ordering matters in your environment, wait for the server pod to become ready before allowing the worker pods to roll:

```bash
kubectl rollout status statefulset/${RELEASE_NAME}-server -n gpustack-system
```

Workers that were registered outside the chart through GPUStack clusters (for example, Docker or Kubernetes clusters added through the UI) are not managed by this Helm release and must be upgraded following the [Upgrade a Cluster Deployment](cluster.md) steps.
