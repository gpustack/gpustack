"""The Pod Security Admission level GPUStack's own namespaces sit at.

One definition, because there are two places namespaces get created — the
bootstrap manifest renders the system and cluster-owner ones, and
`gpu_instances.cluster_apis` creates every `gpustack-<org>` beyond them — and a
label applied in one place and not the other leaves whole families of
namespaces rejecting their own Pods.

A leaf module with no imports of its own, so either side can read it without
dragging the other's dependencies along.
"""

from typing import Dict, Final

# Pod Security Admission is built into Kubernetes and enforced at Pod
# *creation*: a Pod that does not fit the level is rejected outright, not left
# Pending. GPUStack's own workloads legitimately need what the stricter levels
# forbid — host networking (RDMA binds its GID to a NIC address), hostPort (the
# side channel a disaggregated pair hands its peer), host IPC (CUDA-IPC KV
# buffer sharing), and device mounts — so this family of namespaces has to sit
# at `privileged`.
#
# All three keys, not just `enforce`: with `warn`/`audit` left on the cluster
# default, every Pod creation still returns a warning and writes an audit
# annotation, which is noise that hides the real ones.
#
# Blast radius, stated deliberately: `privileged` means every Pod in the
# namespace is exempt from PSA. That is a reason these are namespaces GPUStack
# creates and owns, rather than ones shared with a customer's own workloads. It
# also does nothing about Kyverno / Gatekeeper / OPA, which are separate
# webhooks that these labels do not address.
NAMESPACE_LABELS: Final[Dict[str, str]] = {
    "pod-security.kubernetes.io/enforce": "privileged",
    "pod-security.kubernetes.io/audit": "privileged",
    "pod-security.kubernetes.io/warn": "privileged",
}
