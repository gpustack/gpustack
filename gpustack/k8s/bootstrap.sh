#!/bin/bash
#
# Installs the GPUStack chart into the cluster this Job runs in.
#
# Rendered into a ConfigMap by bootstrap.jinja and run by the bootstrap Job,
# which is named after the moment the manifest was rendered rather than after
# what it contains — so every fetch produces a Job that runs, and this script is
# what decides whether there is anything to do, by comparing the revision it
# carries against the one the release recorded. A name derived from the values
# would be tidier and wrong: A to B and back to A resolves to the Job that
# already installed A.
#
# Every step therefore has to be safe to repeat — on a redundant apply, and
# because `backoffLimit` restarts this from the top after any failure.
#
# Values are read from the mounted ConfigMap rather than baked in, so a Job that
# starts late installs the current configuration; a copy is taken here, and the
# rendering stamp checked, so it cannot install an older one over a newer.

set -euo pipefail

RELEASE="${RELEASE:?RELEASE is required}"
NAMESPACE="${NAMESPACE:?NAMESPACE is required}"
CHART_URL="${CHART_URL:?CHART_URL is required}"
VALUES_FILE="${VALUES_FILE:?VALUES_FILE is required}"
DESIRED_REVISION="${DESIRED_REVISION:?DESIRED_REVISION is required}"
CHART_DIGEST="${CHART_DIGEST:?CHART_DIGEST is required}"
RENDERED_AT="${RENDERED_AT:?RENDERED_AT is required}"
BOOTSTRAP_CONFIGMAP="${BOOTSTRAP_CONFIGMAP:?BOOTSTRAP_CONFIGMAP is required}"

# Releases the operator creates at runtime for the applications this chart
# deploys itself. The operator installs an application only where no chart
# deploys it, so every cluster registered before this one — where the manifest
# deployed the operator alone — has these five, each owning objects this chart
# now renders.
#
# Adopting them is the operator's own documented migration (its
# docs/migration/to-subcharts.md), not a collision: nothing is torn down, the
# objects change release and keep running. The first four are the set that
# migration names; the device managers are here too because the pre-chart
# manifest left the operator to install those as well.
OPERATOR_APP_RELEASES="gpustack-kueue gpustack-node-feature-discovery gpustack-csi-driver-nfs gpustack-csi-driver-s3 gpustack-operator-device-manager"

log() { echo "[bootstrap] $*"; }
fail() {
  echo "[bootstrap] $*" >&2
  exit 1
}

# `set -e` leaves nothing behind but the failing command's own stderr, which for
# a kubectl or helm failure says what went wrong and never says what this script
# was doing at the time. Whoever reads these logs is reading them because
# something failed, so name the step.
trap 'echo "[bootstrap] FAILED at line ${LINENO} (exit $?): ${BASH_COMMAND}" >&2' ERR

for binary in helm kubectl jq curl sha256sum; do
  command -v "${binary}" >/dev/null 2>&1 || fail "${binary} is not available in this image"
done

# What this run is working with, before it can fail at anything. A chart that
# cannot be fetched is the most common first failure and its URL is the first
# thing anyone asks for; it is otherwise only in the Job's env. The Helm version
# is here because this depends on one: `--take-ownership` arrived in 3.17.
helm_version=$(helm version --template '{{.Version}}' 2>/dev/null || echo "")
log "release=${RELEASE} namespace=${NAMESPACE} helm=${helm_version:-unknown}"
# `--take-ownership` arrived in 3.17, and this whole migration rests on it.
# Without this, helm reports an unknown flag, which reads like a typo in this
# script rather than an image too old to run it.
case "${helm_version#v}" in
3.1[0-6].* | 3.[0-9].* | 2.* | 1.*)
  fail "helm ${helm_version} in this image predates --take-ownership, which this install needs; the operator image is expected to carry 3.17 or newer"
  ;;
esac
log "chart=${CHART_URL}"

rendered=$(mktemp)
live=$(mktemp)
chart=$(mktemp)
values=$(mktemp)
scan_errors=$(mktemp)
trap 'rm -f "${rendered}" "${live}" "${chart}" "${values}" "${scan_errors}"' EXIT

# Helm reports a failed hook as "pre-install hooks failed: job failed", naming
# neither the hook nor a reason: both are in the hook's pod, which Helm leaves
# behind precisely because it failed (`hook-delete-policy: hook-succeeded`). The
# operator chart runs the migration this install depends on from such a hook, so
# a failure there is a likely one and its logs are the whole diagnosis.
# A legacy release record whose objects this release now owns is unfinished
# cleanup, and the two cases have to be told apart rather than assumed. A record
# alone proves nothing: a component this cluster deliberately keeps — Kueue it
# already ran, switched off through `helmValues` — is never adopted, and its
# record is supposed to stay. A record whose objects answer to `${RELEASE}`,
# though, outlived the run that adopted them, and left alone it stays loaded: a
# later `helm uninstall` of that record deletes what this release now owns.
#
# Asked of the record itself, through Helm, so no list of each release's objects
# has to be kept here and the answer holds however the adoption happened — this
# run, or one that was killed before it could finish.
retire_adopted_records() {
  local app_release
  for app_release in ${OPERATOR_APP_RELEASES}; do
    if [[ -z "$(kubectl get secret --namespace "${NAMESPACE}" \
      --selector "owner=helm,name=${app_release}" -o name 2>/dev/null)" ]]; then
      continue
    fi

    # Answered in jq rather than by matching text in the shell: adoption is
    # normally partial — the objects this chart's version renders become ours
    # and the rest stay the legacy release's — so the answer is a list, and a
    # substring test over it reads the mixed case, the common one, as a no.
    #
    # A read that fails is a no as well, and deliberately: keeping a record
    # costs nothing on its own, while retiring one whose component this cluster
    # still runs would orphan it from Helm.
    if ! helm get manifest "${app_release}" --namespace "${NAMESPACE}" 2>/dev/null |
      kubectl get -f - --ignore-not-found -o json 2>/dev/null |
      jq -e --arg release "${RELEASE}" '
        any((.items // [.])[];
          (.metadata.annotations // {})["meta.helm.sh/release-name"] == $release)
      ' >/dev/null 2>&1; then
      log "leaving the release record of ${app_release}, whose objects are still its own"
      continue
    fi

    # Prune first, retire second. The record is what brings this function back
    # to a release at all, so deleting it before the objects it accounts for
    # would destroy the evidence that anything is left: a failed delete, or a
    # Job stopped in between, and no later run would ever look again — the
    # orphans, which include controllers and webhooks, would simply stay.
    if ! prune_release_leftovers "${app_release}"; then
      log "keeping the release record of ${app_release} so a later run retries its cleanup"
      continue
    fi

    log "retiring the release record of ${app_release}, whose objects are now this release's"
    kubectl delete secret --namespace "${NAMESPACE}" \
      --ignore-not-found \
      --selector "owner=helm,name=${app_release}"
  done
}

# What a retired release created and this chart's version of it does not render.
# Adoption rewrote `app.kubernetes.io/instance` on everything the render names,
# so an object still carrying the old one is owned by nobody now and invisible
# to any later uninstall. Run per release as it is retired, so a run resuming
# after Helm succeeded prunes what its predecessor could not reach.
#
# CRDs, PersistentVolumes and PersistentVolumeClaims are left alone even when
# they match: deleting a CRD takes every custom resource of that kind with it.
prune_release_leftovers() {
  local app_release="$1" selector kind failed=0
  selector="app.kubernetes.io/instance=${app_release},app.kubernetes.io/managed-by=Helm"

  log "pruning what ${app_release} left behind"
  for kind in deployments daemonsets statefulsets services serviceaccounts \
    configmaps secrets roles rolebindings poddisruptionbudgets jobs networkpolicies; do
    kubectl delete "${kind}" --namespace "${NAMESPACE}" \
      --ignore-not-found --selector "${selector}" || {
      log "WARNING: could not prune every orphaned ${kind}"
      failed=1
    }
  done
  for kind in clusterroles clusterrolebindings mutatingwebhookconfigurations \
    validatingwebhookconfigurations csidrivers storageclasses apiservices; do
    kubectl delete "${kind}" \
      --ignore-not-found --selector "${selector}" || {
      log "WARNING: could not prune every orphaned ${kind}"
      failed=1
    }
  done

  # Reported rather than fatal: the install itself has already succeeded, and a
  # kept record is a retry on the next apply. Failing here would restart this
  # script from the top to reach the same four deletes.
  return "${failed}"
}

# Objects the pre-chart manifest created that this chart does not render. Helm
# prunes only what it rendered before, so it never learns about these: they are
# not in any release and would keep running unmanaged.
#
# By exact name, never by prefix or label: this Job's own name shares a prefix
# with the one it replaces. Idempotent, and cheap enough to run unconditionally
# on both paths — a run killed between the install and here leaves them, and
# testing for them first only to decide whether to reinstall would spend a Helm
# revision to reach four deletes.
remove_pre_chart_objects() {
  log "removing objects left by the pre-chart manifest"
  kubectl delete --namespace "${NAMESPACE}" --ignore-not-found \
    service/gpustack-worker \
    configmap/gpustack-operator-worker-deployment \
    job/gpustack-operator-worker-deployment
  # The blunt grant the pre-chart manifest gave the worker. Left in place it
  # keeps cluster-admin bound to the ServiceAccount, and the chart's
  # fine-grained roles would only be added alongside it.
  kubectl delete --ignore-not-found clusterrolebinding/gpustack-worker
}

dump_failed_hooks() {
  local job pod
  for job in $(
    kubectl get job --namespace "${NAMESPACE}" -o json 2>/dev/null |
      jq -r '.items[]
        | select((.metadata.annotations // {})["helm.sh/hook"] != null)
        | select((.status.failed // 0) > 0)
        | .metadata.name' || true
  ); do
    for pod in $(kubectl get pod --namespace "${NAMESPACE}" \
      --selector "job-name=${job}" -o name 2>/dev/null || true); do
      log "--- logs of the failed hook ${pod} ---"
      kubectl logs --namespace "${NAMESPACE}" "${pod}" --tail=80 2>&1 || true
    done
  done
}

# The chart records the configuration it was installed from, as part of the
# release. When it already matches what this Job was created for, and the
# release is in a state that says the install finished, there is nothing to do.
#
# Not for correctness — `helm upgrade` with the same values changes no objects —
# but for `helm history`: every apply would otherwise add a no-op revision, and
# Helm keeps ten by default, so a handful of redundant applies would push the
# revision anyone would actually want to roll back to out of the list.
#
# Both conditions are required. A `failed` or `pending-*` release may well have
# applied this ConfigMap before stopping, so the recorded revision alone cannot
# say the install finished.
#
# What the pair does not say is that the release is still intact: an object
# deleted out of band leaves the status `deployed` and the revision matching, so
# a re-apply skips and does not put it back. The manifest this replaced did,
# being a `kubectl apply` of every object every time — and that is the property
# traded away for a re-apply that costs a pod instead of a Helm revision, which
# a GitOps loop re-syncing this file makes the common case. Repairing an
# out-of-band deletion is `helm upgrade`'s job, and any configuration change
# runs one.
release_status=$(helm status "${RELEASE}" --namespace "${NAMESPACE}" -o json 2>/dev/null | jq -r '.info.status' || true)
recorded_revision=$(
  kubectl get configmap "${RELEASE}-applied-revision" \
    --namespace "${NAMESPACE}" \
    --ignore-not-found \
    -o jsonpath='{.data.appliedRevision}' 2>/dev/null || true
)
# Nothing to install, but the migration's cleanup still runs: a run killed
# between a successful `helm upgrade` and it records the revision and leaves the
# rest behind, and both halves are by name and idempotent. Doing them here
# rather than testing for them and reinstalling to reach them keeps a redundant
# apply at four deletes instead of a Helm revision.
if [[ "${release_status}" == "deployed" && "${recorded_revision}" == "${DESIRED_REVISION}" ]]; then
  log "release is already at revision ${DESIRED_REVISION}, nothing to install"
  retire_adopted_records
  remove_pre_chart_objects
  exit 0
fi
log "installing revision ${DESIRED_REVISION} (release is ${release_status:-absent} at ${recorded_revision:-none})"

# What this install would create, from the same chart and values it will use.
# Everything below reasons about this set rather than a hardcoded inventory, so
# a chart that starts or stops rendering an object needs no change here.
# Fetched once, and checked against the digest the manifest was rendered with:
# the URL carries no version, so nothing else ties the bytes this Job installs
# to the revision it will record. A server serving a different chart at that
# address — mid-rollout, or through a cache — would otherwise have the old
# templates record the new revision, and every later Job skip.
#
# That digest is also what authenticates the download, which is why the transfer
# does not verify the server's certificate. It cannot: a GPUStack server with a
# private CA is reached by workers because their own image merges that CA into
# its trust store at startup, and this Job runs the operator's image, which does
# not. Verification would therefore fail exactly where the old flow — which
# never fetched from the server in-cluster — worked. Nothing is lost by dropping
# it: the digest arrives with the manifest, applied through the caller's own
# kubeconfig, so bytes that do not match it are refused whatever served them,
# and a Helm chart is not a secret.
#
# The values are copied for the same reason as the digest: the ConfigMap they
# arrive in is updated in place, so reading it twice can render one
# configuration and install another.
log "fetching the chart"
curl --fail --silent --show-error --location --insecure \
  --retry 3 --retry-connrefused --retry-delay 3 \
  --output "${chart}" "${CHART_URL}"
fetched_digest=$(sha256sum "${chart}" | cut -d' ' -f1)
if [[ "${fetched_digest}" != "${CHART_DIGEST}" ]]; then
  fail "the chart served at ${CHART_URL} is not the one this manifest was rendered for: expected ${CHART_DIGEST}, got ${fetched_digest}"
fi
cp "${VALUES_FILE}" "${values}"

# What this install would create, from the same chart and values it will use.
# Everything below reasons about this set rather than a hardcoded inventory, so
# a chart that starts or stops rendering an object needs no change here.
# `--dry-run=server`, so `lookup` resolves against this cluster rather than to
# nothing. The operator chart renders a RuntimeClass only when the cluster does
# not already carry a foreign one — a vendor GPU operator's, typically — and a
# client-side render, blind to that, would produce an object set the install will
# not create. The guard below would then refuse a live RuntimeClass nobody is
# about to touch, and registration would fail on exactly the clusters this chart
# is for.
log "rendering the chart"
helm template "${RELEASE}" "${chart}" \
  --namespace "${NAMESPACE}" \
  --values "${values}" \
  --dry-run=server >"${rendered}"

# Pre-flight, before anything is created or deleted: an object we are about to
# claim that already belongs to a *different* Helm release is a sibling
# release's, and adopting it would hand its objects to this release — the next
# `helm upgrade` on that release would then prune them. Helm's own ownership
# check cannot tell this case from the one below, so it is made here.
#
# Except for the operator's own application releases, where taking them over is
# the point rather than an accident. Which of them this install actually adopts
# is read from the cluster rather than assumed: a release only turns up here
# when the render names one of its objects, so an application the values switch
# off is never touched by the migration below.
#
# `helm template` does not contact the cluster, so this is the first API call
# and a failure at this point has changed nothing.
# Read strictly. `--ignore-not-found` already covers the expected case — an
# object this release has not created yet — so anything left is a real failure:
# denied authorization, an unreachable API server, a malformed render. Swallowing
# those would leave the scan below empty and the guard passing, and the install
# would then take ownership of whatever is out there without ever having looked.
# "The kind is not registered yet" is not among the cases refused here, because
# every kind these values render is built-in — a property of the values rather
# than of the chart, which with its own defaults emits Istio's EnvoyFilter. It
# holds because `higress-core.enabled` and `server.enabled` are the server's to
# set, and refused to a caller for that reason;
# `tests/k8s/test_chart_values.py` fails if a render ever brings a custom
# resource in.
log "checking for objects owned by another release"
if ! kubectl get -f "${rendered}" --ignore-not-found -o json >"${live}" 2>"${scan_errors}"; then
  fail "could not read the state of the objects this release would create, so whether any of them belongs to somebody else is unknown: $(tr '\n' ' ' <"${scan_errors}")"
fi
# Namespace as well as name: every cluster registers a release called
# `gpustack`, so on a Kubernetes cluster hosting two of them in two namespaces —
# which the check below tells you to do — the name alone says "mine" about the
# other one's objects. The cluster-scoped ones are shared between them: adopting
# Kueue's CRDs away from the release that installed them leaves that release
# ready to prune what this one now depends on.
# An empty file is the honest answer on a cluster where none of these exist
# yet, and the only reason jq would have nothing to parse now that the read
# above is strict.
owners=""
if [[ -s "${live}" ]]; then
  owners=$(
    jq -r --arg release "${RELEASE}" --arg namespace "${NAMESPACE}" '
      [ (.items // [.])[]
        | (.metadata.annotations // {}) as $annotations
        | ($annotations["meta.helm.sh/release-name"] // "") as $owner
        | ($annotations["meta.helm.sh/release-namespace"] // "") as $owner_namespace
        | select($owner == "" or $owner != $release or $owner_namespace != $namespace)
        | "\($owner)|\($owner_namespace)|\(.kind)/\(.metadata.name)|\(.metadata.namespace // "")"
      ] | .[]
    ' "${live}"
  )
fi

# The pre-chart manifest only ever created namespaced objects in this namespace,
# plus two ClusterRoleBindings. So an unowned object anywhere else is somebody
# else's — a Kueue installed from raw manifests being the case that costs the
# most, since this chart templates its CRDs and adopting those puts every
# Workload in the cluster behind this release's uninstall.
PRE_CHART_CLUSTER_OBJECTS="ClusterRoleBinding/gpustack-worker ClusterRoleBinding/gpustack-operator-worker"

foreign=""
adopted_releases=""
# `|`, not a tab: a tab is an IFS whitespace character, so a run of them
# collapses into one delimiter and an unowned object — whose first field is
# empty — would shift every field left and read as owned. No Kubernetes name,
# namespace or kind can contain a `|`.
while IFS='|' read -r owner owner_namespace object object_namespace; do
  [[ -n "${object}" ]] || continue
  if [[ -z "${owner}" ]]; then
    if [[ "${object_namespace}" == "${NAMESPACE}" ]] ||
      [[ " ${PRE_CHART_CLUSTER_OBJECTS} " == *" ${object} "* ]]; then
      continue
    fi
    foreign="${foreign}${object} (owned by no release)
"
    continue
  fi
  if [[ "${owner_namespace}" == "${NAMESPACE}" &&
    " ${OPERATOR_APP_RELEASES} " == *" ${owner} "* ]]; then
    [[ " ${adopted_releases} " == *" ${owner} "* ]] || adopted_releases="${adopted_releases} ${owner}"
    continue
  fi
  foreign="${foreign}${object} (release ${owner} in namespace ${owner_namespace})
"
done <<<"${owners}"

if [[ -n "${foreign}" ]]; then
  fail "refusing to install: this release would take over objects it did not create, and a later uninstall of it would delete them:
${foreign}"
fi

adopted_list=""
for app_release in ${adopted_releases}; do
  adopted_list="${adopted_list:+${adopted_list},}${app_release}"
done

# A release of this name that owns a StatefulSet is a server install, not a
# previous run of this path: these values never render one. Upgrading it with
# worker-only values would re-render the server from them and prune whatever
# they leave out — quietly, because Helm considers that a normal upgrade.
#
# The name has to be this one: the chart derives object names from the release
# name, and `gpustack` is what makes them match the manifest that registered
# clusters before the chart, so they can be adopted instead of duplicated.
#
# Only a release that exists can own one, and asking about one that does not
# means suppressing a failure from both commands — which then reads the same as
# an answer.
existing_statefulsets=""
if [[ -n "${release_status}" ]]; then
  existing_statefulsets=$(
    helm get manifest "${RELEASE}" --namespace "${NAMESPACE}" 2>/dev/null |
      kubectl get -f - --ignore-not-found -o jsonpath='{range .items[?(@.kind=="StatefulSet")]}{.metadata.name} {end}' 2>/dev/null || true
  )
fi
if [[ -n "${existing_statefulsets// /}" ]]; then
  fail "refusing to install: release '${RELEASE}' in namespace '${NAMESPACE}' already owns StatefulSet(s) ${existing_statefulsets}, so it is a GPUStack server install. Installing workers here would re-render that release from worker-only values. Register this cluster into a different namespace."
fi

# Overtaken? Two applies can overlap: this Job took its copy of the values when
# it started, and a later manifest updates the ConfigMap in place. Acting now
# would undo what the newer one is doing — and Helm's own bookkeeping only
# orders the two operations, it does not decide which should win.
#
# Asked twice, because both of the things this Job does after here are
# destructive to a concurrent install: the repair below rolls back or drops the
# release record, and the install itself would put the cluster back on this
# Job's configuration.
exit_if_overtaken() {
  local current_stamp
  current_stamp=$(
    kubectl get "configmap/${BOOTSTRAP_CONFIGMAP}" --namespace "${NAMESPACE}" \
      --ignore-not-found -o jsonpath='{.data.renderedAt}' 2>/dev/null || true
  )
  if [[ -n "${current_stamp}" && "${current_stamp}" != "${RENDERED_AT}" ]]; then
    log "a later manifest (${current_stamp}) has been applied, leaving this one to it"
    exit 0
  fi
}

# Before the repair, not only before the install: a stale Job that rolled back
# an upgrade a newer Job is performing would take the newer one down with it,
# and the newer Job's retry would then be repairing damage rather than
# installing.
exit_if_overtaken

# A release left mid-operation by a killed Job blocks every later attempt with
# "another operation is in progress". Repair it, never uninstall: this release
# owns Kueue, whose CRDs are Helm-managed templates and whose custom resources
# carry controller finalizers, so an uninstall tears down the controller while
# the finalizers still pin the CRs — and it would take the worker DaemonSets
# with it.
case "${release_status}" in
pending-install)
  # No previous revision to roll back to: the release record is all that exists,
  # so dropping it lets the install start over. Whatever the interrupted attempt
  # managed to create is adopted below.
  log "clearing an interrupted first install"
  kubectl delete secret --namespace "${NAMESPACE}" \
    --ignore-not-found \
    --selector "owner=helm,name=${RELEASE}"
  ;;
pending-upgrade | pending-rollback)
  log "rolling back an interrupted upgrade"
  helm rollback "${RELEASE}" --namespace "${NAMESPACE}" --wait --timeout 5m
  ;;
esac

# A workload whose `spec.selector` still names the release being retired cannot
# be patched into this one: the field is immutable, and Helm fails the entire
# install on it. Deleting it lets the install recreate it — the same trade the
# operator chart makes, and the reason this runs before anything is installed
# rather than after a failure.
#
# The operator chart frees these itself, but only when its own release is being
# upgraded: on install it logs "nothing is adopted" and skips. That holds
# everywhere except here, where the release is new by construction — a cluster
# registered before the chart has no release of this name — while the objects
# being adopted are years old.
if [[ -n "${adopted_list}" ]]; then
  for kind in deployments daemonsets statefulsets; do
    stale=$(
      kubectl get "${kind}" --namespace "${NAMESPACE}" \
        --selector "app.kubernetes.io/instance in (${adopted_list}),app.kubernetes.io/managed-by=Helm" \
        -o json 2>/dev/null |
        jq -r --arg release "${RELEASE}" '
          .items[]
          | select((.spec.selector.matchLabels["app.kubernetes.io/instance"] // "")
                   | . != "" and . != $release)
          | .metadata.name
        ' || true
    )
    [[ -n "${stale}" ]] || continue
    log "deleting ${kind} whose selector adoption cannot rewrite: ${stale//$'\n'/ }"
    # shellcheck disable=SC2086 # deliberate word splitting: one name per line, none can contain spaces
    kubectl delete "${kind}" --namespace "${NAMESPACE}" ${stale} --ignore-not-found
  done
fi

# `--take-ownership` adopts objects whatever owns them, which is how a cluster
# registered before the chart existed is migrated in place: its worker
# DaemonSets and operator Deployment — owned by nobody — and the applications
# the operator installed as releases of its own keep running and become this
# release's. Safe only because the pre-flight above decided which of those two
# cases each object is, and refused anything that is neither.
#
# No `--wait`: this Job's job is to apply the release, not to babysit it. The
# operator's own startup can take minutes (it installs two custom resources that
# poll for CRDs), and a Job that outlives its deadline waiting would be retried
# into an upgrade it already completed.
# Read the stamp again, as late as possible: the window between this check and
# the install is the one in which a newer manifest can still slip past.
exit_if_overtaken

log "installing ${RELEASE} into ${NAMESPACE}"
if ! helm upgrade --install "${RELEASE}" "${chart}" \
  --namespace "${NAMESPACE}" \
  --values "${values}" \
  --take-ownership; then
  dump_failed_hooks
  fail "helm failed to install ${RELEASE}; the release is now $(helm status "${RELEASE}" --namespace "${NAMESPACE}" -o json 2>/dev/null | jq -r '.info.status' || echo unknown)"
fi

# The release records that still claim what was just adopted. Deleting a record
# leaves its objects alone — `helm uninstall` is what would delete them, and
# that is the hazard being closed: left in place, an `helm uninstall
# gpustack-kueue` by anyone takes Kueue out from under this release.
#
# The operator chart does this itself, in a `post-upgrade` hook. That hook does
# not run here: this release is being created, not upgraded, so Helm fires
# `pre-install` (which reaps a stranded Kueue and applies the subcharts' CRDs)
# and nothing after. Mirrors its files/migrate-post.sh.
retire_adopted_records

# Objects the pre-chart manifest created that this chart does not render. Helm
# prunes only what it rendered before, so it never learns about these: they are
# not in any release and would keep running unmanaged.
#
# By exact name, never by prefix or label: this Job's own name shares a prefix
# with the one it replaces.
remove_pre_chart_objects

log "done"
