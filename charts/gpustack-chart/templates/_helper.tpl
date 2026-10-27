{{/* vim: set filetype=mustache: */}}
{{/*
NB: these two guards never run. Helm parses `_*.tpl` for its `define` blocks but
does not render the file, so top-level actions here are dead code — verified by
`helm template --set server.ingress.tls.cert=x` rendering cleanly. Moving them
to templates/validate.yaml (where guards do run) would start rejecting
configurations that install today, so it is left as a deliberate separate change.
*/}}
{{- if or (and .Values.server.ingress.tls.cert (not .Values.server.ingress.tls.key)) (and .Values.server.ingress.tls.key (not .Values.server.ingress.tls.cert)) }}
{{ fail "Both server.ingress.tls.cert and server.ingress.tls.key must be set together or both be empty." }}
{{- end }}
{{- if gt (int .Values.server.replicas) 1 }}
{{- if not .Values.server.externalDatabaseURL }}
{{ fail "server.externalDatabaseURL is required when server.replicas > 1." }}
{{- end }}
{{- end }}

{{/*
Normalize `worker.gpuVendors` into a deduplicated list, dropping null/empty
entries. Returns a JSON-encoded list so callers can `fromJsonArray` it.
*/}}
{{- define "gpustack.workerVendors" -}}
{{- $out := list -}}
{{- $seen := dict -}}
{{- range (.Values.worker.gpuVendors | default list) -}}
  {{- if . -}}
    {{- $v := . | toString -}}
    {{- if not (hasKey $seen $v) -}}
      {{- $_ := set $seen $v true -}}
      {{- $out = append $out $v -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- $out | toJson -}}
{{- end -}}

{{/*
Whether the CPU worker DaemonSet is part of this release, as "true" or "".

Read through this rather than off `.Values.worker.cpuEnabled` directly — the
DaemonSet, the mode flag below and the guard in validate.yaml all do — for two
reasons. They have to agree: a mode flag reading "labelled" while the DaemonSet
renders the legacy name leaves the worker Service selecting labels no pod
carries. And `nil` has to read as this chart's default (true) rather than as
false, which plain truthiness would give: Helm drops a key set to null, so
`worker: {cpuEnabled: }` in a partial values file would otherwise delete the CPU
workers from a cluster that never asked for that. Compared as a lowercased
string, so `--set-string worker.cpuEnabled=false` and a quoted `"False"` in a
values file turn them off as an unquoted `false` does, rather than being
non-empty strings that silently read as "on".
*/}}
{{- define "gpustack.workerCPUEnabled" -}}
{{- if ne (lower (toString .Values.worker.cpuEnabled)) "false" -}}true{{- end -}}
{{- end -}}

{{/*
True when the chart should render in multi-vendor mode: component + runtime
labels on every worker pod, a hostname anti-affinity between them, and a worker
Service that selects on the component label.

That is every case except the one where a single DaemonSet carries the
unsuffixed legacy name `<release>-worker` — CPU only, which is what a Service
selecting `app: <release>-worker` matches. At least one GPU vendor means a
suffixed DaemonSet exists, and so does turning the CPU one off, which leaves
nothing but suffixed names however few vendors are selected.
*/}}
{{- define "gpustack.multiVendorMode" -}}
{{- $vendors := include "gpustack.workerVendors" . | fromJsonArray -}}
{{- if or (gt (len $vendors) 0) (not (include "gpustack.workerCPUEnabled" .)) -}}true{{- end -}}
{{- end -}}

{{/*
Effective nodeSelector for the server pod, as YAML.
server.nodeSelector REPLACES global.nodeSelector when non-empty; otherwise
global.nodeSelector is used. Empty/null on both yields no output.
*/}}
{{- define "gpustack.serverNodeSelector" -}}
{{- if .Values.server.nodeSelector -}}
{{ toYaml .Values.server.nodeSelector }}
{{- else if .Values.global.nodeSelector -}}
{{ toYaml .Values.global.nodeSelector }}
{{- end -}}
{{- end -}}

{{/*
Effective BASE nodeSelector for worker DaemonSets (before PCI merging).
worker.nodeSelector REPLACES global.nodeSelector when non-empty. Returns
the map itself (not YAML) via JSON round-trip so callers can merge with
PCI labels.
*/}}
{{- define "gpustack.workerBaseNodeSelectorJson" -}}
{{- if .Values.worker.nodeSelector -}}
{{ .Values.worker.nodeSelector | toJson }}
{{- else if .Values.global.nodeSelector -}}
{{ .Values.global.nodeSelector | toJson }}
{{- else -}}
{}
{{- end -}}
{{- end -}}

{{/*
PCI vendor ID per GPU manufacturer. Mirrors _MANUFACTURER_PCI_ID in
gpustack/k8s/manifest_template.py. Used to derive deterministic
nodeSelector labels for each vendor DaemonSet.
*/}}
{{- define "gpustack.pciVendorIds" -}}
{"amd":"1002","ascend":"19e5","cambricon":"cabc","hygon":"1d94","iluvatar":"1e3e","metax":"9999","mthreads":"1ed5","nvidia":"10de","thead":"1ded"}
{{- end -}}

{{/*
Canonical vendor ordering (mirrors _RUNTIME_ORDER in manifest_template.py).
Used for deterministic output ordering of GPU vendor DaemonSets regardless
of the order they were listed in values.yaml. Returns a JSON-encoded list.
*/}}
{{- define "gpustack.canonicalVendorOrder" -}}
["amd","ascend","cambricon","hygon","iluvatar","metax","mthreads","nvidia","thead"]
{{- end -}}

{{/*
Sort the configured vendors into canonical order and return as JSON list.
All GPU vendors get suffixed DaemonSet names; ordering is for deterministic
output only.
*/}}
{{- define "gpustack.sortedVendors" -}}
{{- $vendors := include "gpustack.workerVendors" . | fromJsonArray -}}
{{- $canonical := include "gpustack.canonicalVendorOrder" . | fromJsonArray -}}
{{- $sorted := list -}}
{{- range $canonical -}}
  {{- if has . $vendors -}}
    {{- $sorted = append $sorted . -}}
  {{- end -}}
{{- end -}}
{{- $sorted | toJson -}}
{{- end -}}


{{/*
Tag of this chart's own image.

Required rather than defaulted to `v<appVersion>`: appVersion names the last
release, and its image pins a gpustack-runtime that can be a whole generation
away from the templates sitting next to it in a checkout. Pairing those two
silently is how an install ends up with an operator that derives the Kueue
scheduling chain one way and a worker that reads it another, surfacing as
"Failed to find Kueue queue name on node ..." at deploy time rather than as a
version error at install time. CI patches this value for every published chart,
so only checkout installs have to state it — which is exactly the case that
cannot be defaulted correctly.
*/}}
{{ define "gpustack.imageTag" -}}
{{ required "image.tag is required: name the gpustack image to pair with these templates (e.g. --set image.tag=dev-<sha> from a checkout). Published charts carry it already." .Values.image.tag -}}
{{ end -}}


{{/*
Resolve the registry + namespace prefix for images managed by this chart.

One key covers every image in the release, including the sub-charts': higress-core
reads `global.hub` natively (the Istio convention it inherits), and the
gpustack-operator chart accepts it as an alias for the `global.imageRegistry` its
own tree uses. Anything else would leave a mirrored install pulling half its
images from Docker Hub.
*/}}
{{ define "gpustack.hub" -}}
{{ trimSuffix "/" (required "global.hub is required" .Values.global.hub) -}}
{{ end -}}


{{ define "gpustack.image" -}}
{{ printf "%s/%s" (include "gpustack.hub" .) .Values.image.repository -}}
{{ end -}}


{{ define "server_config" -}}
{{ include "server_external_url" . }}
GPUSTACK_DEBUG: "{{ .Values.debug }}"
GPUSTACK_API_PORT: "{{ .Values.server.apiPort }}"
GPUSTACK_METRICS_PORT: "{{ .Values.server.metricsPort }}"
{{ if .Values.server.externalDatabaseURL -}}
GPUSTACK_DATABASE_URL: "{{ .Values.server.externalDatabaseURL }}"
{{- end }}
{{- with .Values.server.environmentConfig }}
{{- range $key, $value := . }}
{{ $key }}: "{{ $value }}"
{{- end }}
{{- end }}
{{- end -}}

{{ define "worker_config" -}}
GPUSTACK_DEBUG: "{{ .Values.debug }}"
GPUSTACK_WORKER_PORT: "{{ .Values.worker.port }}"
GPUSTACK_WORKER_METRICS_PORT: "{{ .Values.worker.metricsPort }}"
{{- with .Values.worker.environmentConfig }}
{{- range $key, $value := . }}
{{ $key }}: "{{ $value }}"
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Name of the Secret carrying GPUSTACK_TOKEN.

Setting `registrationTokenSecretName` points every consumer at a Secret this
release does not own and does not create. Two cases need that:

  - server and workers installed as two releases in one namespace. The Secret
    name is not release-prefixed, so both releases would render the same object
    and the second install would be refused for not owning it. The second
    release references the first's Secret instead.
  - a registration manifest that creates the Secret with kubectl and then hands
    the install to Helm. Helm never owns it, so re-rendering cannot delete or
    rotate the token.

Left empty the chart creates and references `registration-token`, as before.
*/}}
{{ define "gpustack.registrationTokenSecretName" -}}
{{ default "registration-token" .Values.registrationTokenSecretName -}}
{{ end -}}

{{/*
Address the workers register with.

`worker.serverURL` wins when set, so a worker-only install can point at a server
outside this release. Otherwise the server deployed alongside it is addressed
over its in-cluster Service — which only exists when `server.enabled` is true,
hence the hard failure rather than a silently unreachable default.
*/}}
{{ define "gpustack.workerServerURL" -}}
{{- if .Values.worker.serverURL -}}
{{ .Values.worker.serverURL }}
{{- else if .Values.server.enabled -}}
{{ printf "http://%s-server.%s.svc:%v" .Release.Name .Release.Namespace .Values.server.apiPort }}
{{- else -}}
{{ fail "worker.serverURL is required when server.enabled is false: the workers have no in-release server to register with." }}
{{- end -}}
{{- end -}}

{{ define "higressPlugins.image" -}}
{{ printf "%s/%s:%s" (include "gpustack.hub" .) .Values.higressPlugins.image.repository (required "higressPlugins.image.tag is required" .Values.higressPlugins.image.tag) -}}
{{- end -}}

{{ define "chart_labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end -}}


{{ define "server_external_url" -}}
{{- if not .Values.server.environmentConfig.GPUSTACK_SERVER_EXTERNAL_URL -}}
{{- $schema := "http" -}}
{{- if and .Values.server.ingress.tls.cert .Values.server.ingress.tls.key }}
{{- $schema = "https" -}}
{{- end }}
{{- if .Values.server.ingress.hostname -}}
GPUSTACK_SERVER_EXTERNAL_URL: {{ printf "%s://%s" $schema .Values.server.ingress.hostname }}
{{- end -}}
{{- end -}}
{{- end -}}

{{ define "tls_secret_name" -}}
{{- if .Values.server.ingress.hostname -}}
{{ printf "tls-%s" (.Values.server.ingress.hostname | replace "." "-") }}
{{- end -}}
{{- end -}}


{{ define "ingress_tls" -}}
{{- if and .Values.server.ingress.tls.cert .Values.server.ingress.tls.key .Values.server.ingress.hostname }}
tls:
  - secretName: {{ include "tls_secret_name" . }}
    hosts:
      - {{ .Values.server.ingress.hostname }}
{{- end }}
{{- end -}}


{{/*
Normalize one TLS protocol version onto Higress' spelling, or fail the render.

Refusing the install is the point. Higress fails *open* on a version string it
cannot parse: the Ingress applies, the listener keeps its TLS 1.0 default, and
the only trace is a line in the higress-controller log. A `TLSv1.4` or a
`TLSv1_2` that rendered fine would leave the floor exactly where it was while
looking like it had been raised.

Underscores and case are normalized rather than rejected -- `TLSv1_2` is Envoy's
own spelling and the likeliest thing to reach for. Mirrors
`_normalized_tls_protocol_version` in gpustack/gateway/utils.py, which does the
same for the environment variables the non-in-cluster modes use.

Args: dict with `input` (the configured value) and `field` (its values path,
used in the error message).
*/}}
{{- define "normalized_tls_protocol_version" -}}
{{- $candidate := .input | toString | replace "_" "." | lower -}}
{{- $match := "" -}}
{{- range $supported := list "TLSv1.0" "TLSv1.1" "TLSv1.2" "TLSv1.3" -}}
{{- if eq $candidate (lower $supported) -}}{{- $match = $supported -}}{{- end -}}
{{- end -}}
{{- if not $match -}}
{{/* `.input | toString` before %q: %q on a bool or int renders as %!q(bool=false)
or an escape sequence, which tells the operator nothing about what they typed. */}}
{{- fail (printf "%s: %q is not a TLS version Higress accepts. Valid values are TLSv1.0, TLSv1.1, TLSv1.2, TLSv1.3 -- anything else is ignored by Higress, which keeps accepting TLS 1.0." .field (.input | toString)) -}}
{{- end -}}
{{- $match -}}
{{- end -}}

{{/*
TLS protocol version bounds for this Ingress' listener, as Higress' annotations.

Only rendered here. The Ingress this chart creates is the anchor GPUStack reads
when it generates an Ingress per LLM route, so setting the bounds once here puts
them on the whole gateway -- there is no second place to keep in step.
*/}}
{{- define "ingress_tls_protocol_annotations" -}}
{{- $tls := .Values.server.ingress.tls -}}
{{- $min := "" -}}
{{- $max := "" -}}
{{/* An explicit nil/empty test rather than `with`, which also treats `false`
and `0` as unset. Those are not TLS versions, but letting them skip validation
would render no annotation at all and leave the listener on TLS 1.0 -- the
silent failure this block exists to prevent. Anything not null and not empty
goes to the validator, which names it in the error. */}}
{{- $rawMin := $tls.minProtocolVersion -}}
{{- if and (not (kindIs "invalid" $rawMin)) (ne (toString $rawMin) "") -}}
{{- $min = include "normalized_tls_protocol_version" (dict "input" $rawMin "field" "server.ingress.tls.minProtocolVersion") -}}
{{- end -}}
{{- $rawMax := $tls.maxProtocolVersion -}}
{{- if and (not (kindIs "invalid" $rawMax)) (ne (toString $rawMax) "") -}}
{{- $max = include "normalized_tls_protocol_version" (dict "input" $rawMax "field" "server.ingress.tls.maxProtocolVersion") -}}
{{- end -}}
{{/* Lexical order matches version order across these four, all same length and
differing only in the last digit, so this needs no index lookup. */}}
{{- if and $min $max (gt $min $max) -}}
{{- fail (printf "server.ingress.tls.minProtocolVersion (%s) is higher than server.ingress.tls.maxProtocolVersion (%s); no TLS version would be accepted." $min $max) -}}
{{- end -}}
{{- with $min }}
higress.io/tls-min-protocol-version: "{{ . }}"
{{- end }}
{{- with $max }}
higress.io/tls-max-protocol-version: "{{ . }}"
{{- end }}
{{- end -}}


{{- define "image_pull_secrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}
