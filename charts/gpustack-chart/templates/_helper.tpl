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
True when the chart should render in multi-vendor mode (CPU DS + at least one
GPU vendor DS, meaning 2+ DaemonSets total). Controls anti-affinity, component
labels, and service selector.
*/}}
{{- define "gpustack.multiVendorMode" -}}
{{- $vendors := include "gpustack.workerVendors" . | fromJsonArray -}}
{{- if gt (len $vendors) 0 -}}true{{- end -}}
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


{{- define "image_pull_secrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}
