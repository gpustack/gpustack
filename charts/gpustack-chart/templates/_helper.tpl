{{/* vim: set filetype=mustache: */}}
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
True when the chart should render in multi-vendor mode (2+ distinct vendors).
*/}}
{{- define "gpustack.multiVendorMode" -}}
{{- $vendors := include "gpustack.workerVendors" . | fromJsonArray -}}
{{- if gt (len $vendors) 1 -}}true{{- end -}}
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
Controls which vendor owns the legacy DaemonSet name in multi-vendor mode.
Returns a JSON-encoded list.
*/}}
{{- define "gpustack.canonicalVendorOrder" -}}
["amd","ascend","cambricon","hygon","iluvatar","metax","mthreads","nvidia","thead"]
{{- end -}}

{{/*
Sort the configured vendors into canonical order and return as JSON list.
The first vendor in canonical order keeps the legacy DaemonSet name.
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


{{ define "gpustack.imageTag" -}}
{{ default (printf "v%s" .Chart.AppVersion) .Values.image.tag -}}
{{ end -}}


{{/*
Resolve the registry + namespace prefix for images managed by this chart.
Pulls from `.Values.global.hub`, trimming any trailing slash for safe printf.
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


{{/*
Operator image tag. Requires operator.image.tag to be set (patched
automatically by CI from gpustack/__init__.py __operator_version__).
Fails explicitly when the tag is unset instead of falling back to a
stale default.
*/}}
{{- define "gpustack.operatorImageTag" -}}
{{- required "operator.image.tag is required (set it explicitly or rely on CI patching)" .Values.operator.image.tag -}}
{{- end -}}


{{/*
Full operator image reference: {global.hub}/{operator.image.repository}:{tag}
*/}}
{{- define "gpustack.operatorImage" -}}
{{ printf "%s/%s:%s" (include "gpustack.hub" .) .Values.operator.image.repository (include "gpustack.operatorImageTag" .) -}}
{{- end -}}


{{/*
Effective nodeSelector for the operator pod.
operator.nodeSelector REPLACES global.nodeSelector when non-empty.
*/}}
{{- define "gpustack.operatorNodeSelector" -}}
{{- if .Values.operator.nodeSelector -}}
{{ toYaml .Values.operator.nodeSelector }}
{{- else if .Values.global.nodeSelector -}}
{{ toYaml .Values.global.nodeSelector }}
{{- end -}}
{{- end -}}
