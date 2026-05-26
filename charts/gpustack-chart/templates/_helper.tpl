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
Effective BASE nodeSelector for worker DaemonSets (before vendor merging).
worker.nodeSelector REPLACES global.nodeSelector when non-empty. Returns
the map itself (not YAML) via JSON round-trip so callers can merge with a
vendor override.
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
Multi-vendor mode invariants. Mirrors the cluster route layer:
  - every configured vendor must declare a non-empty
    gpuVendorOverrides[<vendor>].nodeSelector;
  - no two vendors may share an identical nodeSelector;
  - no base nodeSelector key may appear in any vendor override.
Single-vendor or zero-vendor mode skips all of these checks.
*/}}
{{- define "gpustack.validateMultiVendor" -}}
{{- $vendors := include "gpustack.workerVendors" . | fromJsonArray -}}
{{- if gt (len $vendors) 1 -}}
{{- $overrides := .Values.worker.gpuVendorOverrides | default dict -}}
{{- $base := include "gpustack.workerBaseNodeSelectorJson" . | fromJson -}}
{{- $seenSelectors := dict -}}
{{- range $vendor := $vendors -}}
  {{- $sel := dict -}}
  {{- with index $overrides $vendor -}}
    {{- with .nodeSelector -}}{{- $sel = . -}}{{- end -}}
  {{- end -}}
  {{- if not $sel -}}
    {{- fail (printf "worker.gpuVendorOverrides.%s.nodeSelector is required in multi-vendor mode" $vendor) -}}
  {{- end -}}
  {{- range $k, $_ := $sel -}}
    {{- if hasKey $base $k -}}
      {{- fail (printf "worker.gpuVendorOverrides.%s.nodeSelector key %q also appears in the worker/global base nodeSelector; the CPU DaemonSet would simultaneously require and forbid it" $vendor $k) -}}
    {{- end -}}
  {{- end -}}
  {{- $sig := $sel | toJson -}}
  {{- if hasKey $seenSelectors $sig -}}
    {{- fail (printf "worker.gpuVendorOverrides.%s.nodeSelector is identical to %s — vendor DaemonSets would target the same nodes and podAntiAffinity would leave one Pending forever" $vendor (index $seenSelectors $sig)) -}}
  {{- end -}}
  {{- $_ := set $seenSelectors $sig $vendor -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Union of label keys across every configured vendor's nodeSelector override.
Consumed by the CPU DaemonSet as `DoesNotExist` matchExpressions so it
avoids nodes belonging to any GPU vendor. Returns a JSON-encoded list.
*/}}
{{- define "gpustack.cpuExclusionKeysJson" -}}
{{- $vendors := include "gpustack.workerVendors" . | fromJsonArray -}}
{{- $overrides := .Values.worker.gpuVendorOverrides | default dict -}}
{{- $keys := list -}}
{{- $seen := dict -}}
{{- range $vendor := $vendors -}}
  {{- with index $overrides $vendor -}}
    {{- with .nodeSelector -}}
      {{- range $k, $_ := . -}}
        {{- if not (hasKey $seen $k) -}}
          {{- $_ := set $seen $k true -}}
          {{- $keys = append $keys $k -}}
        {{- end -}}
      {{- end -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- $keys | sortAlpha | toJson -}}
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
