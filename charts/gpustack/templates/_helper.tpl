{{/* vim: set filetype=mustache: */}}
{{- if or (and .Values.server.ingress.tls.cert (not .Values.server.ingress.tls.key)) (and .Values.server.ingress.tls.key (not .Values.server.ingress.tls.cert)) }}
{{ fail "Both server.ingress.tls.cert and server.ingress.tls.key must be set together or both be empty." }}
{{- end }}
{{- if gt (int .Values.server.replicas) 1 }}
{{- if not .Values.server.externalDatabaseURL }}
{{ fail "server.externalDatabaseURL is required when server.replicas > 1." }}
{{- end }}
{{- end }}


{{ define "gpustack.imageTag" -}}
{{ default (printf "v%s" .Chart.AppVersion) .Values.image.tag -}}
{{ end -}}


{{ define "system_default_registry" -}}
{{ if .Values.systemDefaultContainerRegistry -}}
{{ trimSuffix "/" .Values.systemDefaultContainerRegistry }}/
{{- end -}}
{{ end -}}


{{ define "gpustack.image" -}}
{{ printf "%s%s" (include "system_default_registry" .) .Values.image.repository -}}
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
{{ printf "%s%s" (include "system_default_registry" .) .Values.higressPlugins.image.repository }}:{{ required "higressPlugins.image.tag is required" .Values.higressPlugins.image.tag }}
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
