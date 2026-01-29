{{- if not (lookup "apiextensions.k8s.io/v1" "CustomResourceDefinition" "" "wasmplugins.extensions.higress.io") }}
{{ fail "Higress CRD wasmplugins.extensions.higress.io is not installed! Please install Higress first." }}
{{- end }}

{{ define "chart_labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end -}}

{{ define "gpustack_service_full_name" -}}
{{ printf "server.%s.svc.%s" .Values.gpustackNamespace .Values.clusterDomain }}
{{- end -}}

{{ define "gpustack_service_port_suffix" -}}
{{- if eq .Values.gpustackAPIPort "80" -}}
{{ "" }}
{{- else -}}
{{ printf ":%s" .Values.gpustackAPIPort }}
{{- end -}}
{{- end -}}
