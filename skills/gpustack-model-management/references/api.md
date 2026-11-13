# GPUStack model management API reference

Endpoints used by the `gpustack-model-management` skill, as of GPUStack v2.3.0. All paths are relative to
the server URL. Every request needs `Authorization: Bearer <api key>` (or `X-API-Key: <api key>`).

- `/v2` calls need a key with `management` (or `*`) scope; `/v1` calls need `inference` (or `*`).
- Models, model instances, model routes, route targets and model evaluations also need the key's user to
  be an owner of the organization, or a platform admin.
- An admin key created with no organization sees all organizations; `X-Organization-Id: ORG_ID` scopes a
  request to one.

List endpoints accept `page`, `perPage` (default 100) and `sort_by`, and return
`{"items": [...], "pagination": {"page", "perPage", "total", "totalPage"}}`.

## Clusters and workers

| Method | Path | Notes |
|---|---|---|
| GET | `/v2/clusters` | Filters: `name`, `search`, `mine=true` (only clusters the caller's organization owns, i.e. the ones it can deploy to). Fields: `id`, `name`, `state`, `workers`, `ready_workers`, `gpus`, `models`. |
| GET | `/v2/clusters/{id}` | |
| POST | `/v2/clusters` | Required: `name`. `provider` defaults to `Docker`. |
| GET | `/v2/clusters/{id}/registration-token` | Token for registering a worker. |
| GET | `/v2/workers` | Filters: `cluster_id`, `name`, `search`. `state`: `ready`, `not_ready`, `unreachable`, `pending`, `provisioning`, `initializing`, `deleting`, `error`, `maintenance`; `state_message` explains a non-ready state. |
| GET | `/v2/workers/{id}` | |

## Models

| Method | Path | Notes |
|---|---|---|
| GET | `/v2/models` | Filters: `state`, `search`, `categories`, `cluster_id`, `backend`. |
| GET | `/v2/models/{id}` | |
| POST | `/v2/models` | Create. See fields below. `409` if the name is taken. |
| PUT | `/v2/models/{id}` | Update. `name` and `source` are required. Fields left out are mostly kept, but `distributed_inference_across_workers` is recomputed unless sent. No PATCH. |
| DELETE | `/v2/models/{id}` | Removes instances and route targets, and the auto-created route unless other targets use it. Returns `200` with a `null` body. |
| POST | `/v2/models/{id}/restart` | Replaces every instance. Returns `{"restarted", "deleted_instances": [names], "message", ...}`; `restarted: false` when there are no instances; `409` while a previous restart is still bringing the model up. |
| GET | `/v2/models/{id}/instances` | Same rows as `/v2/model-instances?model_id={id}`. |

Model `state`: `pending` (also the stopped state, with `replicas: 0`), `partial`, `running`, `error`.
`running` means at least one replica is ready. The create response has `state: null`. Use instance states
to decide whether a change took effect.

### Create fields

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Unique per organization. With `enable_model_route`, must also be a valid route name. |
| `source` | yes | `huggingface`, `model_scope` or `local_path`. |
| `huggingface_repo_id` / `model_scope_model_id` / `local_path` | yes, matching `source` | |
| `huggingface_filename` / `model_scope_file_path` | no | Select one file from the repo, e.g. a GGUF. |
| `cluster_id` | send it | See gotchas. |
| `replicas` | no | Default `1`; `0` creates the model stopped. |
| `backend` | no | `vLLM`, `SGLang`, `MindIE`, `VoxBox`, `Custom`, or any other backend enabled on the server (`GET /v2/inference-backends`). |
| `backend_version` | no | Not together with `image_name` on built-in backends. |
| `backend_parameters` | no | List of strings. Built-in backends reject `--port`, `--api-key`, `--served-model-name`. |
| `image_name`, `run_command` | no | For `Custom`. `run_command` placeholders: `{{model_path}}`, `{{port}}`, `{{model_name}}`, `{{worker_ip}}`, `{{gpu_count}}`, `{{gpu_ids}}`. |
| `categories` | no | `llm`, `embedding`, `image`, `reranker`, `speech_to_text`, `text_to_speech`, `unknown`. |
| `enable_model_route` | no | Create a route named after the model. |
| `worker_selector` | no | Map of worker labels, e.g. `{"gpu": "a100"}`. |
| `env` | no | Map of environment variables for the backend container. |
| `cpu_offloading` | no | Allow CPU memory; needed for CPU-only inference with a Custom backend. |
| `distributed_inference_across_workers` | no | Defaults to `true` for vLLM, SGLang and MindIE, `false` otherwise. |
| `restart_on_error` | no | Default `true`. |

## Model instances

| Method | Path | Notes |
|---|---|---|
| GET | `/v2/model-instances` | Filters: `model_id`, `worker_id`, `cluster_id`, `state`, `search`. |
| GET | `/v2/model-instances/{id}` | |
| GET | `/v2/model-instances/{id}/logs` | Text, served as `application/octet-stream`. GPUStack's provisioning lines, then the backend container log; `tail=N` limits the container part. |
| DELETE | `/v2/model-instances/{id}` | A replacement is created to keep `replicas`. |

Instance `state`: `pending`, `analyzing`, `scheduled`, `initializing`, `downloading`, `starting`,
`running`, `error`, `unreachable`. Fast deployments skip some states. Also `state_message`, `worker_name`,
`restart_count`.

## Catalog

| Method | Path | Notes |
|---|---|---|
| GET | `/v2/model-sets` | Filters: `search`, `categories`. Each set has `size` and `size_unit` (`M`, `B` or `T` parameters). |
| GET | `/v2/model-sets/{id}/specs` | Deployable specs, optionally filtered with `cluster_id`. A listed spec is not guaranteed to run on that cluster. |
| POST | `/v2/model-evaluations` | Body `{"cluster_id": N, "model_specs": [spec, ...]}`. Returns `{"results": [{"compatible", "compatibility_messages", "scheduling_messages", "default_spec", "error", "error_message", ...}]}`. The specific reason for `compatible: false` is usually in `scheduling_messages`. |

A spec is almost a create body: drop `quantization`, `mode`, `gpu_filters` and `ready_replicas`; add
`name`, `cluster_id` and optionally `enable_model_route`.

## Model routes

| Method | Path | Notes |
|---|---|---|
| GET | `/v2/model-routes` | `targets` and `ready_targets` are counts. `created_model_id` is present only on a route created by `enable_model_route`. |
| GET | `/v2/model-routes/{id}` | |
| POST | `/v2/model-routes` | `{"name": "...", "targets": [{"model_id": N}]}`. |
| PUT | `/v2/model-routes/{id}` | `name` required. Sending `targets` makes that list the complete set: entries with an `id` keep that target, entries without one create a target, unlisted targets are deleted. |
| DELETE | `/v2/model-routes/{id}` | |
| POST | `/v2/model-routes/{id}/add-targets` | Body is a bare JSON list of targets. Replaces the whole set, like PUT. |
| GET | `/v2/model-route-targets` | Filter with `route_id`. Fields: `id`, `route_id`, `model_id`, `state` (`active` when serving), `weight`. |
| GET / POST | `/v2/model-routes/{id}/access` | Who may call the route: `{"access_policy": ..., "principals": [...]}`; `principals` replaces all grants. |
| GET | `/v2/my-models` | Routes visible to the caller. |

A target has exactly one of `model_id` or `provider_id`. `weight` defaults to `0` (the target created by
`enable_model_route` for a single replica gets `100`); weights on one route must be all positive or all
zero. Route names match `^[A-Za-z](?:[A-Za-z0-9_\-.:]*[A-Za-z0-9])?$` and are unique per organization.
In v2.3.0, route write responses can report `ready_targets` a few seconds out of date (fixed on main).

## OpenAI-compatible API

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/models` | Routes with at least one ready target, limited to `allowed_model_names` if the key has them. Non-platform organizations' routes appear as `<org>/<route>`. |
| POST | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/audio/*`, `/v1/images/*`, `/v1/rerank` | `model` is the route name. |

`404 Model not found`: unknown route, or not visible to or allowed for this key. `503 No running instances
available`: route exists, no active target.

## API keys

| Method | Path | Notes |
|---|---|---|
| POST | `/v2/api-keys` | Required: `name`. Optional: `scope` (`["*"]` default, or `management` / `inference`), `allowed_model_names`, `expires_in` (seconds), `description`. The key value is returned only in this response. |
| GET | `/v2/api-keys` | |
| DELETE | `/v2/api-keys/{id}` | |
