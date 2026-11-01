---
name: gpustack-model-management
description: Deploy, scale, stop, route, troubleshoot and delete models on a GPUStack server through its REST API.
---

# GPUStack model management

Manage model deployments on a GPUStack server with its REST API: inspect the cluster, deploy a model
(from a Hugging Face / ModelScope repo or from the catalog), start, stop and scale it, give it an
OpenAI-compatible route, recover it when an instance fails, and delete it. Everything here is plain
HTTP, so any agent that can run `curl` can use it.

Tested against GPUStack v2.3.0. Endpoint details and payloads: [`references/api.md`](references/api.md).
Behavior that surprises agents: [`references/gotchas.md`](references/gotchas.md). Using this skill from
OpenClaw: [`references/openclaw.md`](references/openclaw.md).

## Setup

Two environment variables:

```bash
export GPUSTACK_URL=http://your-gpustack-server     # no trailing slash
export GPUSTACK_API_KEY=gpustack_...                # created in the UI under API Keys, or POST /v2/api-keys
```

Send the key on every request as `Authorization: Bearer $GPUSTACK_API_KEY` (an `X-API-Key` header also
works). What the key can do depends on its scope and on who created it:

- Scope `management` (or `*`) is needed for every `/v2/...` call in this skill, and `inference` (or `*`)
  for `/v1/...` calls from OpenAI-compatible clients. An `inference`-only key gets `403` on `/v2`.
- Models, instances and routes also require the key's user to be an **owner** of the organization (or a
  platform admin). Otherwise these calls return `403 Insufficient organization role`.
- A platform admin key created with no organization selected sees every organization's resources. Send
  `X-Organization-Id: ORG_ID` to act inside one organization.
- A key created inside an organization sees that organization's resources, plus clusters shared with it.
  It can only deploy to clusters its organization owns; list those with `GET /v2/clusters?mine=true`.

The examples below use this shorthand:

```bash
gs() { curl -sS -H "Authorization: Bearer $GPUSTACK_API_KEY" -H "Content-Type: application/json" "$@"; }
```

List responses look like `{"items": [...], "pagination": {"page", "perPage", "total", "totalPage"}}`.
When a step says "wait", poll every few seconds; downloads of large models can take many minutes.

## 1. Inspect

```bash
gs "$GPUSTACK_URL/v2/clusters"                        # id, name, state, workers, ready_workers
gs "$GPUSTACK_URL/v2/workers"                         # state: ready | not_ready | unreachable | ...
gs "$GPUSTACK_URL/v2/models"                          # id, name, replicas, ready_replicas, state
gs "$GPUSTACK_URL/v2/model-instances?model_id=ID"     # one row per replica: id, state, restart_count
gs "$GPUSTACK_URL/v2/model-routes"                    # id, name, targets (a count), created_model_id
gs "$GPUSTACK_URL/v2/model-route-targets?route_id=ROUTE_ID"  # the targets themselves
```

You need a `cluster_id` with at least one `ready` worker before deploying. A server started without a
worker or a registration token has no cluster at all; create one in the UI (Clusters page) or with
`POST /v2/clusters`, then register a worker.

## 2. Deploy a model

```bash
gs -X POST "$GPUSTACK_URL/v2/models" -d '{
  "name": "qwen3-8b",
  "source": "huggingface",
  "huggingface_repo_id": "Qwen/Qwen3-8B",
  "backend": "vLLM",
  "cluster_id": 1,
  "replicas": 1,
  "enable_model_route": true
}'
```

- Required: `name`, `source` (`huggingface`, `model_scope` or `local_path`) and the matching field
  (`huggingface_repo_id`, `model_scope_model_id` or `local_path`). Add `huggingface_filename` to pick one
  file, e.g. a single GGUF.
- Always send `cluster_id`.
- `enable_model_route: true` creates an OpenAI-compatible route with the same name as the model. Without
  it the model runs but `/v1` answers `404 Model not found` until you add a route (section 6).
- Backend flags go in `backend_parameters` as a list of strings, e.g. `["--max-model-len=32768"]`.
  GPUStack sets the port, API key and served model name itself; built-in backends reject `--port`,
  `--api-key` and `--served-model-name`.
- `backend` is `vLLM`, `SGLang`, `MindIE`, `VoxBox`, `Custom`, or the name of any other inference backend
  enabled on the server (`GET /v2/inference-backends`).

For a backend that is not enabled, use `"backend": "Custom"` with `image_name` and a `run_command` that
uses the `{{model_path}}`, `{{port}}` and `{{model_name}}` placeholders. Make the server report the model
under its GPUStack name, or clients see a file path in the response's `model` field. For example, a GGUF
served by llama.cpp on CPU-only workers:

```json
{
  "name": "qwen2.5-0.5b",
  "source": "huggingface",
  "huggingface_repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
  "huggingface_filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
  "backend": "Custom",
  "image_name": "ghcr.io/ggml-org/llama.cpp:server",
  "run_command": "-m {{model_path}} --host 0.0.0.0 --port {{port}} --alias {{model_name}} -c 4096",
  "cpu_offloading": true,
  "cluster_id": 1,
  "enable_model_route": true
}
```

`-c` caps the context size; without it llama.cpp uses the model's full training context, which can
exhaust a small worker's memory. GPUStack also offers llama.cpp as a community inference backend that an
admin can add under Inference Backends.

**Confirm:** wait until every replica's instance is `running`:

```bash
gs "$GPUSTACK_URL/v2/model-instances?model_id=ID"
```

Instances move through `pending → analyzing → scheduled → initializing → downloading → starting →
running`, though fast deployments skip some of these. `error` means the backend failed to start; see
section 5. The create response has `state: null`; ignore it. A name already in use returns `409`.

## 3. Deploy from the catalog

1. Find a model set: `gs "$GPUSTACK_URL/v2/model-sets?search=qwen&categories=llm"`. Each set has a
   `size` with a `size_unit` (`M`, `B` or `T` parameters) to help pick one that fits.
2. Get its specs: `gs "$GPUSTACK_URL/v2/model-sets/SET_ID/specs?cluster_id=1"`. A spec being listed does
   not mean it can run on that cluster.
3. Check a spec against the cluster before deploying:

   ```bash
   gs -X POST "$GPUSTACK_URL/v2/model-evaluations" -d '{"cluster_id": 1, "model_specs": [SPEC]}'
   ```

   Each entry in `results` has `compatible`. When it is `false`, the specific reason is usually in
   `scheduling_messages` (for example `No backend versions are available for CPU device`);
   `compatibility_messages` only gives the summary.
4. Deploy: take the spec, remove `quantization`, `mode`, `gpu_filters` and `ready_replicas`, add `name`,
   `cluster_id` and `enable_model_route: true`, and `POST /v2/models` as in section 2.

## 4. Stop, start and scale

There are no start/stop endpoints. Set `replicas` with a PUT: `0` stops the model, `N` runs N replicas.
The safe way is to GET the model, change `replicas`, and PUT the whole object back:

```bash
gs "$GPUSTACK_URL/v2/models/ID" > model.json
# edit model.json: set "replicas"; remove id, state, state_message, ready_replicas, created_at, updated_at
gs -X PUT "$GPUSTACK_URL/v2/models/ID" -d @model.json
```

- `name` and `source` (with its repo field) are required in every PUT.
- A shorter PUT that sends only a few fields keeps most of the others, but it recomputes
  `distributed_inference_across_workers` from what the body contains and can flip it. If you send a
  partial body, include `backend` and `distributed_inference_across_workers` explicitly.
- A change to how the model runs (`run_command`, `backend_parameters`, `env`, image) is saved but does
  not touch running instances. Restart the model (section 5) to apply it.

**Confirm:** the number of `running` instances equals `replicas`. A stopped model has no instances and
its model row shows `state: pending`; that is normal. The PUT response still shows the old `state`.

## 5. Restart and recover

- Restart every replica: note the current instance ids, then
  `gs -X POST "$GPUSTACK_URL/v2/models/ID/restart"`. The response lists the replaced instances by name in
  `deleted_instances`; the instance list is briefly empty, then new instances appear.
  **Confirm:** instances with new ids reach `running`.
  - With no instances to restart it returns `200` with `restarted: false`.
  - While a restart is still bringing the model up, another restart returns `409`; wait for `running`.
- Replace one replica: `gs -X DELETE "$GPUSTACK_URL/v2/model-instances/INSTANCE_ID"`; GPUStack creates a
  new one to keep `replicas`.
- Read why an instance failed: its `state_message`, and
  `gs "$GPUSTACK_URL/v2/model-instances/INSTANCE_ID/logs?tail=200"`. The log is text (served as
  `application/octet-stream`); `tail` limits the backend container's lines, and GPUStack's own
  provisioning lines come first.
- With `restart_on_error` (on by default) the worker retries a failed instance with growing back-off, up
  to 5 minutes apart, and increments `restart_count`. A rising `restart_count` means it keeps failing the
  same way: read the log and fix the model's configuration rather than waiting.
- After a node outage, bring models back one at a time and confirm each is `running` before the next.
  Several large models starting at once can exhaust GPU memory and fail each other.

## 6. Routes

A model deployed without `enable_model_route` can be routed later, without redeploying:

```bash
gs -X POST "$GPUSTACK_URL/v2/model-routes" -d '{"name": "qwen3-8b", "targets": [{"model_id": ID}]}'
```

Route names start with a letter, end with a letter or digit, and use only letters, digits, `_`, `-`, `.`
and `:` in between.

To change which models a route serves:

1. List its current targets: `gs "$GPUSTACK_URL/v2/model-route-targets?route_id=ROUTE_ID"`. The route
   itself only reports a count.
2. PUT the route with `name` and the complete `targets` list. Keep each existing target by sending its
   `id` (`{"id": 13, "model_id": 11}`). An entry without `id` creates a new target, and any existing target
   you leave out is deleted. Weights on one route must be all zero or all positive.
3. **Confirm:** list the targets again and check they match.

`POST /v2/model-routes/ROUTE_ID/add-targets` also replaces the whole target set, despite its name. Write
responses for routes in v2.3.0 can show `ready_targets` counts that are a few seconds old.

**Confirm a new route:** its name appears in `gs "$GPUSTACK_URL/v1/models"`.

## 7. Use the model from a client

GPUStack serves an OpenAI-compatible API at `$GPUSTACK_URL/v1`. Point any OpenAI client at it, with an API
key whose scope includes `inference`, and use the route name as the model:

```bash
gs "$GPUSTACK_URL/v1/chat/completions" -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "hello"}]}'
```

- `/v1/models` lists routes that have at least one ready target. A key with `allowed_model_names` only
  sees, and can only call, those models.
- Routes owned by an organization other than the platform one are addressed as `<org>/<route>`.
- `404 Model not found`: no such route, or not visible to this key. `503 No running instances
  available`: the route exists but nothing behind it is running.

## 8. Delete

```bash
gs -X DELETE "$GPUSTACK_URL/v2/models/ID"
```

This removes the model's instances and route targets. The route created by `enable_model_route` (the one
whose `created_model_id` is this model) is removed too, unless other targets still use it. A route you
created yourself has no `created_model_id` and stays behind with no targets; delete it with
`gs -X DELETE "$GPUSTACK_URL/v2/model-routes/ROUTE_ID"`.

**Confirm:** no instances remain for the model, `/v1/models` no longer lists it, and a request to its
route name returns `404` or `503`. Requests can still be served for a few seconds after the delete, so
retry before concluding the delete failed.

## Hard rules

- Verify after every change, using the **Confirm** step for that workflow. A `200` only means the request
  was accepted.
- Judge a model by its instances, not by its model row.
- Always send `cluster_id` on create. Prefer GET → edit → PUT of the whole model over a partial PUT.
- Restart after changing how a model runs; a PUT alone does not apply it.
- Make one change at a time when recovering, and confirm it before the next.
- Never change a route's targets without listing them first and sending back the ones to keep with their
  `id`.
- Do not put `--port`, `--api-key` or `--served-model-name` in `backend_parameters`.
