# Gotchas

Behavior that commonly trips up agents driving GPUStack v2.3.0 through its API.

## A server without a worker has no cluster

A server started without `--enable-worker` and without a registration token has no cluster, so
`/v2/clusters` is empty. That is the normal starting state, not a permission problem. Create a cluster,
then register a worker before deploying.

## Always send `cluster_id` on create

`cluster_id` is optional in the schema, but creating a model without it can fail with
`500 Failed to create model`. Pick the cluster from `/v2/clusters` and send it.

## Organization keys see clusters they cannot deploy to

A key created inside an organization also sees clusters shared with that organization, including the
default organization's clusters. Deploying to one of those returns
`403 The selected cluster does not belong to the current organization`. Use `GET /v2/clusters?mine=true`
to list the clusters the organization owns.

## The model row lags; read the instances

The response to a PUT still shows the model's previous `state`, and the model row reports `running` as
soon as one replica is ready. Decide whether a deploy, scale or restart worked from
`/v2/model-instances?model_id=ID`: the number of instances in `running` should equal `replicas`. For a
model with `roles`, `replicas` is only an on/off switch; check `role_status` instead, where each role's
`ready` should equal its `desired`.

## A partial PUT can change `distributed_inference_across_workers`

PUT keeps most fields you leave out, but `distributed_inference_across_workers` is recomputed from the
request body whenever the body does not include it: `true` if the body looks like a vLLM, SGLang or
MindIE model, `false` otherwise. For example, a partial PUT to a GGUF model on a `Custom` backend that
leaves out `backend` and `huggingface_filename` turns the flag on. GET the model and PUT it back whole,
or send `distributed_inference_across_workers` explicitly.

## A PUT does not restart running instances

Changing `run_command`, `backend_parameters`, `env` or the image saves the new configuration, but running
instances keep the old one. `POST /v2/models/{id}/restart` applies it.

## A stopped model shows `pending`

With `replicas: 0` the model row reports `state: pending` and there are no instances. That is the stopped
state, not a stuck deploy.

## Route target changes replace the whole set

Both `PUT /v2/model-routes/{id}` with `targets` and `POST /v2/model-routes/{id}/add-targets` make the list
you send the route's complete target set. Existing targets you leave out are deleted, and an entry without
an `id` creates a new target even for a model that is already on the route. `add-targets` takes a bare
JSON list (`[{"model_id": 1}]`, not `{"targets": [...]}`). List the current targets first and send back
the ones to keep with their `id`.

## Routes you create outlive the model

Deleting a model removes the route created by `enable_model_route` (its `created_model_id` is the model's
id) unless other targets still use it. A route created with `POST /v2/model-routes` has no
`created_model_id` key at all, stays with no targets, and answers `503 No running instances available`.
Delete it explicitly.

## Deletes reach the gateway a few seconds late

Right after a model is deleted, its route can still answer requests for a few seconds even though the
instances are gone and `/v1/models` no longer lists it. Retry before deciding a delete did not work.

## No route, no `/v1`

A running model without a route answers `404 Model not found` on `/v1`. Add a route with
`POST /v2/model-routes`; there is no need to recreate the model. A route created this way has empty
`categories`, unlike one created by `enable_model_route`.

## Invalid route names fail at create

With `enable_model_route: true`, the model name must be a valid route name: it starts with a letter,
ends with a letter or digit, and contains only letters, digits, `_`, `-`, `.` and `:`. In v2.3.0 an
invalid one, such as a purely numeric name, returns `500 Failed to create model` instead of a validation
error.

## Custom backends report their own model name

An OpenAI-compatible server started by a `Custom` backend returns whatever model name it chooses; llama.cpp
returns the GGUF file path. Pass the GPUStack name in `run_command` (for llama.cpp,
`--alias {{model_name}}`) so the `model` field in responses matches the route name.

## Failed instances retry on their own

With `restart_on_error` (the default), an instance in `error` is restarted with growing back-off (roughly
doubling, at most 5 minutes apart) and its `restart_count` goes up. A configuration error will not fix
itself: read the instance log and correct the model instead of waiting.

## A listed catalog spec may not run on your cluster

`/v2/model-sets/{id}/specs?cluster_id=N` can return specs the cluster cannot run, such as a vLLM spec on
a cluster with no GPUs. Only `POST /v2/model-evaluations` tells you: check `compatible`, and read
`scheduling_messages` for the reason (for example `No backend versions are available for CPU device`).

## Some backend flags belong to GPUStack

GPUStack assigns the port, API key and served model name. Built-in backends reject `--port`, `--api-key`
and `--served-model-name` in `backend_parameters` (`400 Setting the port using --port is not supported`);
`Custom` backends are not checked, so leave them out of a custom `run_command` too and use the
`{{port}}` and `{{model_name}}` placeholders instead.

## Key scope and organization role

An `inference`-only key gets `403` on `/v2`. Managing models, instances and routes also needs the key's
user to be an owner of the organization (or a platform admin); otherwise the response is
`403 Insufficient organization role`.
