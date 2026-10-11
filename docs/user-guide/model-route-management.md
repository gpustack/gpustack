# Model Route Management

GPUStack provides model route management capabilities. Through model routes, you can implement model aliases, traffic distribution, disaster recovery, and unified entry for both public and private models.

## Create Route

1. Go to `Routes` page.
2. Click `Add Route`.
3. Fill the route `Name` as the serving model name.
4. Select one or more `Route Targets`.
5. Click the `Save` button.

## Manage Route Targets

1. Go to `Routes` page.
2. Unfold the model route you want to manage targets for.
3. Click the `Delete` button in `Operations` column for the target you don't want to keep.
4. Click the `Edit` button in the `Operations` column. On the edit route page, add or remove targets for the model route, or adjust the traffic weight for the targets.
5. On the edit route page, select or clear the fallback route target for this route.

## Load Balancing

When a model route has multiple targets, GPUStack's gateway-side LB plugin decides which target serves each request. The routing behavior is derived from the route's configuration: the target weights and the enabled policy plugins (Session Affinity, Least Inflight Request, Decision Service Routing) together determine the `Route By` mode shown on the route.

### Route By

The `Route By` column on the route list shows how requests are distributed across the route's targets. It is derived from the configuration — the target weights and the enabled policy plugins — computed by GPUStack on every route change, never client input.

| Route By | Configuration shape | Routing behavior |
| --- | --- | --- |
| `Target Weight` | Every target has a weight greater than 0 | Split traffic across targets by weight; every target must have a weight greater than 0 |
| `Policy` | No weights, at least one policy plugin (Session Affinity, Least Inflight Request, Decision Service Routing) enabled on the route | Targets are picked by the enabled policy plugins |
| *(round-robin)* | No target has a weight and no policy enabled | Round-robin across targets |

Notes:

- Weights and policies are mutually exclusive: every target must carry a weight, or none of them may. A mixed configuration makes the route unavailable — set every target weight to greater than 0, or all to 0.
- Fallback targets never count as candidates; their weight is ignored by the mode derivation.
- The mode describes the configuration's shape, not live instance health — an all-weighted route stays `Target Weight` while its instances are down.

### Policy Plugins

Policy plugins participate in `Policy` routing: each one returns a score for every target, and the gateway picks the target with the best weighted sum. Each plugin's `Weight` (a positive number, fractional allowed) dials how much it counts in the sum; leaving it unset uses the plugin's built-in default.

#### Session Affinity

Route requests of the same session to the same target, keeping prompt caches warm. Sessions are identified by an ordered key chain.

- `Session Keys (Ordered, First Match Wins)` — ordered chain of key sources; the first one that yields a value wins. Each entry is either a request `Header` (e.g. `session-id`) or a `Body Key` (e.g. `prompt_cache_key`). At least one session key is required when session affinity is enabled.

#### Least Inflight Request

Score targets by their number of inflight requests and prefer routing requests to the target with the fewest inflight requests.

When Session Affinity and Least Inflight Request are both enabled, session affinity rewards the sticky owner while least inflight request rewards the idle target. A session migrates away from its owner only when the least inflight request plugin's vote gap exceeds affinity's — at the default weights this happens easily on 2–3 target routes and never on routes with 5 or more targets. Tune the plugins' `Weight` to shift that balance; see [Routing with LB Policies](../tutorials/routing-with-lb-policies.md) for the numbers and worked examples.

#### Decision Service Routing

Score targets by task difficulty: each request is sent (truncated) with a model-selection question to a Jev-compatible decision service — a `TypeSafe Decision Service (Jev)` provider on the `Model` - `Provider` page — whose answer becomes a weighted vote. Decision failures, timeouts, or non-2xx responses fall back silently to the other plugins.

Settings:

- `Decision Service` — the decision service provider (required).
- `Decision Model` — route-level decision model, required when this policy is enabled. Options are read from the engines cached in the selected provider; if unset or invalid, decision routing is skipped and requests fall back to the other plugins.
- `Instructions` — extra guidance for the model-selection question sent to the decision service.
- `Model Criteria` — model name → capability description: the basis the decision service uses to pick a model. At least one criterion is required; entries can be generated from the route's targets.
- `Weight` — contribution weight in the weighted sum when combined with other policies; unset uses the built-in default (10).

See [Routing with LB Policies](../tutorials/routing-with-lb-policies.md) for an end-to-end example routing between a small and a large model.

Fallback targets (`fallback_status_codes`) are handled by the separate fallback plugin and do not participate in LB candidate selection. A route can combine LB candidate splitting with fallback targets for disaster recovery.

## Authorize Route Access

1. Go to `Routes` page.
2. Find the route for which you want to change the authorization setting.
3. Click the `Access Settings` button in the `Operations` column.
4. Change the `Access Scope` as needed.
5. For the `Allowed Users` scope, select the users you want to authorize for this route and click `>` to confirm.
6. Click the `Save` button.

## Edit Route

1. Go to `Routes` page.
2. Find the route you want to edit.
3. Click the `Edit` button in the `Operations` column.
4. Modify name, model category, description and route targets as needed.
5. Click the `Save` button.

## Delete Route

1. Go to `Routes` page.
2. Find the route you want to delete.
3. Click the `Delete` button in the `Operations` column.
4. Confirm the deletion.
