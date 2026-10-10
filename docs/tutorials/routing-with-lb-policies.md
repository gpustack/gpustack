# Routing with LB Policies

This tutorial walks through GPUStack's route-level load-balancing policies in practice: keeping sessions sticky while still spreading load, and letting a decision service pick between a small and a large model per request. For the configuration reference, see [Model Route Management](../user-guide/model-route-management.md#load-balancing).

## Prerequisites

- A running GPUStack installation. The AI gateway that applies these policies is deployed by default with GPUStack, so a standard installation needs no extra setup.
- Two deployed models serving as route targets, e.g. `qwen3-0.6b` and `qwen3-32b` (the examples below use these).
- For the decision-service part: a Jev decision service registered on the `Model` - `Provider` page as a `TypeSafe Decision Service (Jev)` provider (see [Model Provider Management](../user-guide/model-provider-management.md)).

## Step 1: Create the Route

1. Go to `Routes` page.
2. Click `Add Route`, name it e.g. `qwen3`.
3. Select `qwen3-0.6b` and `qwen3-32b` as `Route Targets`, leaving all weights unset.
4. Click `Save`.

With no weights and no policy, the route's `Route By` is round-robin — plain round-robin between the two models.

## Step 2: Understand How Session Affinity and Least Inflight Request Interact

When policy plugins are enabled, every plugin scores every candidate target and the finisher picks the target with the best weighted sum:

```
total(candidate) = Σ over plugins ( plugin weight × normalized score )
```

Session affinity gives its top score to the session's sticky owner; least inflight request gives its top score to the idlest target. A session therefore migrates away from its owner only when the least inflight request plugin's vote gap exceeds affinity's vote gap.

With both plugins at their default weight (1), the table below shows what happens when the owner has accumulated `x` in-flight requests and every other target is idle — the condition most favorable to a flip:

| Targets (N) | x=0 | x=1 | x=2 | x=3 | x=4 | Ever flips? |
| ----------- | --- | --- | --- | --- | --- | ----------- |
| 2 | hold | tie ⚠️ | **FLIP** | **FLIP** | **FLIP** | yes |
| 3 | hold | hold | tie ⚠️ | **FLIP** | **FLIP** | yes |
| 4 | hold | hold | hold | hold | hold | only at x ≈ 5.3 |
| ≥5 | hold | hold | hold | hold | hold | **no** |

- **FLIP**: the session migrates to the next candidate.
- **tie ⚠️**: the vote gaps are exactly equal; the finisher breaks the tie randomly — the same request may or may not migrate.
- **N ≥ 5**: the least inflight request plugin's vote gap is capped at `1/(N−1) ≤ 0.25`, below affinity's gap — no amount of load imbalance can break the stickiness by itself.

To make load matter on larger routes, rebalance the weights — see [How Weights Work](#how-weights-work) for the flip thresholds under scaled weight ratios.

## Step 3: Enable Session Affinity and Least Inflight Request

1. Edit the route from Step 1.
2. Enable `Session Affinity` — accept the default session-key chain (`session-id` header, `x-client-request-id` header, then the `prompt_cache_key` body field) or specify your own.
3. Enable `Least Inflight Request` with its default weight.
4. Save and verify the route's `Route By` is `Policy`.

Sessions are now sticky: requests carrying the same session key stay on the same target — prompt caches stay warm — while a heavily loaded owner eventually sheds sessions to the idle target, per the table above.

Note which key sources actually apply on each path: header keys (e.g. `session-id`) work everywhere, but body keys such as `prompt_cache_key` only take effect on `/responses` and `/messages` requests — on `/chat/completions`, which has no standard session body field, only the header keys of the chain are considered. Prefer a header key for chat traffic.

## Step 4: Add Decision Service Routing

Now make the route model-aware: instead of always pinning a session to whichever model it started on, ask the decision engine which model each request deserves.

1. Edit the route and enable `Decision Service Routing`:
    - `Decision Service` — select the registered provider.
    - `Decision Model` — e.g. `jev-latest`.
    - `Instructions` — "Pick the smallest model that can handle the request well. Simple, short or factual requests go to the fast model; complex reasoning, math or code requests go to the large model."
    - `Model Criteria` — one entry per target (or click `Generate from targets` and refine):
        - `qwen3-0.6b` — "Very fast, low-cost 0.6B model. Handles greetings, simple factual questions, formatting and short chit-chat. Fails on multi-step reasoning, math and code."
        - `qwen3-32b` — "High-capability 32B reasoning model. Handles complex reasoning, math, code generation and long-context analysis. Slower and more expensive per request."

2. Save, then send two authenticated requests against the route name `qwen3` (replace `$GPUSTACK_API_KEY` with an API key created on the `API Keys` page):

```bash
curl http://my-gpustack/v1/chat/completions \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "session-id: demo-session" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Write a one-line greeting card."}]}'
```

```bash
curl http://my-gpustack/v1/chat/completions \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "session-id: demo-session" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Prove that there are infinitely many primes, then implement the proof sketch in Python."}]}'
```

The first request is routed to `qwen3-0.6b`, the second to `qwen3-32b`.

## How Weights Work

Weights appear in two different places, with different meanings:

### Target Weight (traffic share)

In `Target Weight` mode, a target's weight is that target's **total** share of the traffic — regardless of how many instances it runs. GPUStack scales the per-instance weights so the split always matches the configured target weights:

- Target A (weight 100, 1 instance) and Target B (weight 100, 2 instances) split traffic 50/50 between the two targets — not 1/3 per instance.
- Scaling a target up from 1 to 2 instances does not change its share; each instance then receives half of it.

Example with `qwen3-0.6b` and `qwen3-32b`: weight 100 on `qwen3-0.6b` and weight 300 on `qwen3-32b` sends 25% of the requests to the small model and 75% to the large one, whatever their replica counts.

### Policy Weight (influence in the sum)

Each policy plugin's `Weight` dials how much it counts in the weighted sum of `Policy` mode. It is a positive number and fractions are meaningful. Unset `Weight` keeps each plugin's built-in default (1 for Session Affinity and Least Inflight Request, 10 for Decision Service Routing).

The calculation rules for the session-affinity × least-inflight-request balance, with the owner at `x` inflight requests and every other target idle:

- session affinity's vote gap: `affDiff = 0.5 / (2·(1 − 2⁻ᴺ))`
- least inflight request's vote gap: `loadDiff = x / ((1+x)·(N−1) + 1)`
- the session flips when `w_leastInflight × loadDiff > w_affinity × affDiff`

So scaling the weights shifts the flip threshold: the table below gives the minimum owner inflight `x` that flips a session, per weight ratio (Least Inflight Request : Session Affinity) and target count N:

| N | 1:1 (default) | 1.5:1 | 2:1 |
| - | ------------- | ----- | --- |
| 2 | 2 | 1 | 1 |
| 3 | 3 | 1 | 1 |
| 4 | 6 | 2 | 1 |
| 5 | never | 3 | 2 |
| 6 | never | 7 | 3 |
| 7 | never | never | 4 |
| 8 | never | never | 9 |

Reading the table:

- At the default 1:1 ratio, routes with 5 or more targets can never flip — least inflight request's gap is capped below affinity's.
- Halving Session Affinity's `Weight` (a 2:1 ratio) makes load matter from `x` = 2–4 on routes of up to 8 targets; equivalently, doubling Least Inflight Request's `Weight` has the same effect, since only the ratio matters.
- Exact ties (e.g. 1:1 at N=2/x=1) are broken randomly by the finisher.

## How the Policies Divide the Work

All enabled policies contribute to the same weighted sum, each doing one job:

- **Decision Service Routing** decides *which model* fits the request (its default `Weight` of 10 reflects that it is usually the dominant voice);
- **Session Affinity** keeps a session on its current model, preserving prompt-cache locality;
- **Least Inflight Request** spreads requests across the instances of whichever model wins.

When a model is deployed as multiple instances, the decision narrows the choice to that model and Least Inflight Request picks the instance within it.
