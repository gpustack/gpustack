# PD Disaggregation

Prefill/decode (PD) disaggregation splits a deployment into roles that run
separately: **prefill** computes the prompt's KV cache, **decode** generates
tokens from it, and a **router** sits in front and carries each request across
the two. The KV cache moves between them over the network rather than being
recomputed.

The two phases have opposite appetites. Prefill is compute-bound and wants wide
tensor parallelism; decode is memory-bandwidth-bound and wants many concurrent
sequences. A single deployment has to choose one set of parameters for both.
Disaggregating lets each side be sized and tuned for what it actually does, and
lets the ratio between them follow the workload — a prompt-heavy workload wants
more prefill, a long-generation workload more decode.

It also costs something real. A request now crosses the network mid-flight,
there are three roles to keep alive instead of one, and a group is only as
available as its least available role. Disaggregation is worth it when the two
phases genuinely conflict, and not otherwise.

## When it helps, and when it does not

Three questions decide it:

1. **Do prefill and decode want different parameters?** If a single set of
   engine parameters serves both phases acceptably, disaggregation adds
   machinery without adding throughput.
2. **Is the prompt-to-generation ratio far from 1:1?** The gain comes from
   sizing the two sides independently, which only pays when they are
   differently loaded.
3. **Is the KV transfer fast enough?** The prompt's KV cache crosses the network
   on every request. On a slow or shared link, transfer time replaces the
   prefill time you saved.

A single GPU, or a workload whose prompts and generations are both short, will
usually be faster without it.

## Deploying a disaggregated model

1. Deploy a model as usual, then expand the `Disaggregation` section and turn it
   on.
2. Choose a **PD mode**. A mode is one complete recipe — engine, KV connector
   and router — so the connector, its handshake variables and its ports are all
   decided by this one choice and none of them appear in the form. The available
   modes depend on the backend; a mode built for another engine is shown
   disabled with the reason.
3. Set each role's **replicas**. This is the ratio: 2 prefill and 1 decode is a
   2P1D group.
4. Override per-role settings where the two sides differ. Every field left
   inheriting takes the deployment-level value, so only the differences need
   typing.
5. Save.

The router is managed for you: its image, command, peer addresses and health
path all come from the mode's recipe, so the form does not ask for them. What
it does ask for is what a recipe cannot know — the router's own arguments and
environment — and the connection arguments the platform derived are seeded
into that same list, so you can see what was rendered rather than trust it. A
row you leave untouched is dropped again on save, and the server goes on
rendering it.

A router is structurally one replica, so there is no count to set: a second
would split the prefix cache and give the group two addresses.

If you need your own router, choose the `custom` mode and set the image and
run command on the **deployment** rather than on the router role — the role
inherits both. One image therefore carries your router beside the engine and
dispatches on the role, which is the same work as naming it in two places
without the second place to keep in sync.

## How a group starts

The order is not incidental and is worth knowing, because it is what you will
see while a group comes up:

1. The prefill and decode instances are created **together**, in one
   transaction. They are also admitted together where gang admission is
   available — a group that starts half-way is a group that cannot serve.
2. Once each of them has a running member, the **router** is created. It is
   created last because its command line contains its peers' addresses, and
   those addresses do not exist until the peers have started.
3. The group's endpoint is registered once the router is up. Only the router is
   registered — the prefill and decode members serve an OpenAI-shaped API on
   their own ports, but neither can answer a whole request alone.

A group therefore reports `Partially ready` for a while before it serves, and
that is the normal path rather than a fault.

## Reading a group's state

| What you see | What it means |
| --- | --- |
| `Pending` | No member is running yet. |
| `Partially ready` | Members are up and the group still cannot serve — a role with no ready member, or the router's endpoint not yet registered. |
| `Running` | Every role has at least one ready member and the endpoint is registered. |
| `Ratio 2:1 (currently 1:1)` | The group serves, but not at the ratio you asked for. Shown beside the state, not instead of it. |
| Clock icon / `Config changed` | The running members predate the configuration shown, and until you restart, a role cannot be scaled up. See [Applying a configuration change](#applying-a-configuration-change). |

Being short of the requested ratio is a degradation, not a state: a 3P1D group
running 2P1D still serves, just not at the throughput you sized for.

## Applying a configuration change

Editing a disaggregated deployment does **not** restart it. The change is saved,
the group is marked as running an older configuration, and nothing moves until
you restart it.

This is deliberate. Restarting members one at a time would, for a while, pair a
prefill running the new configuration with a decode running the old one — and
the engines accept that pairing. A mismatched context length, for instance,
completes its handshake, transfers KV successfully, and answers short prompts
correctly; only a prompt longer than the decode side's window fails, after the
prefill work has already been paid for. A wrong answer is worse than a
restart, so the whole group stops before any of it starts again.

Use `Restart` on the deployment to apply the change. It retires the whole group
and lets it re-form on the current configuration. Restarting only some roles is
not offered, because that is exactly the request that produces the mixed pairing
above.

Scaling changes each role's instance count without restarting the whole group,
provided no configuration changes are pending. Router behavior depends on
`router.membership_update` in `pd-modes.yaml`:

| Policy | Shipped modes | Behavior on scaling or member recovery |
| --- | --- | --- |
| `dynamic` | vLLM NIXL, SGLang Mooncake/NIXL, vLLM Ascend Mooncake | GPUStack updates the router through its membership API. |
| `restart` | vLLM Mooncake | GPUStack replaces the router after added GPU members are ready, refreshing startup peers and bootstrap metadata. Removed or restarted members trigger a refresh as soon as both sides have a usable peer. Surviving GPU instances keep running, but service is briefly unavailable and in-flight requests may fail. |
| `manual` | Custom | The operator updates router peers and bootstrap metadata and restarts the router when necessary. |

A user-supplied router run command uses manual membership management, even when
the selected recipe supports automatic updates. The deployment form and replica
editor display the policy from the catalog. Instance deletion also warns when
recovery requires router replacement or operator action.

Static router snapshots include instance identity, restart generation, HTTP
address and named ports. A replacement or restart at the same address therefore
still refreshes Mooncake bootstrap metadata. Existing routers without a snapshot
are replaced once when their peers are ready.

**With a saved change still waiting for its restart, a role cannot be scaled
up.** A new member would be built from the edited configuration and would then
run beside members that were not, which is the mixed pairing above. The scale
takes effect when you restart, along with the change itself.

Scaling **down** still applies immediately, edit pending or not: losing a
member cannot produce a mismatched pair.

## Using a shared KV cache with it

A disaggregated deployment and a shared KV cache are complementary rather than
alternatives, and GPUStack combines them for you: attach a cache service as
usual and the two connectors are folded into one configuration the engine
accepts.

Each vLLM role with a cache generates `MultiConnector[PD, Store]`. The wrapper uses
`kv_producer` on prefill and `kv_consumer` on decode. Each child retains the
role required by its transport or store.

| Mode | No cache | In-process cache | Cache service |
| --- | --- | --- | --- |
| `vllm-nixl` (NVIDIA default) | `NixlConnector` | `NixlConnector` + `LMCacheConnectorV1` | Prefill only: `NixlConnector` + LMCache; LMCache only |
| `vllm-mooncake` (NVIDIA) | `MooncakeConnector` | `MooncakeConnector` + `LMCacheConnectorV1` | `MooncakeConnector` + `MooncakeStoreConnector`; Mooncake only |
| `vllm-ascend-mooncake` | Model-selected Mooncake PD connector | PD connector + `SimpleCPUOffloadConnector` | PD connector + `AscendStoreConnector`; Mooncake only |

The in-process LMCache child uses `kv_both` on both roles. Shared LMCache is
supported on prefill only for NIXL. Its MP connector can load a cached prefix
even when MultiConnector assigns the request's load to NIXL; the second async
completion can crash decode ([vLLM #53049](https://github.com/vllm-project/vllm/issues/53049)).
NVIDIA Mooncake Store uses `kv_both` on prefill and `kv_consumer` on decode.
Ascend Store uses `kv_producer` on prefill and `kv_consumer` on decode. These
configurations follow
[vLLM 0.29 Mooncake Store](https://docs.vllm.ai/en/v0.29.0/features/mooncake_store_connector_usage/#disaggregated-prefill-decode-xpyd)
and [vLLM-Ascend 0.23 KV pool](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/kv_pool.html).

Ascend selects `MooncakeConnectorV1` for standard KV layouts, including Qwen3,
GLM-5 and DeepSeek V3.1, and `MooncakeHybridConnector` for hybrid layouts such
as DeepSeek V4, V4 Flash and V4 Pro. Selection reads the pretrained configuration,
including compressed attention and recurrent layers, rather than the deployment
name. Both roles use the same stored selection. If layout metadata is unavailable,
the hybrid connector handles the standard-layout fallback. GPUStack retains
that choice for the deployment's model inputs so metadata becoming available
after a member starts cannot make another member use a different connector.

NIXL cache composition uses the real-block contract available from vLLM 0.26.
PD recipes declare supported cache providers and connector settings; they do
not define or check cache-service releases or client-package requirements. Cache providers own
their service and client compatibility. The runner dependency remains at 0.5.4.
Both NVIDIA modes use vLLM 0.27.1 or later as the bundled runner baseline:
those CUDA runners provide `vllm-router` 0.1.15, which runs from the engine's
image. The Ascend mode uses the CANN 0.23 baseline for the same reason. These
floors describe GPUStack's bundled images, rather than the first upstream
release of each connector.

NVIDIA Mooncake with a shared Store has an additional 0.29 baseline. GPUStack's
0.29 runners include the rejected-load fix: a Store child must skip a load
assigned to the PD child. MultiConnector passes zero external tokens to an
unselected Store, and the fix checks `can_load` before issuing a pending load.
This also applies to decode-side cache hits. Stock vLLM 0.29 requires that
backport; upstream 0.30 includes it. See the
[runner fix](https://github.com/gpustack/runner/commit/f0264ed).

Attach a service separately on each role that needs it. The dropdown filters
services by mode, provider and role. The API enforces these compatibility rules.
PD engine-version ranges are advisory: selecting an out-of-range built-in engine
version shows a warning for the mode or enabled cache, but deployment and cache
injection are still allowed. Custom versions and explicit images do not show
these warnings.
Startup failures are reported through the instance's status and logs.

Auto currently selects a runner by backend, accelerator runtime and platform;
it does not filter versions by the PD mode or cache composition ranges. SGLang
attaches LMCache or MeshFusion through flags and a config file rather than a
vLLM MultiConnector.

NVIDIA Mooncake requires its transfer-engine runtime and RDMA support in the
inference image. GPUStack runner images also expose Mooncake transfer/expiry
metrics; transfer counters are read on prefill. GPUStack allocates a unique
prefill bootstrap port and passes
it to the router. With vllm-router 0.1.15, membership changes restart the router
to refresh bootstrap metadata; they may briefly interrupt traffic.

## Networking requirements

### On Kubernetes: what the Pods ask the cluster for

A group's members need host networking, hostPort, host IPC and device mounts —
all four at once, which is more than the `baseline` Pod Security Admission
level allows. GPUStack labels the namespaces it owns accordingly; see
[Namespaces and Pod Security Admission](cluster-management.md#namespaces-and-pod-security-admission)
for what that means on a cluster you brought yourself, including what it does
*not* cover.

### The KV transfer interface

The KV cache crosses the network on every request, and the interface it crosses
matters. Left to itself, the transport layer will happily pick up a container
bridge (`docker0`, `br-*`, `flannel.1`) and advertise an address the peer cannot
reach — which fails as a connector error at handshake time, not as a
configuration error.

GPUStack derives the interface per worker:

- **One routable interface** — it is used. Nothing to configure.
- **Several routable interfaces** — GPUStack refuses to guess. Set
  `--kv-transfer-ifname` (or the `kv_transfer_ifname` worker config key, or
  `GPUSTACK_KV_TRANSFER_IFNAME`) on that worker to the interface carrying KV traffic. The
  worker log lists the candidates it found.

Set it explicitly on any machine with more than one fabric — a management NIC
plus an RDMA NIC is the common case, and the management NIC is almost never the
right answer.

### Ports

Every role needs its own HTTP port plus the ports its connector requires, and
GPUStack allocates them from the worker's service port range as contiguous
bands so that two members on one host cannot collide.

The default range holds **64 ports**, which is enough for most deployments but
not for all: a tensor-parallel-8 member using a connector with a per-rank port
consumes nine of them. If you run several such members per host, widen
`--service-port-range`.

!!! warning "Three kinds of port GPUStack cannot manage"

    Some connectors bind ports that are chosen at run time by the engine and
    are not declared anywhere GPUStack can read:

    - SGLang's rank port, taken from the host's ephemeral range;
    - Mooncake's per-rank transfer-engine handshake ports;
    - Mooncake's segment ports.

    For these, the value written in a configuration file is **not** the port
    actually bound, so GPUStack can neither reserve them nor declare them as
    host ports. They have been observed around 15000–17000 and 20000–20800,
    which is why the default service range (40000–40063) does not meet them —
    the separation is what keeps them apart, not any reservation.

    **So do not move `--service-port-range` into those bands.** Measured:
    with a connector port band placed at 20001, a member's own second rank
    tried to bind 20003 and found it already taken by a transfer-engine port
    the engine had chosen for itself moments earlier. The same deployment
    started first try with the band in the default range.

    If a member restarts repeatedly and never reaches `Running`, check its
    engine log for `Address already in use`.

    Port deduplication also only covers one worker process. A second GPUStack
    worker on the same host, or a container you started by hand, is outside it.

## Constraints

Configuration that would produce a group serving wrong answers is rejected at
save time rather than at run time:

- **Prefill and decode must declare the same context length.** No engine checks
  this, and a mismatch is invisible until a long prompt arrives.
- **Decode's tensor parallelism must be at least prefill's.** The engine does
  assert this, but reports it as an internal error inside decode rather than as
  a configuration problem.
- **Data type, KV cache data type, block size and KV cache layout must agree**
  between the two sides.
- **A role's engine must be one the chosen mode can configure.** Mixing engines
  across roles requires the `custom` mode, where the connection parameters are
  yours to supply.
- **A disaggregated deployment using the `custom` mode cannot also use an
  extended KV cache.** Every other mode composes the two for you (see
  [Using a shared KV cache with it](#using-a-shared-kv-cache-with-it));
  `custom` injects no connector configuration at all, so there is nothing to
  compose the cache into — write the combined configuration yourself.
- **Scheduled scaling and disaggregation cannot be combined.**

For `vllm-nixl`, compatibility checks include role-specific environment variables.
`VLLM_KV_CACHE_LAYOUT` accepts equivalent aliases (`NHD`/`LBNHC` and
`HND`/`LBHNC`). On MLA models, `VLLM_MLA_DISABLE` must produce compatible latent
KV representations. NIXL's recurrent-state transfer requires
`VLLM_SSM_CONV_STATE_LAYOUT=DS` on supported runtimes. For chunked local attention,
`VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE` is checked together with the
HMA flags. Explicit attention backend choices are also compared.

A role with an unset environment inherits the deployment's environment; an
explicit dictionary replaces it, including an empty dictionary. Model-dependent
checks use the source configuration and role overrides. Unknown configurations,
runtime versions and transfer layouts are reported through `pairing_unverified`.
Network addresses, ports, device names and performance settings may differ between
roles. Experimental layout permutation remains subject to the engine's direction
and HMA restrictions.

## Limitations

- The router runs as a single replica.
- **A `custom` router is deployed exactly as written.** The built-in modes set
  the upstream router's circuit breaker to its default threshold of ten failed
  requests; a router you supply keeps whatever defaults it ships with.
- **On the vLLM modes, a prefill member that returns errors is not taken out
  of rotation.** The upstream router treats only a transport error as a
  prefill failure, so an HTTP 500 from prefill is logged and the request
  proceeds to decode and returns 200 — measured with a stub prefill failing
  every one of twelve requests, none of which opened the circuit. The SGLang
  modes do feed the breaker. Either way, the **PD Effectiveness** figure in
  the group's panel is what detects a member that has stopped contributing;
  the router's own request counters will not show it.
- **A member that dies while the group is idle is noticed on the next request,
  not before it.** Detection rides the request path, so with no traffic there
  is nothing to detect on; the router's background sweep runs once a minute.
  The first requests after an idle period are retried and then routed around
  the dead member.
- All members of a group must use the same GPU type; a group mixing card types
  cannot be admitted atomically.
- Roles are limited to prefill, decode and router.
- **Prefix-aware routing is used to pick prefill, and not to pick decode.**
  Prefill is where a shared prefix pays off, so the built-in modes route it by
  approximate prefix match; decode holds a request for its whole generation,
  so it is balanced round-robin to distribute requests. All built-in modes,
  including `vllm-ascend-mooncake`, explicitly set `--prefill-policy cache_aware`
  and `--decode-policy round_robin`. Override these in the router role's
  parameters when a different routing policy is needed.
