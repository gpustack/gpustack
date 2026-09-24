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

Scaling a role is different and does **not** require a restart — on a
deployment with nothing pending. A replica count is not part of the shape the
group is pinned to, so adding a prefill adds a prefill.

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

The order differs per role, and it is the order that makes the pair worth
having:

- **Prefill asks the cache first.** A prefix the cache already holds is prefill
  work that does not have to happen at all, and what remains is the part
  disaggregation exists to speed up. The KV it does compute is written back to
  the cache *and* handed to decode in the same step.
- **Decode asks its own prefill first.** The request already carries the
  handshake saying its KV is waiting there; the cache is the fallback behind
  it.

This also widens where disaggregation pays. Its usual objection is that a
workload with a high prefix hit rate turns prefill memory-bound, which removes
the asymmetry the split is built on — attaching a cache on the prefill side
removes the hit part of that cost entirely, leaving the part the split actually
helps with.

Which roles take a cache is per role, so attaching one only to prefill is a
supported and often sufficient configuration.

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
  so it is balanced round-robin to keep the load even. The `vllm-ascend-mooncake`
  mode keeps the router's own defaults for both.
