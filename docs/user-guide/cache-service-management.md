# Cache Service Management

A cache service is a shared KV cache that model deployments attach to instead of keeping their KV cache to themselves. The cache lives outside the inference engine, so deployments and replicas on the same worker reuse each other's prefixes, and an L2 storage backend extends that reuse across restarts and across nodes.

Cache services attach to the built-in vLLM and SGLang backends only, and a deployment can only attach to a cache service in its own cluster.

## Create a Cache Service

1. Navigate to the `Cache Service` page.
2. Click the `Add Cache Service` button and select a provider.
3. Enter a `Name`.
4. Select a `Version`, or select `Custom` and enter a `Container Image`.
5. Fill in the provider settings, such as `RAM Size (GiB)` — the cache capacity held in the cache server's memory.
6. _(Optional)_ Set a `Worker Label Selector` to limit which workers run the service.
7. _(Optional)_ Add an `L2 Storage Backend` to spill cache to a larger tier (a local directory, Redis or Valkey, or an S3-compatible object store). Several entries are prioritized in order: reads prefer the first, writes go to all.
8. _(Optional)_ Under `Advanced`, add `Parameters` and `Environment Variables`. A service running more than one component keeps a separate parameter list per component, since each one runs a different binary.
9. Click the `Save` button.

!!! tip

    Changes to an existing cache service take effect only after its instances are deleted and recreated.

## Attach a Deployment

1. Create or edit a model deployment using the built-in vLLM or SGLang backend.
2. Select `Enable Extended KV Cache`.
3. Set `Cache Backend` to the cache service. Only cache services in the same cluster and compatible with the selected backend are listed.
4. Click the `Save` button.

Each instance reports the cache service it attached to and its recent hit rate. An instance that started without the cache — because no cache instance was available on its worker, for example — reports the shared KV cache as not active rather than failing.

## Running Hybrid Attention Models with LMCache

A hybrid model interleaves recurrent layers — Mamba, or Gated-DeltaNet (GDN) linear attention — with full-attention layers. The Qwen3.5 and Qwen3.6 series, Qwen3-Next, Kimi-Linear and Kimi K3 are all of this kind.

A recurrent layer holds a fixed-size state instead of one key/value pair per token, and that state can only be snapshotted where a block ends. The engine therefore works in blocks far larger than the usual handful of tokens, sized from the model itself, and LMCache can serve the model only if it stores cache in units those block boundaries agree with. Neither side arranges this on its own: left as they come, the engine either refuses to start or attaches and never records a hit.

### Step 1: Find the block size the engine chose

vLLM raises the attention block size until an attention page is at least as large as a recurrent-state page, and logs the result when the deployment starts:

```
Setting attention block size to 544 tokens to ensure that attention page size is >= mamba page size.
```

That number is `N`. It follows from the model, the data type and the parallelism, so read it from the deployment's own log rather than assuming it. The values below were read from single-GPU runs at the model's own precision — a sanity check on what you see, not a lookup table:

| Model                                     | `N` |
| ----------------------------------------- | --- |
| `Qwen/Qwen3.5-0.8B`                       | 544 |
| `Qwen/Qwen3.6-27B`                        | 784 |
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 944 |
| `moonshotai/Kimi-K3`                      | 768 |

### Step 2: Configure the cache service

- Set `Chunk Size` to `N`, or to a multiple of it. The cache server's chunk size must be a multiple of the engine's block size or cache registration fails.
- Add `--separate-object-groups` to the cache server's `Parameters`, so each attention window is stored as its own object group. The server keeps this off by default.

!!! note

    Chunk size belongs to the cache service, not to a deployment. A service tuned for one `N` still serves ordinary models whose block size divides it, but two hybrid models with different `N` need a cache service each.

### Step 3: Deploy the model against the service

Deploy the model as in [Attach a Deployment](#attach-a-deployment), choosing the cache service from step 2 as its `Cache Backend`. A running instance reports the service it attached to and the hit rate it is seeing.

### Reference

[Hybrid Attention Models](https://docs.lmcache.ai/mp/hybrid_models.html) — LMCache's own account of the same requirement, and the flags its server takes for it.

## Customize Cache Providers

The providers you can pick from are the ones this release carries, plus any an installed extension adds. Platform admins can replace that set with a document of their own — to pin a different image, add a version, or declare a provider GPUStack does not ship.

1. Navigate to the `Cache Service` page.
2. Click the `Manage Providers` button.
3. In the `Provider Source` drawer, choose one of the three sources:

    - `Yaml File`: paste the content into the editor, or click `Import` to load a file into it.
    - `URL`: fill in `Source URL`. The server fetches it when you save, and again when you click `Update Now`. It must be `http(s)`, must not carry credentials, must point at the file itself rather than at a repository page, and must be UTF-8 and no larger than 4 MB.
    - `Embedded`: go back to the providers this release carries.

4. Click the `Save` button.

!!! warning

    **Start from the built-in file, not from an empty editor.** A document of your own replaces the whole set outright — every provider it does not declare goes out of service, including the ones an installed extension contributes. Click `Built-in File` in the editor header to download what this installation currently serves, and edit that.

Your document is validated when you save it, and refused as a whole if anything in it will not serve — a declaration this version cannot read, a placeholder that would render literally, or a field name that does not exist. The last one matters more than it looks: an unknown key is simply ignored, so a component whose `run_command` was spelled `run_cmd` would otherwise launch with no command at all.

A save is also refused when it would take away a provider, or the version, that a cache service still runs. The message names every service affected. Delete those services first, or move them onto something the new document carries.

!!! note

    Running instances keep the declaration they started with. A new one takes effect when an instance is recreated, the same way a provider changes across a GPUStack upgrade.

    Nothing updates the providers on a schedule: a URL source is read when you save it and when you click `Update Now`. While a document of your own is configured, providers added by a GPUStack upgrade or by a newly installed extension do not appear until you fold them in — download the built-in file again to see what they added.
