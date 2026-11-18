# Proposal contract

CI supplies `request.json`, `proposal-schema.json`, `runners.json`, both catalogs and
the collector. Write `proposal.json` using the generated JSON Schema. Local users can
generate it with `Proposal.model_json_schema()` from `hack.catalog_sync.models`.

Top level: `schema_version: 1`, `candidates: [...]`. Each candidate has:

- `target`: the exact normalized ID from the request, e.g. `hf:publisher/model`.
- `status`: `ready`, `blocked`, or `unchanged`; `reason`: specific findings.
- `hubs`: exactly `huggingface` and `model_scope`, each with `status` (`available`,
  `unavailable`, `unverified`), `repo_id` (or null), `reason`, and optional `evidence`.
  An available hub included in edits must identify one of that card's spec repos.
- `ascend_910b`: support findings, recipe choice or why it cannot be added.
- `entries`: zero for blocked/unchanged; one per hub being changed for ready.
  The requested repository must occur in a proposed spec. Do not silently substitute
  another checkpoint. Put cross-hub identity evidence in `identity`.

Each entry contains:

- `source`: `huggingface` or `model_scope`.
- `card`: the complete target model-set mapping, including `specs`, using existing
  catalog fields. Name, description, licenses, categories, size, release date and
  local icon are required. `release_date` is an ISO string. Do not include IDs,
  ownership, commands, or server-only fields. Official recipe images may be set
  through `image_name` (see the deployment selection rules in SKILL.md).
  vLLM, SGLang and MindIE specs must omit nonempty `huggingface_filename` and
  `model_scope_file_path`; their collectors use no pattern and deployment downloads
  the full repository, including configuration and tokenizer assets.
  Packaged runner specs must include `backend_version` equal to the evidenced
  `service_version`. With an explicit official `image_name`, `backend_version`
  is optional and `runner_image` must exactly match `image_name`; a deployment
  citation must quote the image tag. An existing card name can only be reused when at least one spec retains a repository ID from that card
  in the same hub.
- `parameter_method`: `official` (preferred) or `safetensors_metadata`.
- `parameter_notes`: publisher's stated total/active size, or sources checked and the
  fallback calculation. Calculated values must equal collected safetensors parameters
  in the declared M/B/T unit; do not round them to a marketed model size.
- `evidence`: mapping from field to nonempty citation list. Required: `identity`,
  `size`, `release_date`, `licenses`; also `capabilities` and `activated_size` when
  present. For a calculation, `size` citations establish checkpoint/architecture and
  explain absence of an official count.
- `specs`: one proof per `card.specs` item, in the same order: `snapshot`, `revision`
  from the collector; `deployment` and `quantization` citation lists; `runner_image`
  from the packaged catalog or the explicit official recipe image; `hardware`
  (`nvidia`, `ascend:910b`, `ascend:910c`);
  `compatibility` describing the broad recipe, backend version assumptions, flags
  and any necessary restrictions. The 910C hardware label maps to CANN variant `a3`.
  Resource claim overrides in `card.specs[].env` need supporting `deployment`
  citations and workload/memory assumptions in `compatibility`; express claims as
  decimal integer strings in bytes, not weight file sizes.
- `icon`: null to reuse an existing asset, or `filename` (new PNG basename), `url`,
  `github_org` (or null for a documented official fallback), `evidence`, and `reason`.
  The card's icon must be `/static/catalog_icons/<filename>`.

A citation is `{ "url": "https://...", "quote": "verbatim source excerpt",
"explanation": "what this excerpt establishes" }`. Excerpts must be at least 12
characters, compact and independently retrievable as text. The validator checks
that excerpts exist; semantic correctness remains part of PR review.

Collector examples (the CI prompt supplies the Python executable):

```sh
python -m hack.catalog_sync.hubs huggingface publisher/model
python -m hack.catalog_sync.hubs model_scope publisher/model --revision master
python -m hack.catalog_sync.hubs huggingface publisher/model-GGUF --pattern 'model-Q8_0.gguf'
```

Copy `snapshot` and `revision` exactly; do not invent a digest. Keep patterns aligned
with `huggingface_filename`/`model_scope_file_path`. Ambiguous multi-component layouts
without a complete selection remain blocked. Validation uses public access; gated
or private repositories that cannot be independently inspected are unsupported.
