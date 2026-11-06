# Model catalog automation

The **Model Catalog Sync** GitHub Actions workflow accepts up to ten model IDs and
opens one draft PR against `main`. Inputs may be comma/whitespace-separated IDs
(`hf:org/model`, `ms:org/model`, bare IDs defaulting to HF) or repository URLs.
Both hubs are investigated independently. Enable `dry_run` to keep the validated
bundle, binary diff and report without publishing a PR. Run the workflow from `main`.

## Repository configuration

Configure these Actions variables and secrets before the first run:

| Kind | Name | Value |
| --- | --- | --- |
| Variable | `CI_GPUSTACK_CATALOG_LLM_URL` | OpenAI-compatible base URL, including `/v1` if required by the provider |
| Variable | `CI_GPUSTACK_CATALOG_LLM_MODEL` | Model ID supporting tool use and 32K output tokens |
| Secret | `CI_GPUSTACK_CATALOG_LLM_AUTH_TOKEN` | LLM API credential |
| Secret | `CI_PRT_GENERATOR_ID` | GitHub App ID, following runner auto-sync's convention |
| Secret | `CI_PRT_GENERATOR_KEY` | GitHub App private key |

Install the App on this repository with **Contents: write** and **Pull requests:
write**. The App token lets normal PR checks run on generated PRs. No App credentials
are needed for `dry_run`; the LLM credential is still required. The workflow does not
merge PRs, push directly to `main`, or create issue-closing references.

Check these names under **Settings > Secrets and variables > Actions** in this
repository. Organization-level secrets or variables must grant this repository
access. The workflow declares no GitHub environment, so environment-only secrets
are not available to its jobs. `GITHUB_TOKEN` is supplied by Actions; no personal
access token, Hugging Face token or ModelScope token is required.

Workflow Actions are pinned to full commit SHAs. The Qwen Code 0.25.0 archive is
SHA-256 checked before installation. Research
gets an isolated settings directory, the catalog skill, selected repository files
and packaged runner records. Its explicit subprocess environment includes no GitHub
token, cloud credentials or SSH agent socket. Each session is
limited to 80 turns and 120 tool calls, with at most one schema repair and a shared
40-minute deadline. These are execution limits, not an exact billing cap. Research
checks public DNS for evidence hosts before calling the LLM. Full agent logs are not
uploaded; a failed process preserves its terminal structured error with the LLM
credential redacted as `research-error.txt`, excluding reasoning and tool transcripts.

### Research trust boundary

This workflow uses the runner auto-sync approach of separating research, validation
and publication privileges. The research process still receives the LLM API token
and can execute commands and access public network destinations. Temporary settings,
file permissions, prompt instructions and log redaction are not a security sandbox:
malicious upstream content could induce credential access or exfiltration during
research. Independent validation protects publication, not the research credential.

Use a dedicated, revocable catalog LLM key with a provider-enforced spending limit;
restrict models and request rates where the provider supports it. Execution deadlines
and turn limits do not substitute for that spending limit. The residual risk must
be accepted by the repository maintainers before enabling research. Local preview
also runs as your OS user and can access that user's files; omitting credentials
from its environment does not make those files inaccessible.

Each session starts with private settings recreated by the controller and removes
workspace startup configuration before a repair attempt. Ambient Git configuration,
hooks and interactive credential prompts are disabled. On Linux, a dedicated child
subreaper cleans up detached descendants on completion or timeout. On macOS, cleanup
is limited to the research process group; use Linux when detached-child cleanup is
required. These measures bound normal operation, not malicious code execution.

## Review and validation

The reusable policy is
[`model-catalog-update`](../../.github/skills/model-catalog-update/SKILL.md).
It prioritizes official parameter counts, broadly compatible deployments, NVIDIA
and Ascend 910B. NVIDIA compute capability splits are only for evidenced constraints.
Missing icons prefer the publisher's verified official GitHub organization avatar.
The skill links to vLLM Recipes and vLLM Ascend model tutorials and support tables;
examples must match the selected runner or explicit official image and target hardware.

Research emits a typed proposal. A fresh job at the frozen `main` revision rebuilds
the input request and independently checks:

- Both hubs have explicit assessments; requested IDs are present in proposed specs.
- Quotes are retrievable; release dates, sizes and catalog fields are well formed.
- Selected weight file sizes, content IDs, config and index metadata match the
  research snapshot. Alternate formats are not summed; incomplete indexes and
  ambiguous additional weight files block the target. Safetensors indexes, including
  `consolidated.safetensors.index.json`, have a separate 64 MiB read limit. Artifacts
  retain their metadata, shard list, tensor count and a digest of the full index.
- vLLM, SGLang and MindIE proposals use full-repository downloads. Nonempty filename
  filters are rejected because collecting configuration as evidence does not include
  it in the worker's filtered download. File-based backends can still select weights.
- Official parameter counts take precedence over calculated counts. The automatic
  fallback uses HF's unquantized safetensors parameter metadata; unsupported layouts
  need official evidence or remain blocked. Weight bytes are reported separately.
- A compatible packaged runner or an explicit official recipe image exists and
  its registry manifest can be read. The agent must cite architecture, quantization
  and backend parameter support for that runner. Packaged runner specs pin
  `backend_version` to the evidenced
  runner's service version, preventing host-dependent automatic selection of a
  different backend version. Official recipe images use `image_name`, exactly
  matching the proof and quoting the tag in deployment evidence; `backend_version`
  is optional for this path. The validator restricts image repositories to the
  backend publishers listed in the skill. Mutable tags may change after validation;
  a tag plus digest can be used when an immutable reference is needed.
  This does not guarantee runtime/driver compatibility on every host.
  NVIDIA/910B/910C matching checks detect shadowed specs.
- Icons are local assets; new raster images are checked and converted to PNG.
- Only the target cards and new icons change; shared anchors and unrelated cards
  are preserved. Same-name replacements must share a repository ID with the existing
  card in that hub; unrelated models need distinct names. Repository migrations with
  no shared ID require manual review outside automatic replacement. Offline tests
  and `make lint` run before publication.

Quote presence does **not** prove semantic correctness. Review the source links for
release-date interpretation, official parameter statements, mirror identity and
backend compatibility. CPU CI does **not** perform GPU/NPU inference. Generated PRs
remain drafts for review and hardware testing. A public repository's absence is
different from a timeout or access restriction; unverified mirrors are reported.

The skill also covers custom RAM/VRAM claims via supported model environment
variables. Prefer normal resource estimation; overrides require evidence, byte units
and explicit workload assumptions. Their adequacy remains part of deployment review,
and cannot be established from weight size alone.

Each fully verified target is atomic across its proposed entries. If one target
fails, other verified targets can still enter the PR, and blocked targets remain in
the report. A batch with no verified changes and blocked targets fails without a PR.
An all-unchanged batch succeeds without a PR. Concurrent HF/MS aliases resolving to
the same card are rejected rather than letting one overwrite another; submit one
input ID for each checkpoint.

Publication verifies the validated artifact digest and unchanged base revision
before creating a write token. The publisher verifies the bundle again and uses the
GitHub Git data API to create a commit whose sole parent is the frozen base, changing
only validated paths. It never rebases the generated changes. It rechecks `main`
before creating the branch and immediately before opening the draft PR. If `main`
advances, rerun against the new base; a branch already created before that failure
can remain without a PR. There is no lock on `main`, so normal PR checks and review
must still assess later base changes before merge. A new run/attempt uses a new
branch; existing branches and manual review edits are never overwritten. Close
superseded draft PRs manually.

Artifacts (`catalog-proposal-*`, `catalog-validated-*`) are retained for 30 days.
The validated artifact includes `report.json`, `pr-body.md`, `catalog.patch`, and the
digest-protected file bundle. Copy durable supporting evidence into the PR discussion
when it must outlive artifact retention.

### Correcting review findings

The workflow runs only on manual dispatch from `main`. It does not consume PR
comments, reviews or approval events, and does not automatically revise a proposal.

- For a small correction, edit the catalog YAML or icon on the generated PR branch
  and push a commit. Update the PR description with the corrected evidence and
  rationale, since its original report describes the generated proposal. Run the
  catalog tests and lint below; normal PR checks follow their own triggers. These
  checks do not rerun the LLM or independently verify the new evidence.
- To repeat evidence validation without another LLM call, correct a copy of
  `proposal.json`, including the field and its citations, and rerun `validate` on
  the frozen base commit using the matching request. Use a new output directory,
  as in the retry example below. Review its diff and transfer the intended changes
  and updated evidence to the PR branch. Do not edit `validated.json` or apply a
  bundle over a different commit; publication checks its digest and base.
- For a recurring research error, update the skill or proposal instructions and
  merge that policy change into `main` before a new manual run. A rerun creates a
  new branch and draft PR, does not include the previous review comments, and does
  not update the existing PR. Close superseded drafts manually.

For example, if review finds that an FP16 checkpoint's release date was taken from
an earlier GGUF preview, correct the date and cite the FP16 release announcement.
Merely rerunning the same research does not guarantee that interpretation changes.

After manual edits, run:

```sh
uv run pytest tests/hack/test_catalog_sync.py tests/hack/test_catalog_sync_review.py tests/server/test_catalog.py tests/server/test_catalog_accessors.py
make lint
```

Keep the PR in draft until evidence review and any required hardware testing are
complete, then mark it ready and merge through the normal repository review process.

## Local use

Run from the repository root on macOS or Linux. The full research flow requires
Node.js 24, a tool-capable LLM endpoint, and the Docker CLI with registry access for
validation. It needs no GitHub App credentials and does not create a PR. Keep the
same Git commit checked out between `prepare` and `validate`.

Start with the offline tests and optionally inspect public metadata. These collector
commands do not call an LLM, load model code, or download full weights:

```sh
uv sync --locked
env -u HF_TOKEN uv run pytest tests/hack/test_catalog_sync.py tests/hack/test_catalog_sync_review.py tests/server/test_catalog.py
uv run python -m hack.catalog_sync.hubs huggingface Qwen/Qwen3-0.6B
uv run python -m hack.catalog_sync.hubs model_scope Qwen/Qwen3-0.6B
```

To exercise the actual LLM flow, install the pinned CLI into a temporary directory,
configure the endpoint, and enter the token without putting it in shell history.
Replace the example endpoint, model and target IDs with your own values. Choose a
checkpoint not already in the catalog if you want to test a new entry; existing
checkpoints may correctly produce an unchanged result.

```sh
export CATALOG_TEST_DIR="$(mktemp -d /tmp/gpustack-catalog.XXXXXX)"
npm install --prefix "$CATALOG_TEST_DIR/tools" --ignore-scripts @qwen-code/qwen-code@0.25.0
export PATH="$CATALOG_TEST_DIR/tools/node_modules/.bin:$PATH"
qwen --version

export CATALOG_LLM_URL='https://your-provider.example/v1'
export CATALOG_LLM_MODEL='your-tool-capable-model'
printf 'LLM API token: '
read -r -s CATALOG_LLM_TOKEN
printf '\n'
export CATALOG_LLM_TOKEN
export CATALOG_MODEL_IDS='hf:org/model,ms:publisher/another-model'

uv run python -m hack.catalog_sync.run preview \
  --output "$CATALOG_TEST_DIR/run" && \
  cat "$CATALOG_TEST_DIR/run/validated/pr-body.md"
unset CATALOG_LLM_TOKEN
```

`preview` runs prepare, research and validation in order and stops at the first
failure. Use a new empty output directory for every attempt to avoid stale results.
It produces files under `CATALOG_TEST_DIR/run` without applying catalog changes.
Review `validated/report.json` for per-target blocked reasons and
`research/research-error.txt` if the CLI process failed. `changed: false` may mean
all targets are unchanged or blocked; inspect `partial` and the report.

The terminal error includes provider failures such as HTTP 429 with insufficient
balance. Refill the provider account or select an endpoint with quota before retrying;
raising turn limits does not address insufficient balance. DNS preflight failures
stop before an LLM request. These are separate from the provider's quota errors.

To apply the verified bundle to your working tree for a diff, run this optional step.
It checks the recorded base and file hashes before writing:

```sh
export CATALOG_BUNDLE_SHA256="$(uv run python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["sha256"])' \
  "$CATALOG_TEST_DIR/run/validated/result.json")"
uv run python -m hack.catalog_sync.run apply \
  --output "$CATALOG_TEST_DIR/run/validated"
git diff -- gpustack/assets/model-catalog.yaml gpustack/assets/model-catalog-modelscope.yaml
git status --short
```

All tests mock network and registry boundaries. The `validate` command is the
separate, explicitly networked acceptance step. HTTP evidence fetching requires
public HTTPS destinations and rejects credentials in URLs or private-address DNS
responses. Gated/private models are not supported by this public-catalog workflow.

### Local fake-IP proxies

If your trusted local TUN proxy resolves public hosts into `198.18.0.0/15`, you can
keep fake-IP mode and explicitly list the evidence hosts you intend to access:

```sh
export CATALOG_FAKE_IP_HOSTS='huggingface.co,cdn-thumbnails.huggingface.co,modelscope.cn,github.com,api.github.com,raw.githubusercontent.com,avatars.githubusercontent.com'
```

Set this in the same terminal before `preview`, `research`, standalone collection
or `validate`. The research subprocess inherits this specific option. If another
official documentation host or redirect is blocked, inspect the hostname in the
error and append that exact host. Wildcards, URLs, ports and IP literals are rejected.
The exception permits only the fake-IP subnet for the listed hosts; loopback,
private LAN, link-local and other reserved addresses remain blocked. Each redirect
is checked separately, and HTTPS certificate validation stays enabled.

If research produced a ready proposal but validation was blocked by an evidence or
icon host, append that host and rerun only `validate`. This reuses the proposal
without spending another LLM request. Keep the original base commit checked out:

```sh
uv run python -m hack.catalog_sync.run validate \
  --request "$CATALOG_TEST_DIR/run/request/request.json" \
  --proposal "$CATALOG_TEST_DIR/run/research/proposal.json" \
  --output "$CATALOG_TEST_DIR/run/revalidated" && \
  cat "$CATALOG_TEST_DIR/run/revalidated/result.json"
```

Use the `revalidated` directory for the report and any subsequent `apply` step.
`preview` prints each blocked target's reason directly in the terminal as well.

This option is rejected in GitHub Actions. Use real public DNS there. The collector
does not inherit HTTP proxy environment variables; this option supports transparent
TUN routing, not an explicit HTTP/SOCKS proxy. Alternatively, configure the proxy
to use real DNS for evidence hosts instead of enabling this exception.
