# Catalog model configuration fixtures

The matrix contains 30 different root `architectures` from the packaged catalog
as of 2026-10-29. `manifest.json` records the catalog checksum, model release dates,
source repositories and revisions, fixture checksums, and expected evaluation fields.
Each JSON snapshot comes from:

`https://huggingface.co/<repo_id>/resolve/<revision>/config.json`

Selection follows `release_date` descending, preserving catalog order for ties,
and keeps the first model for each distinct root architecture. Recipes prefer
vLLM, non-Ascend hardware, and standard mode, in that order.

This matrix exercises pretrained configuration inspection. Image-category
diffusion pipelines use a separate `model_index.json` path and are outside this
matrix. Voxtral-4B-TTS-2603, within the selected date range, has no root
`config.json` on either catalog source and is excluded. DiffusionGemma is included
because its language model uses the pretrained configuration path.

Tests run offline and reject repository Python loading. Scheduling tests use
catalog backend parameters, a simulated worker, and a fixed 20 GiB weight size.
This isolates metadata behavior; it does not establish whether the actual
checkpoint fits on that hardware.

## Updating fixtures

Update each JSON snapshot together with its source revision and SHA-256 checksum
in `manifest.json`. When refreshing the model selection, also update the selection
date and catalog checksum using the rules above.

Review expected fields against the source JSON and the installed Transformers
configuration semantics. Do not regenerate expectations from the loader under
test. Run the offline checks from the repository root:

```sh
uv run pytest tests/scheduler/test_catalog_model_configs.py tests/utils/test_model_config.py
```
